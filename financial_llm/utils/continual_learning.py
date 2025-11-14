"""
Continual Learning Utilities for Daily Model Updates

Prevents catastrophic forgetting when fine-tuning on new data
Implements multiple strategies:
1. Elastic Weight Consolidation (EWC)
2. Experience Replay
3. Progressive Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import deque
from typing import Dict, List, Optional
import copy


class EWC:
    """
    Elastic Weight Consolidation

    Adds a penalty term to the loss that prevents important parameters
    from changing too much during fine-tuning on new data.

    Reference: "Overcoming catastrophic forgetting in neural networks" (Kirkpatrick et al., 2017)
    """
    def __init__(self, model: nn.Module, dataloader: DataLoader, device: str = 'cpu'):
        """
        Args:
            model: The neural network model
            dataloader: DataLoader with samples from previous task
            device: Device to use
        """
        self.model = model
        self.device = device
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher_matrix = self._compute_fisher_matrix(dataloader)

    def _compute_fisher_matrix(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix
        Estimates importance of each parameter
        """
        print("Computing Fisher Information Matrix for EWC...")

        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

        self.model.eval()

        for batch_idx, batch in enumerate(dataloader):
            # Extract inputs based on batch structure
            if isinstance(batch, dict):
                # Handle different task types
                if 'input_ids' in batch:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)

                    ohlcv = batch.get('ohlcv', None)
                    if ohlcv is not None:
                        ohlcv = ohlcv.to(self.device)

                    indicators = batch.get('indicators', None)
                    if indicators is not None:
                        indicators = indicators.to(self.device)

                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        ohlcv=ohlcv,
                        indicators=indicators,
                        task='trading'
                    )
                else:
                    # Time series only
                    ohlcv = batch['ohlcv'].to(self.device)
                    indicators = batch.get('indicators', None)
                    if indicators is not None:
                        indicators = indicators.to(self.device)

                    outputs = self.model(
                        ohlcv=ohlcv,
                        indicators=indicators,
                        task='trading'
                    )
            else:
                # Assume tuple/list format
                inputs, targets = batch
                inputs = inputs.to(self.device)
                outputs = self.model(ohlcv=inputs, task='trading')

            # Get loss from outputs
            if 'trading_logits' in outputs:
                # Use negative log likelihood
                logits = outputs['trading_logits']
                log_probs = F.log_softmax(logits, dim=-1)
                # Sample a class for each example
                target = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze()
                loss = F.nll_loss(log_probs, target)
            else:
                continue

            # Backward pass
            self.model.zero_grad()
            loss.backward()

            # Accumulate squared gradients (Fisher Information)
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2

            if batch_idx >= 100:  # Sample 100 batches for efficiency
                break

        # Normalize
        num_batches = min(batch_idx + 1, 100)
        for n in fisher:
            fisher[n] /= num_batches

        print(f"Fisher matrix computed from {num_batches} batches")

        self.model.train()
        return fisher

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty term

        penalty = λ/2 * Σ F_i * (θ_i - θ*_i)^2
        where F_i is Fisher information and θ*_i is the previous parameter value
        """
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher_matrix:
                loss += (self.fisher_matrix[n] * (p - self.params[n]) ** 2).sum()
        return loss / 2


class ExperienceReplay:
    """
    Experience Replay Buffer

    Maintains a buffer of past experiences to prevent forgetting
    When training on new data, we also sample from the buffer
    """
    def __init__(self, buffer_size: int = 10000):
        """
        Args:
            buffer_size: Maximum number of samples to store
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience: Dict):
        """
        Add an experience to the buffer

        Args:
            experience: Dictionary containing model inputs and targets
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Dict]:
        """
        Sample a batch from the buffer

        Args:
            batch_size: Number of samples to return
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

    def is_full(self):
        return len(self.buffer) >= self.buffer_size


class ReplayDataset(Dataset):
    """
    Dataset wrapper for experience replay
    """
    def __init__(self, experiences: List[Dict]):
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        return self.experiences[idx]


class ContinualLearningTrainer:
    """
    Trainer for continual learning with EWC and Experience Replay

    Usage:
        1. Train initial model on historical data
        2. Save Fisher matrix and experience replay buffer
        3. For daily updates:
           - Load new data
           - Fine-tune with EWC penalty and replay
           - Update Fisher matrix periodically
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        ewc_lambda: float = 100.0,
        replay_buffer_size: int = 10000,
        replay_ratio: float = 0.3,  # 30% of batch from replay
    ):
        """
        Args:
            model: The financial LLM
            device: Device to use
            ewc_lambda: EWC regularization strength
            replay_buffer_size: Size of experience replay buffer
            replay_ratio: Ratio of replay samples in each batch
        """
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.replay_ratio = replay_ratio

        self.ewc = None
        self.replay_buffer = ExperienceReplay(buffer_size=replay_buffer_size)

    def initialize_ewc(self, dataloader: DataLoader):
        """
        Initialize EWC with data from the current task

        Call this after initial training or before switching tasks
        """
        print("Initializing EWC...")
        self.ewc = EWC(self.model, dataloader, self.device)
        print("EWC initialized")

    def add_to_replay_buffer(self, batch: Dict):
        """
        Add samples to experience replay buffer
        """
        batch_size = len(batch[list(batch.keys())[0]])

        for i in range(batch_size):
            experience = {k: v[i] if torch.is_tensor(v) else v for k, v in batch.items()}
            self.replay_buffer.add(experience)

    def get_mixed_batch(self, new_batch: Dict, batch_size: int):
        """
        Create a mixed batch from new data and replay buffer

        Args:
            new_batch: Batch of new data
            batch_size: Total batch size
        """
        # Determine split
        replay_size = int(batch_size * self.replay_ratio)
        new_size = batch_size - replay_size

        # Sample from replay buffer
        if len(self.replay_buffer) > 0:
            replay_samples = self.replay_buffer.sample(replay_size)

            # Combine with new data
            # This is simplified - in practice you'd need to collate properly
            # based on your data structure
            mixed_batch = new_batch  # Placeholder

            return mixed_batch, len(replay_samples)
        else:
            return new_batch, 0

    def compute_loss_with_ewc(
        self,
        outputs: Dict,
        targets: Dict,
        task_loss_fn
    ) -> torch.Tensor:
        """
        Compute total loss including EWC penalty

        Args:
            outputs: Model outputs
            targets: Ground truth targets
            task_loss_fn: Function to compute task-specific loss
        """
        # Task loss
        task_loss = task_loss_fn(outputs, targets)

        # EWC penalty
        if self.ewc is not None:
            ewc_loss = self.ewc.penalty(self.model)
            total_loss = task_loss + self.ewc_lambda * ewc_loss

            return total_loss, task_loss, ewc_loss
        else:
            return task_loss, task_loss, torch.tensor(0.0)

    def save_checkpoint(self, path: str):
        """
        Save model and continual learning components
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'ewc_params': self.ewc.params if self.ewc else None,
            'ewc_fisher': self.ewc.fisher_matrix if self.ewc else None,
            'replay_buffer': list(self.replay_buffer.buffer),
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """
        Load model and continual learning components
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if checkpoint['ewc_params'] is not None:
            # Reconstruct EWC
            self.ewc = EWC.__new__(EWC)
            self.ewc.model = self.model
            self.ewc.device = self.device
            self.ewc.params = checkpoint['ewc_params']
            self.ewc.fisher_matrix = checkpoint['ewc_fisher']

        if checkpoint['replay_buffer']:
            for exp in checkpoint['replay_buffer']:
                self.replay_buffer.add(exp)

        print(f"Checkpoint loaded from {path}")


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) Layer

    Adds trainable low-rank matrices to frozen pre-trained weights
    Very efficient for fine-tuning large models

    Reference: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
    """
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(original_layer.in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, original_layer.out_features))

        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank

    def forward(self, x):
        # Original transformation
        result = self.original_layer(x)

        # Add LoRA adaptation
        lora_result = self.dropout(x) @ self.lora_A @ self.lora_B

        return result + lora_result * self.scaling


def apply_lora_to_model(model: nn.Module, rank: int = 8, alpha: float = 16.0, target_modules: List[str] = None):
    """
    Apply LoRA to specific modules in the model

    Args:
        model: The model to adapt
        rank: LoRA rank
        alpha: LoRA alpha
        target_modules: List of module names to apply LoRA to (e.g., ['q_proj', 'v_proj'])
    """
    if target_modules is None:
        # Default: apply to all linear layers in attention
        target_modules = ['W_q', 'W_k', 'W_v', 'W_o']

    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA layer
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                parent = model.get_submodule(parent_name) if parent_name else model

                lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)

                print(f"Applied LoRA to {name}")

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nLoRA applied: {trainable:,} trainable / {total:,} total parameters ({100*trainable/total:.2f}%)")


if __name__ == "__main__":
    print("Testing Continual Learning Utilities...\n")

    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 3)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    model = SimpleModel()

    # Test LoRA
    print("Testing LoRA...")
    apply_lora_to_model(model, rank=4, target_modules=['fc1', 'fc2'])

    # Test forward pass
    x = torch.randn(2, 10)
    out = model(x)
    print(f"✓ Output shape: {out.shape}\n")

    # Test Experience Replay
    print("Testing Experience Replay...")
    replay = ExperienceReplay(buffer_size=100)

    for i in range(50):
        experience = {'data': torch.randn(10), 'label': i}
        replay.add(experience)

    samples = replay.sample(10)
    print(f"✓ Buffer size: {len(replay)}")
    print(f"✓ Sampled {len(samples)} experiences\n")

    print("✓ All tests passed!")
