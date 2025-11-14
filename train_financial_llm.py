"""
Training Pipeline for Financial LLM

Supports:
- Initial training on historical data
- Continual learning with EWC and Experience Replay
- Incremental daily updates
- LoRA-based efficient fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

import argparse
import os
from tqdm import tqdm
from datetime import datetime
import json

from financial_llm.models.financial_llm import FinancialLLM, count_parameters
from financial_llm.data_processors.market_data import (
    create_market_datasets,
    MarketDataFetcher,
    MarketDataset
)
from financial_llm.utils.continual_learning import (
    ContinualLearningTrainer,
    apply_lora_to_model
)


class FinancialLLMTrainer:
    """
    Main trainer for Financial LLM
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        use_continual_learning: bool = False,
        ewc_lambda: float = 100.0,
        replay_buffer_size: int = 10000
    ):
        self.model = model.to(device)
        self.device = device

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Continual learning
        self.use_continual_learning = use_continual_learning
        if use_continual_learning:
            self.cl_trainer = ContinualLearningTrainer(
                model=model,
                device=device,
                ewc_lambda=ewc_lambda,
                replay_buffer_size=replay_buffer_size
            )
        else:
            self.cl_trainer = None

        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        scheduler=None
    ) -> float:
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            ohlcv = batch['ohlcv'].to(self.device)
            labels = batch['label'].to(self.device)
            indicators = batch.get('indicators', None)
            if indicators is not None:
                indicators = indicators.to(self.device)

            # Forward pass
            outputs = self.model(
                ohlcv=ohlcv,
                indicators=indicators,
                task='trading'
            )

            # Compute loss
            trading_logits = outputs['trading_logits']
            loss = F.cross_entropy(trading_logits, labels)

            # Add continual learning penalties if enabled
            if self.use_continual_learning and self.cl_trainer.ewc is not None:
                ewc_loss = self.cl_trainer.ewc.penalty(self.model)
                total_loss_value = loss + self.cl_trainer.ewc_lambda * ewc_loss
            else:
                total_loss_value = loss
                ewc_loss = torch.tensor(0.0)

            # Backward pass
            self.optimizer.zero_grad()
            total_loss_value.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Metrics
            total_loss += loss.item()

            predictions = torch.argmax(trading_logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # Add to replay buffer if using continual learning
            if self.use_continual_learning:
                self.cl_trainer.add_to_replay_buffer(batch)

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'ewc_loss': ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else 0,
                'acc': total_correct / total_samples
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def evaluate(self, val_loader: DataLoader) -> tuple:
        """
        Evaluate on validation set
        """
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        # For more detailed metrics
        all_predictions = []
        all_labels = []
        all_price_predictions = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                ohlcv = batch['ohlcv'].to(self.device)
                labels = batch['label'].to(self.device)
                indicators = batch.get('indicators', None)
                if indicators is not None:
                    indicators = indicators.to(self.device)

                outputs = self.model(
                    ohlcv=ohlcv,
                    indicators=indicators,
                    task='trading'
                )

                trading_logits = outputs['trading_logits']
                loss = F.cross_entropy(trading_logits, labels)

                total_loss += loss.item()

                predictions = torch.argmax(trading_logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if 'price_prediction' in outputs:
                    all_price_predictions.extend(outputs['price_prediction'].cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy, all_predictions, all_labels, all_price_predictions

    def save_checkpoint(self, path: str, epoch: int, val_loss: float, val_acc: float):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }

        # Add continual learning state if enabled
        if self.use_continual_learning:
            checkpoint['ewc_params'] = self.cl_trainer.ewc.params if self.cl_trainer.ewc else None
            checkpoint['ewc_fisher'] = self.cl_trainer.ewc.fisher_matrix if self.cl_trainer.ewc else None
            checkpoint['replay_buffer'] = list(self.cl_trainer.replay_buffer.buffer)

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])

        # Load continual learning state if available
        if self.use_continual_learning and 'ewc_params' in checkpoint:
            if checkpoint['ewc_params'] is not None:
                from financial_llm.utils.continual_learning import EWC
                self.cl_trainer.ewc = EWC.__new__(EWC)
                self.cl_trainer.ewc.model = self.model
                self.cl_trainer.ewc.device = self.device
                self.cl_trainer.ewc.params = checkpoint['ewc_params']
                self.cl_trainer.ewc.fisher_matrix = checkpoint['ewc_fisher']

            if 'replay_buffer' in checkpoint and checkpoint['replay_buffer']:
                for exp in checkpoint['replay_buffer']:
                    self.cl_trainer.replay_buffer.add(exp)

        print(f"Checkpoint loaded from {path}")


def main(args):
    print("=" * 60)
    print("Financial LLM Training Pipeline")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create datasets
    print(f"\nCreating datasets for {args.ticker}...")
    train_dataset, val_dataset, test_dataset = create_market_datasets(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        seq_len=args.seq_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        include_indicators=True,
        task='classification'
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Create model
    print("\nInitializing model...")
    model = FinancialLLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_text_layers=args.num_text_layers,
        num_ts_layers=args.num_ts_layers,
        num_fusion_layers=args.num_fusion_layers,
        ts_input_dim=7,
        num_indicators=train_dataset.indicator_data.shape[1] if hasattr(train_dataset, 'indicator_data') and train_dataset.indicator_data is not None else 20,
        max_seq_len_ts=args.seq_len,
        dropout=args.dropout
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Apply LoRA if requested
    if args.use_lora:
        print("\nApplying LoRA...")
        apply_lora_to_model(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_modules=['W_q', 'W_v']
        )

    # Create trainer
    trainer = FinancialLLMTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_continual_learning=args.use_continual_learning,
        ewc_lambda=args.ewc_lambda,
        replay_buffer_size=args.replay_buffer_size
    )

    # Load checkpoint if resuming
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
        start_epoch = len(trainer.train_losses) + 1
    else:
        start_epoch = 1

    # Learning rate scheduler
    scheduler = OneCycleLR(
        trainer.optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader)
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch, scheduler)
        trainer.train_losses.append(train_loss)

        # Evaluate
        val_loss, val_acc, _, _, _ = trainer.evaluate(val_loader)
        trainer.val_losses.append(val_loss)
        trainer.val_accuracies.append(val_acc)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            trainer.save_checkpoint(save_path, epoch, val_loss, val_acc)
            print(f"  ✓ New best model saved!")

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            trainer.save_checkpoint(save_path, epoch, val_loss, val_acc)

        # Initialize EWC after first epoch if using continual learning
        if args.use_continual_learning and epoch == 1:
            print("\nInitializing EWC for continual learning...")
            trainer.cl_trainer.initialize_ewc(train_loader)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Financial LLM')

    # Data args
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--seq-len', type=int, default=60, help='Sequence length')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation set ratio')

    # Model args
    parser.add_argument('--vocab-size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-text-layers', type=int, default=4, help='Number of text encoder layers')
    parser.add_argument('--num-ts-layers', type=int, default=3, help='Number of time series encoder layers')
    parser.add_argument('--num-fusion-layers', type=int, default=2, help='Number of fusion layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training args
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')

    # Continual learning args
    parser.add_argument('--use-continual-learning', action='store_true', help='Enable continual learning (EWC + replay)')
    parser.add_argument('--ewc-lambda', type=float, default=100.0, help='EWC regularization strength')
    parser.add_argument('--replay-buffer-size', type=int, default=10000, help='Experience replay buffer size')

    # LoRA args
    parser.add_argument('--use-lora', action='store_true', help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--lora-rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=float, default=16.0, help='LoRA alpha')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/financial_llm', help='Checkpoint directory')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    main(args)
