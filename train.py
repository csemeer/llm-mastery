"""
Training script for the small GPT model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import argparse

from src.model import GPTModel, count_parameters
from src.dataset import CharDataset, WordDataset


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(train_loader)


def evaluate(model, val_loader, device):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def generate_sample(model, dataset, device, seed_text="The ", max_new_tokens=200, temperature=0.8):
    """Generate text sample"""
    model.eval()

    # Encode seed text
    if hasattr(dataset, 'encode'):
        context = dataset.encode(seed_text)
    else:
        context = [0]  # Fallback

    context = torch.tensor([context], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_new_tokens,
                                   temperature=temperature, top_k=50)

    # Decode
    generated_text = dataset.decode(generated[0].cpu().tolist())
    return generated_text


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading dataset from {args.data_path}...")
    if args.level == 'char':
        dataset = CharDataset(args.data_path, seq_len=args.seq_len)
    else:
        dataset = WordDataset(args.data_path, seq_len=args.seq_len)

    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    print("\nInitializing model...")
    model = GPTModel(
        vocab_size=dataset.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        max_seq_len=args.seq_len,
        dropout=args.dropout
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...\n")
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Generate sample
        if epoch % args.sample_every == 0:
            sample = generate_sample(model, dataset, device, temperature=0.8)
            print(f"\nGenerated sample:\n{'-'*50}\n{sample}\n{'-'*50}\n")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'vocab_size': dataset.vocab_size,
                    'embed_dim': args.embed_dim,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'ff_dim': args.ff_dim,
                    'max_seq_len': args.seq_len,
                    'dropout': args.dropout
                }
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a small GPT model')

    # Data args
    parser.add_argument('--data-path', type=str, default='data/input.txt',
                       help='Path to training text file')
    parser.add_argument('--level', type=str, default='char', choices=['char', 'word'],
                       help='Character-level or word-level modeling')
    parser.add_argument('--seq-len', type=int, default=128,
                       help='Sequence length for training')

    # Model args
    parser.add_argument('--embed-dim', type=int, default=256,
                       help='Embedding dimension')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--ff-dim', type=int, default=1024,
                       help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')

    # Training args
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--sample-every', type=int, default=5,
                       help='Generate sample every N epochs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')

    args = parser.parse_args()
    main(args)
