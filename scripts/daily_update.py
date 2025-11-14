"""
Daily Update Script for Financial LLM

This script:
1. Fetches latest market data
2. Fine-tunes the model on new data using continual learning
3. Updates the model without catastrophic forgetting
4. Saves the updated model
"""

import torch
from torch.utils.data import DataLoader
import argparse
from datetime import datetime, timedelta
import os

from financial_llm.models.financial_llm import FinancialLLM
from financial_llm.data_processors.market_data import MarketDataFetcher, MarketDataset, TechnicalIndicators
from train_financial_llm import FinancialLLMTrainer


def fetch_recent_data(ticker: str, days_back: int = 30):
    """
    Fetch recent market data for update

    Args:
        ticker: Stock ticker
        days_back: Number of days of data to fetch

    Returns:
        DataFrame with OHLCV data
    """
    fetcher = MarketDataFetcher()

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    print(f"Fetching {ticker} data from {start_date} to {end_date}...")

    try:
        data = fetcher.fetch_stock_data(ticker, start_date, end_date)
        print(f"✓ Fetched {len(data)} records")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def daily_update(args):
    """
    Perform daily model update
    """
    print("=" * 60)
    print("Financial LLM Daily Update")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load existing model
    print(f"\nLoading model from {args.checkpoint_path}...")

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found at {args.checkpoint_path}")
        print("Please train an initial model first using train_financial_llm.py")
        return

    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Recreate model (you'd need to save config in checkpoint for production)
    model = FinancialLLM(
        vocab_size=50000,
        d_model=256,
        num_heads=8,
        num_text_layers=4,
        num_ts_layers=3,
        num_fusion_layers=2,
        ts_input_dim=7,
        num_indicators=20,
        max_seq_len_ts=60,
        dropout=0.1
    )

    # Create trainer with continual learning enabled
    trainer = FinancialLLMTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        use_continual_learning=True,
        ewc_lambda=args.ewc_lambda,
        replay_buffer_size=args.replay_buffer_size
    )

    # Load checkpoint
    trainer.load_checkpoint(args.checkpoint_path)

    print("✓ Model loaded successfully")

    # Fetch recent data for each ticker
    all_datasets = []

    for ticker in args.tickers:
        print(f"\nProcessing {ticker}...")

        data = fetch_recent_data(ticker, days_back=args.days_back)

        if data is None or len(data) < args.seq_len:
            print(f"Skipping {ticker} - insufficient data")
            continue

        # Add indicators
        data = TechnicalIndicators.add_indicators(data)

        # Create dataset
        dataset = MarketDataset(
            data,
            seq_len=args.seq_len,
            pred_horizon=1,
            include_indicators=True,
            task='classification'
        )

        all_datasets.append(dataset)

        print(f"✓ Created dataset with {len(dataset)} samples")

    if not all_datasets:
        print("\nNo new data to train on. Exiting.")
        return

    # Combine datasets
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset(all_datasets)

    print(f"\nTotal samples for update: {len(combined_dataset)}")

    # Create data loader
    update_loader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    # Fine-tune on new data
    print("\nFine-tuning on new data...")
    print("-" * 60)

    for epoch in range(1, args.update_epochs + 1):
        train_loss, train_acc = trainer.train_epoch(update_loader, epoch)

        print(f"Epoch {epoch}/{args.update_epochs}")
        print(f"  Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")

    # Update EWC for next update
    if trainer.use_continual_learning:
        print("\nUpdating EWC fisher matrix...")
        trainer.cl_trainer.initialize_ewc(update_loader)

    # Save updated model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(
        args.output_dir,
        f"model_updated_{timestamp}.pt"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_checkpoint(output_path, epoch=0, val_loss=0, val_acc=0)

    print(f"\n✓ Updated model saved to {output_path}")

    # Also update the "latest" symlink/copy
    latest_path = os.path.join(args.output_dir, "latest_model.pt")
    import shutil
    shutil.copy(output_path, latest_path)
    print(f"✓ Latest model updated at {latest_path}")

    print("\n" + "=" * 60)
    print("Daily update complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Daily update for Financial LLM')

    # Model args
    parser.add_argument('--checkpoint-path', type=str, required=True,
                       help='Path to existing model checkpoint')
    parser.add_argument('--output-dir', type=str, default='checkpoints/financial_llm',
                       help='Directory to save updated model')

    # Data args
    parser.add_argument('--tickers', type=str, nargs='+', default=['AAPL', 'GOOGL', 'MSFT'],
                       help='List of tickers to update on')
    parser.add_argument('--days-back', type=int, default=30,
                       help='Number of days of recent data to fetch')
    parser.add_argument('--seq-len', type=int, default=60,
                       help='Sequence length (must match training)')

    # Update args
    parser.add_argument('--update-epochs', type=int, default=3,
                       help='Number of epochs for fine-tuning')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for update')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate (lower than initial training)')

    # Continual learning args
    parser.add_argument('--ewc-lambda', type=float, default=1000.0,
                       help='EWC regularization (higher than training to prevent forgetting)')
    parser.add_argument('--replay-buffer-size', type=int, default=10000,
                       help='Experience replay buffer size')

    args = parser.parse_args()

    daily_update(args)
