"""
Quick Start Example for Financial LLM

Demonstrates:
1. Fetching market data
2. Creating a simple model
3. Training for a few epochs
4. Making predictions
"""

import torch
from torch.utils.data import DataLoader
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_llm.models.financial_llm import FinancialLLM, count_parameters
from financial_llm.data_processors.market_data import create_market_datasets
from train_financial_llm import FinancialLLMTrainer


def main():
    print("=" * 60)
    print("Financial LLM - Quick Start Example")
    print("=" * 60)

    # Check dependencies
    try:
        import yfinance as yf
    except ImportError:
        print("\nError: yfinance not installed")
        print("Install with: pip install yfinance")
        return

    # Configuration
    TICKER = 'AAPL'
    START_DATE = '2023-01-01'
    END_DATE = '2024-01-01'
    EPOCHS = 5  # Just for demonstration

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")

    # Step 1: Fetch and prepare data
    print("Step 1: Fetching market data...")
    print("-" * 60)

    try:
        train_dataset, val_dataset, test_dataset = create_market_datasets(
            ticker=TICKER,
            start_date=START_DATE,
            end_date=END_DATE,
            seq_len=60,
            train_ratio=0.7,
            val_ratio=0.15,
            include_indicators=True,
            task='classification'
        )
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Make sure you have internet connection and yfinance installed")
        return

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Step 2: Create model
    print("\nStep 2: Creating model...")
    print("-" * 60)

    # Small model for quick testing
    model = FinancialLLM(
        vocab_size=10000,
        d_model=128,
        num_heads=4,
        num_text_layers=2,
        num_ts_layers=2,
        num_fusion_layers=1,
        ts_input_dim=7,
        num_indicators=train_dataset.indicator_data.shape[1] if hasattr(train_dataset, 'indicator_data') and train_dataset.indicator_data is not None else 20,
        max_seq_len_ts=60,
        dropout=0.1
    )

    print(f"Model created with {count_parameters(model):,} parameters")

    # Step 3: Train
    print(f"\nStep 3: Training for {EPOCHS} epochs...")
    print("-" * 60)

    trainer = FinancialLLMTrainer(
        model=model,
        device=device,
        learning_rate=3e-4
    )

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        val_loss, val_acc, _, _, _ = trainer.evaluate(val_loader)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Step 4: Make predictions
    print("\nStep 4: Making predictions...")
    print("-" * 60)

    model.eval()

    # Get a sample from test set
    sample = test_dataset[0]
    ohlcv = sample['ohlcv'].unsqueeze(0).to(device)
    indicators = sample.get('indicators')
    if indicators is not None:
        indicators = indicators.unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model.predict_trading_action(
            ohlcv=ohlcv,
            indicators=indicators
        )

    action_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
    predicted_action = predictions['action'].item()
    action_probs = predictions['action_probs'][0]

    print(f"\nPrediction:")
    print(f"  Recommended Action: {action_map[predicted_action]}")
    print(f"  Confidence:")
    print(f"    BUY:  {action_probs[0]:.2%}")
    print(f"    SELL: {action_probs[1]:.2%}")
    print(f"    HOLD: {action_probs[2]:.2%}")

    if 'price_prediction' in predictions:
        price_change = predictions['price_prediction'].item()
        print(f"  Expected Price Change: {price_change:.2%}")

    # Step 5: Save model
    print("\nStep 5: Saving model...")
    print("-" * 60)

    os.makedirs('checkpoints/examples', exist_ok=True)
    trainer.save_checkpoint(
        'checkpoints/examples/quick_start_model.pt',
        epoch=EPOCHS,
        val_loss=val_loss,
        val_acc=val_acc
    )

    print("\n" + "=" * 60)
    print("Quick start complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train on more data: python train_financial_llm.py")
    print("2. Set up daily updates: python scripts/daily_update.py")
    print("3. Read FINANCIAL_LLM_README.md for advanced features")


if __name__ == "__main__":
    main()
