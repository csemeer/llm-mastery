"""
Quick test script to verify the model works
"""

import torch
from src.model import GPTModel, count_parameters


def test_model():
    print("Testing GPT Model Implementation...\n")

    # Create a small model
    vocab_size = 100
    model = GPTModel(
        vocab_size=vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        max_seq_len=32,
        dropout=0.1
    )

    print(f"✓ Model created successfully")
    print(f"✓ Parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size, seq_len = 4, 16
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits, loss = model(x, targets=x)

    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    # Test generation
    context = torch.randint(0, vocab_size, (1, 5))
    generated = model.generate(context, max_new_tokens=10, temperature=1.0)

    print(f"✓ Generation successful")
    print(f"  Context length: {context.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")

    print("\n✓ All tests passed! The model is working correctly.")
    print("\nNext steps:")
    print("1. Train the model: python train.py --epochs 10")
    print("2. Generate text: python generate.py --interactive")


if __name__ == "__main__":
    test_model()
