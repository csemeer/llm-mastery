"""
Text generation script using trained model
"""

import torch
import argparse
import os

from src.model import GPTModel
from src.dataset import CharDataset, WordDataset


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Create model with saved config
    model = GPTModel(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        ff_dim=config['ff_dim'],
        max_seq_len=config['max_seq_len'],
        dropout=0.0  # No dropout during inference
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs, val_loss: {checkpoint['val_loss']:.4f}")

    return model, config


def generate_text(model, dataset, device, prompt="", max_new_tokens=500,
                 temperature=1.0, top_k=50):
    """
    Generate text from the model

    Args:
        model: Trained GPT model
        dataset: Dataset (for encoding/decoding)
        device: torch device
        prompt: Starting text
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Sample from top-k tokens only
    """
    model.eval()

    # Encode prompt
    if prompt:
        context = dataset.encode(prompt)
        print(f"Prompt: {prompt}")
    else:
        context = [0]  # Start with first token
        print("Generating from scratch...")

    context = torch.tensor([context], dtype=torch.long, device=device)

    # Generate
    print(f"\nGenerating {max_new_tokens} tokens with temperature={temperature}, top_k={top_k}...\n")
    print("=" * 60)

    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

    # Decode and print
    generated_text = dataset.decode(generated[0].cpu().tolist())
    print(generated_text)
    print("=" * 60)

    return generated_text


def interactive_mode(model, dataset, device):
    """Interactive text generation"""
    print("\n" + "=" * 60)
    print("Interactive Generation Mode")
    print("=" * 60)
    print("Commands:")
    print("  - Type a prompt and press Enter to generate")
    print("  - Type 'config' to change generation settings")
    print("  - Type 'quit' to exit")
    print("=" * 60 + "\n")

    temperature = 0.8
    top_k = 50
    max_tokens = 200

    while True:
        prompt = input("Prompt (or 'quit'/'config'): ").strip()

        if prompt.lower() == 'quit':
            break
        elif prompt.lower() == 'config':
            try:
                temperature = float(input(f"Temperature ({temperature}): ") or temperature)
                top_k = int(input(f"Top-k ({top_k}): ") or top_k)
                max_tokens = int(input(f"Max tokens ({max_tokens}): ") or max_tokens)
                print(f"Updated: temp={temperature}, top_k={top_k}, max_tokens={max_tokens}\n")
            except ValueError:
                print("Invalid input, keeping previous settings\n")
            continue

        if not prompt:
            prompt = ""

        # Generate
        try:
            generate_text(model, dataset, device, prompt, max_tokens, temperature, top_k)
        except Exception as e:
            print(f"Error during generation: {e}")

        print()


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Load dataset for vocab
    print(f"\nLoading dataset from {args.data_path} for vocabulary...")
    if args.level == 'char':
        dataset = CharDataset(args.data_path, seq_len=config['max_seq_len'])
    else:
        dataset = WordDataset(args.data_path, seq_len=config['max_seq_len'])

    # Verify vocab size matches
    if dataset.vocab_size != config['vocab_size']:
        print(f"Warning: Dataset vocab size ({dataset.vocab_size}) != model vocab size ({config['vocab_size']})")

    # Interactive or single generation
    if args.interactive:
        interactive_mode(model, dataset, device)
    else:
        generate_text(model, dataset, device, args.prompt,
                     args.max_tokens, args.temperature, args.top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text using trained GPT model')

    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data-path', type=str, default='data/input.txt',
                       help='Path to training text file (for vocabulary)')
    parser.add_argument('--level', type=str, default='char', choices=['char', 'word'],
                       help='Character-level or word-level modeling')

    parser.add_argument('--prompt', type=str, default='',
                       help='Starting prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=500,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Sample from top-k tokens only')

    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive mode')

    args = parser.parse_args()
    main(args)
