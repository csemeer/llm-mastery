# Small GPT - A Learning-Oriented Language Model

A minimal, well-documented implementation of a GPT-style transformer for educational purposes. This project helps you understand how modern language models work by building one from scratch in PyTorch.

## Features

- Clean, educational implementation with extensive comments
- Character-level and word-level language modeling
- Multi-head self-attention mechanism
- Transformer architecture with residual connections and layer normalization
- Text generation with temperature and top-k sampling
- Interactive generation mode
- Training with validation and checkpointing

## Project Structure

```
llm-mastery/
├── src/
│   ├── model.py          # GPT model architecture
│   └── dataset.py        # Dataset utilities
├── data/
│   └── input.txt         # Training data (Shakespeare by default)
├── checkpoints/          # Saved models
├── train.py              # Training script
├── generate.py           # Text generation script
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-mastery
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train the Model

Train on the provided Shakespeare dataset:

```bash
python train.py --epochs 20 --batch-size 32
```

For a smaller/faster model (good for testing):
```bash
python train.py --embed-dim 128 --num-layers 4 --num-heads 4 --epochs 10
```

### 2. Generate Text

After training, generate text:

```bash
python generate.py --prompt "To be or not to be" --max-tokens 200
```

Interactive mode:
```bash
python generate.py --interactive
```

## Understanding the Model

### Architecture Components

#### 1. **Multi-Head Self-Attention** (`src/model.py:17`)
- Allows the model to focus on different parts of the input simultaneously
- Splits embedding into multiple "heads" for parallel attention computation
- Uses scaled dot-product attention: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`

#### 2. **Feed-Forward Network** (`src/model.py:64`)
- Position-wise fully connected layers
- Applies non-linear transformations independently to each position
- Uses GELU activation (smooth approximation of ReLU)

#### 3. **Transformer Block** (`src/model.py:79`)
- Combines attention and feed-forward layers
- Uses residual connections (helps with gradient flow)
- Layer normalization (stabilizes training)

#### 4. **GPT Model** (`src/model.py:104`)
- Token + positional embeddings
- Stack of transformer blocks
- Output projection to vocabulary

### Training Process

The model learns to predict the next character/word given previous context:

1. **Input**: Sequence of tokens → `[h, e, l, l, o]`
2. **Target**: Next token for each position → `[e, l, l, o, !]`
3. **Loss**: Cross-entropy between predictions and targets
4. **Optimization**: Adam with learning rate scheduling

## Training Options

### Model Size

```bash
# Tiny model (~100K params, fast training on CPU)
python train.py --embed-dim 64 --num-layers 2 --num-heads 2 --ff-dim 256

# Small model (~1M params, good balance)
python train.py --embed-dim 128 --num-layers 4 --num-heads 4 --ff-dim 512

# Medium model (~5M params, better quality)
python train.py --embed-dim 256 --num-layers 6 --num-heads 8 --ff-dim 1024
```

### Data Level

```bash
# Character-level (default, learns spelling/grammar)
python train.py --level char --seq-len 128

# Word-level (faster, requires more data)
python train.py --level word --seq-len 64
```

### Custom Dataset

Replace `data/input.txt` with your own text file:

```bash
python train.py --data-path path/to/your/text.txt
```

## Generation Options

### Temperature
Controls randomness (higher = more creative, lower = more conservative):

```bash
python generate.py --temperature 0.5  # Conservative
python generate.py --temperature 1.0  # Balanced
python generate.py --temperature 1.5  # Creative
```

### Top-k Sampling
Only sample from k most likely tokens:

```bash
python generate.py --top-k 10   # Very focused
python generate.py --top-k 50   # Balanced (default)
python generate.py --top-k 100  # More diverse
```

## Model Details

### Default Configuration

- **Vocabulary size**: Depends on dataset (65 chars for Shakespeare)
- **Embedding dimension**: 256
- **Number of heads**: 8
- **Number of layers**: 6
- **Feed-forward dimension**: 1024
- **Parameters**: ~5-10M (depending on vocab size)
- **Context window**: 128 tokens

### Key Concepts

1. **Autoregressive Generation**: Model generates one token at a time, using previous tokens as context

2. **Causal Masking**: During training, each position can only attend to previous positions (enforces left-to-right generation)

3. **Positional Encoding**: Learned embeddings that encode position information (since attention has no inherent order)

4. **Layer Normalization**: Normalizes activations to stabilize training

5. **Residual Connections**: Skip connections that help gradients flow through deep networks

## Educational Exercises

### Beginner
1. Train a tiny model and observe how it learns over epochs
2. Experiment with different temperature values during generation
3. Try character-level vs word-level modeling

### Intermediate
1. Modify the attention mechanism to add attention visualization
2. Implement different positional encoding schemes (sinusoidal, rotary)
3. Add beam search for generation
4. Try different datasets (code, poetry, etc.)

### Advanced
1. Implement sparse attention patterns
2. Add model parallelism for larger models
3. Implement knowledge distillation
4. Add reinforcement learning from human feedback (RLHF)

## Performance Tips

### CPU Training
- Use smaller models (embed_dim=64-128, num_layers=2-4)
- Reduce batch size (8-16)
- Use shorter sequences (seq_len=64-128)

### GPU Training
- Increase batch size (64-128)
- Use larger models (embed_dim=256-512)
- Enable mixed precision training (add to code)

## Troubleshooting

### Model not learning
- Decrease learning rate: `--lr 1e-4`
- Increase model size
- Train for more epochs
- Check your data quality

### Out of memory
- Reduce batch size: `--batch-size 16`
- Reduce sequence length: `--seq-len 64`
- Reduce model size

### Generated text is repetitive
- Increase temperature: `--temperature 1.2`
- Increase top-k: `--top-k 100`
- Train for more epochs
- Use more diverse training data

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - Minimal GPT implementation

## Next Steps

1. **Try larger datasets**: Books, Wikipedia, code repositories
2. **Implement improvements**: Better tokenization, byte-pair encoding
3. **Add features**: Temperature scheduling, nucleus sampling
4. **Scale up**: Distributed training, model parallelism
5. **Fine-tuning**: Adapt pre-trained model to specific tasks

## License

MIT License - feel free to use for learning and experimentation!

## Contributing

This is an educational project. Feel free to:
- Add more comments and documentation
- Implement additional features
- Create tutorials and examples
- Share your trained models

Happy learning! 🚀
