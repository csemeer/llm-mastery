"""
Simple GPT-style Transformer for Learning
A minimal implementation to understand transformer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    Allows the model to attend to different positions simultaneously
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        # scores = Q * K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask (for autoregressive generation)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)

        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)

        output = self.out_proj(attn_output)
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network
    Applied to each position separately and identically
    """
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)  # GELU activation (used in GPT)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block
    Combines self-attention and feed-forward layers with residual connections
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_output)

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_output)

        return x


class GPTModel(nn.Module):
    """
    Simple GPT-style transformer language model
    """
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=6,
                 ff_dim=1024, max_seq_len=512, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"

        # Create position indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        pos = pos.unsqueeze(0).expand(batch_size, seq_len)

        # Embeddings
        tok_emb = self.token_embedding(idx)  # (batch, seq_len, embed_dim)
        pos_emb = self.position_embedding(pos)  # (batch, seq_len, embed_dim)
        x = self.dropout(tok_emb + pos_emb)

        # Create causal mask (lower triangular matrix)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=idx.device)).unsqueeze(0).unsqueeze(0)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final layer norm and projection to vocabulary
        x = self.ln_f(x)
        logits = self.head(x)  # (batch, seq_len, vocab_size)

        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively

        Args:
            idx: Initial context (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context if it exceeds max sequence length
                idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]

                # Get predictions
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature  # Focus on last position

                # Optional top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    vocab_size = 50
    model = GPTModel(vocab_size=vocab_size, embed_dim=128, num_heads=4,
                     num_layers=4, ff_dim=512, max_seq_len=128)

    print(f"Model has {count_parameters(model):,} trainable parameters")

    # Test forward pass
    batch_size, seq_len = 4, 32
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, loss = model(x, targets=x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Test generation
    context = torch.randint(0, vocab_size, (1, 10))
    generated = model.generate(context, max_new_tokens=20, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
