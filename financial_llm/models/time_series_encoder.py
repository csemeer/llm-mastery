"""
Time Series Encoder for Financial Data

Handles OHLCV data, technical indicators, and other time series features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Temporal positional encoding for time series
    Uses sinusoidal encoding to capture temporal patterns
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesAttention(nn.Module):
    """
    Self-attention mechanism adapted for time series
    Includes causal masking for autoregressive prediction
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        # Linear projections and reshape
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.W_o(out)

        return out, attn


class TimeSeriesEncoderLayer(nn.Module):
    """
    Single encoder layer for time series processing
    Combines attention, feed-forward, and normalization
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = TimeSeriesAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.self_attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x


class TimeSeriesEncoder(nn.Module):
    """
    Transformer-based encoder for financial time series

    Processes multi-dimensional time series data (OHLCV, indicators, etc.)
    and produces contextualized embeddings
    """
    def __init__(
        self,
        input_dim,           # Number of features (e.g., OHLCV = 5, + indicators)
        d_model=256,         # Model dimension
        num_heads=8,         # Number of attention heads
        num_layers=6,        # Number of encoder layers
        d_ff=1024,          # Feed-forward dimension
        max_seq_len=252,    # Max sequence length (e.g., 1 trading year)
        dropout=0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Project input features to model dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder layers
        self.layers = nn.ModuleList([
            TimeSeriesEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Encoded tensor of shape (batch, seq_len, d_model)
        """
        # Project and add positional encoding
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return x


class PriceEncoder(nn.Module):
    """
    Specialized encoder for price data
    Includes technical analysis features
    """
    def __init__(
        self,
        d_model=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1
    ):
        super().__init__()

        # Base features: OHLCV (5) + returns (1) + volume changes (1)
        base_features = 7

        # Additional technical indicators will be added
        self.feature_dim = base_features

        self.encoder = TimeSeriesEncoder(
            input_dim=self.feature_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_model * 4,
            dropout=dropout
        )

    def preprocess_prices(self, ohlcv):
        """
        Preprocess raw OHLCV data

        Args:
            ohlcv: Tensor of shape (batch, seq_len, 5) [Open, High, Low, Close, Volume]

        Returns:
            Processed features tensor
        """
        # Calculate returns
        close_prices = ohlcv[:, :, 3]  # Close prices
        returns = torch.zeros_like(close_prices)
        returns[:, 1:] = (close_prices[:, 1:] - close_prices[:, :-1]) / (close_prices[:, :-1] + 1e-8)

        # Calculate volume changes
        volumes = ohlcv[:, :, 4]
        volume_changes = torch.zeros_like(volumes)
        volume_changes[:, 1:] = (volumes[:, 1:] - volumes[:, :-1]) / (volumes[:, :-1] + 1e-8)

        # Combine features
        features = torch.cat([
            ohlcv,
            returns.unsqueeze(-1),
            volume_changes.unsqueeze(-1)
        ], dim=-1)

        return features

    def forward(self, ohlcv, mask=None):
        """
        Args:
            ohlcv: Raw OHLCV data (batch, seq_len, 5)

        Returns:
            Encoded representations
        """
        features = self.preprocess_prices(ohlcv)
        return self.encoder(features, mask)


class IndicatorEncoder(nn.Module):
    """
    Encoder for technical indicators and derived features
    """
    def __init__(
        self,
        num_indicators=20,  # Number of technical indicators
        d_model=256,
        num_heads=8,
        num_layers=3,
        dropout=0.1
    ):
        super().__init__()

        self.encoder = TimeSeriesEncoder(
            input_dim=num_indicators,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_model * 4,
            dropout=dropout
        )

    def forward(self, indicators, mask=None):
        """
        Args:
            indicators: Technical indicators (batch, seq_len, num_indicators)

        Returns:
            Encoded representations
        """
        return self.encoder(indicators, mask)


def calculate_technical_indicators(ohlcv_data):
    """
    Calculate common technical indicators from OHLCV data

    This is a placeholder - you would typically use libraries like 'ta'
    for comprehensive technical analysis

    Args:
        ohlcv_data: Tensor of shape (batch, seq_len, 5)

    Returns:
        Dictionary of technical indicators
    """
    close = ohlcv_data[:, :, 3]
    high = ohlcv_data[:, :, 1]
    low = ohlcv_data[:, :, 2]
    volume = ohlcv_data[:, :, 4]

    indicators = {}

    # Simple Moving Averages
    for period in [5, 10, 20, 50]:
        if close.shape[1] >= period:
            sma = F.avg_pool1d(
                close.unsqueeze(1),
                kernel_size=period,
                stride=1,
                padding=period//2
            ).squeeze(1)
            indicators[f'SMA_{period}'] = sma

    # Price momentum
    for period in [5, 10, 20]:
        if close.shape[1] > period:
            momentum = close[:, period:] - close[:, :-period]
            # Pad to maintain shape
            momentum = F.pad(momentum, (period, 0))
            indicators[f'MOM_{period}'] = momentum

    # Relative Strength Index (simplified)
    # In practice, use proper RSI calculation
    returns = close[:, 1:] - close[:, :-1]
    returns = F.pad(returns, (1, 0))
    indicators['RETURNS'] = returns

    return indicators


if __name__ == "__main__":
    # Test time series encoder
    print("Testing Time Series Encoder...\n")

    batch_size = 4
    seq_len = 60  # 60 trading days

    # Test with OHLCV data
    ohlcv = torch.randn(batch_size, seq_len, 5)  # Random price data

    # Create encoder
    price_encoder = PriceEncoder(d_model=128, num_heads=4, num_layers=3)

    # Encode
    encoded = price_encoder(ohlcv)

    print(f"✓ Input shape: {ohlcv.shape}")
    print(f"✓ Encoded shape: {encoded.shape}")
    print(f"✓ Model parameters: {sum(p.numel() for p in price_encoder.parameters()):,}")

    # Test with indicators
    num_indicators = 15
    indicators = torch.randn(batch_size, seq_len, num_indicators)

    indicator_encoder = IndicatorEncoder(
        num_indicators=num_indicators,
        d_model=128,
        num_heads=4,
        num_layers=2
    )

    encoded_indicators = indicator_encoder(indicators)

    print(f"\n✓ Indicator input shape: {indicators.shape}")
    print(f"✓ Indicator encoded shape: {encoded_indicators.shape}")
    print(f"✓ Indicator encoder parameters: {sum(p.numel() for p in indicator_encoder.parameters()):,}")

    print("\n✓ Time series encoder tests passed!")
