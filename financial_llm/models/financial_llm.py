"""
Financial LLM - Hybrid Architecture for Multi-Modal Financial Data

Combines:
- Text understanding (SEC filings, news, reports)
- Time series analysis (price data, indicators)
- Cross-modal reasoning

Supports continual learning for daily updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from .time_series_encoder import PriceEncoder, IndicatorEncoder


class CrossModalAttention(nn.Module):
    """
    Cross-attention between text and time series modalities
    Allows the model to ground text in numerical data and vice versa
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: Query modality (batch, seq_len_q, d_model)
            key_value: Key-value modality (batch, seq_len_kv, d_model)
            mask: Attention mask
        """
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_kv = key_value.shape[1]

        # Project and reshape
        Q = self.W_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key_value).view(batch_size, seq_len_kv, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(key_value).view(batch_size, seq_len_kv, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        out = self.W_o(out)

        return out


class FusionLayer(nn.Module):
    """
    Fuses text and time series representations
    Uses gated mechanism to balance different modalities
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        self.cross_attn_text_to_ts = CrossModalAttention(d_model, num_heads=8, dropout=dropout)
        self.cross_attn_ts_to_text = CrossModalAttention(d_model, num_heads=8, dropout=dropout)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, text_features, ts_features):
        """
        Args:
            text_features: Text representations (batch, seq_len_text, d_model)
            ts_features: Time series representations (batch, seq_len_ts, d_model)
        """
        # Cross-attention: text attending to time series
        text_enhanced = self.cross_attn_text_to_ts(text_features, ts_features)
        text_features = self.norm1(text_features + text_enhanced)

        # Cross-attention: time series attending to text
        ts_enhanced = self.cross_attn_ts_to_text(ts_features, text_features)
        ts_features = self.norm2(ts_features + ts_enhanced)

        # Combine with gating
        # Pool sequences for gating
        text_pooled = text_features.mean(dim=1)  # (batch, d_model)
        ts_pooled = ts_features.mean(dim=1)      # (batch, d_model)

        gate_input = torch.cat([text_pooled, ts_pooled], dim=-1)
        gate = self.gate(gate_input)  # (batch, d_model)

        # Weighted combination
        combined_pooled = gate * text_pooled + (1 - gate) * ts_pooled

        # Feed-forward
        output = self.feed_forward(combined_pooled)

        return output, text_features, ts_features


class FinancialLLM(nn.Module):
    """
    Complete Financial LLM with multi-modal understanding

    Architecture:
    1. Text Encoder: Processes SEC filings, news, reports
    2. Time Series Encoder: Processes OHLCV and indicators
    3. Fusion Layer: Combines both modalities
    4. Task-specific heads: Trading, risk, portfolio management
    """
    def __init__(
        self,
        # Vocabulary
        vocab_size=50000,

        # Model dimensions
        d_model=512,
        num_heads=8,
        num_text_layers=6,
        num_ts_layers=4,
        num_fusion_layers=2,
        d_ff=2048,

        # Time series config
        ts_input_dim=7,  # OHLCV + derived features
        num_indicators=20,
        max_seq_len_text=512,
        max_seq_len_ts=252,  # 1 trading year

        # Dropout
        dropout=0.1,

        # Task heads
        num_trading_actions=3,  # Buy, Sell, Hold
        enable_risk_prediction=True,
        enable_portfolio_optimization=True
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # ==================== TEXT ENCODER ====================
        # Using a simplified GPT-style encoder
        # In production, you might use a pre-trained model like FinBERT

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.text_pos_embedding = nn.Embedding(max_seq_len_text, d_model)

        self.text_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_text_layers)
        ])

        # ==================== TIME SERIES ENCODER ====================
        self.price_encoder = PriceEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_ts_layers,
            dropout=dropout
        )

        self.indicator_encoder = IndicatorEncoder(
            num_indicators=num_indicators,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_ts_layers // 2,
            dropout=dropout
        )

        # Time series fusion
        self.ts_fusion = nn.Linear(d_model * 2, d_model)

        # ==================== CROSS-MODAL FUSION ====================
        self.fusion_layers = nn.ModuleList([
            FusionLayer(d_model, dropout)
            for _ in range(num_fusion_layers)
        ])

        # ==================== TASK-SPECIFIC HEADS ====================

        # Trading action head (buy/sell/hold prediction)
        self.trading_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_trading_actions)
        )

        # Price movement prediction head
        self.price_prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Predict next day return
        )

        # Risk assessment head
        if enable_risk_prediction:
            self.risk_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 5)  # Risk factors: volatility, beta, etc.
            )
        else:
            self.risk_head = None

        # Portfolio optimization head
        if enable_portfolio_optimization:
            self.portfolio_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)  # Weight for this asset in portfolio
            )
        else:
            self.portfolio_head = None

        # Text generation head (for explanations)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode_text(self, input_ids, attention_mask=None):
        """
        Encode text input (SEC filings, news, etc.)

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.text_pos_embedding(pos_ids)

        x = token_emb + pos_emb

        # Apply transformer layers
        for layer in self.text_encoder_layers:
            x = layer(x, src_key_padding_mask=~attention_mask if attention_mask is not None else None)

        return x

    def encode_timeseries(self, ohlcv, indicators=None):
        """
        Encode time series data

        Args:
            ohlcv: OHLCV data (batch, seq_len, 5)
            indicators: Technical indicators (batch, seq_len, num_indicators)
        """
        # Encode price data
        price_features = self.price_encoder(ohlcv)

        if indicators is not None:
            # Encode indicators
            indicator_features = self.indicator_encoder(indicators)

            # Combine price and indicator features
            # Match sequence lengths if different
            min_len = min(price_features.shape[1], indicator_features.shape[1])
            combined = torch.cat([
                price_features[:, :min_len, :],
                indicator_features[:, :min_len, :]
            ], dim=-1)

            ts_features = self.ts_fusion(combined)
        else:
            ts_features = price_features

        return ts_features

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ohlcv: Optional[torch.Tensor] = None,
        indicators: Optional[torch.Tensor] = None,
        task: str = 'trading',  # 'trading', 'risk', 'portfolio', 'generation'
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple tasks

        Args:
            input_ids: Text token IDs
            attention_mask: Text attention mask
            ohlcv: OHLCV price data
            indicators: Technical indicators
            task: Which task head to use
            return_features: Whether to return intermediate features
        """
        outputs = {}

        # Encode modalities
        text_features = None
        ts_features = None

        if input_ids is not None:
            text_features = self.encode_text(input_ids, attention_mask)

        if ohlcv is not None:
            ts_features = self.encode_timeseries(ohlcv, indicators)

        # Handle single-modality inputs
        if text_features is not None and ts_features is None:
            # Text-only mode
            fused_features = text_features.mean(dim=1)
        elif ts_features is not None and text_features is None:
            # Time series-only mode
            fused_features = ts_features.mean(dim=1)
        else:
            # Multi-modal fusion
            for fusion_layer in self.fusion_layers:
                fused_features, text_features, ts_features = fusion_layer(text_features, ts_features)

        # Apply task-specific head
        if task == 'trading':
            outputs['trading_logits'] = self.trading_head(fused_features)
            outputs['price_prediction'] = self.price_prediction_head(fused_features)

        elif task == 'risk':
            if self.risk_head is not None:
                outputs['risk_scores'] = self.risk_head(fused_features)
            else:
                raise ValueError("Risk head not enabled")

        elif task == 'portfolio':
            if self.portfolio_head is not None:
                outputs['portfolio_weights'] = self.portfolio_head(fused_features)
            else:
                raise ValueError("Portfolio head not enabled")

        elif task == 'generation':
            # Text generation (for explanations)
            if text_features is not None:
                outputs['lm_logits'] = self.lm_head(text_features)
            else:
                raise ValueError("Text input required for generation task")

        if return_features:
            outputs['fused_features'] = fused_features
            outputs['text_features'] = text_features
            outputs['ts_features'] = ts_features

        return outputs

    def predict_trading_action(self, input_ids=None, attention_mask=None, ohlcv=None, indicators=None):
        """Convenience method for trading prediction"""
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ohlcv=ohlcv,
            indicators=indicators,
            task='trading'
        )

        action_probs = F.softmax(outputs['trading_logits'], dim=-1)
        predicted_action = torch.argmax(action_probs, dim=-1)

        return {
            'action': predicted_action,  # 0: Buy, 1: Sell, 2: Hold
            'action_probs': action_probs,
            'price_prediction': outputs['price_prediction']
        }

    def generate_explanation(self, input_ids, max_length=100, temperature=1.0):
        """
        Generate text explanation for trading decisions

        Args:
            input_ids: Starting tokens
            max_length: Maximum length to generate
            temperature: Sampling temperature
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(input_ids, task='generation')
                logits = outputs['lm_logits'][:, -1, :] / temperature

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Stop if we generate an end token (you'd define this)
                # if next_token == END_TOKEN:
                #     break

        return input_ids


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing Financial LLM...\n")

    # Create model
    model = FinancialLLM(
        vocab_size=10000,
        d_model=256,
        num_heads=8,
        num_text_layers=4,
        num_ts_layers=3,
        num_fusion_layers=2
    )

    print(f"Model parameters: {count_parameters(model):,}\n")

    # Test with both modalities
    batch_size = 2

    # Text input (e.g., SEC filing excerpt)
    seq_len_text = 128
    input_ids = torch.randint(0, 10000, (batch_size, seq_len_text))
    attention_mask = torch.ones_like(input_ids).bool()

    # Time series input
    seq_len_ts = 60
    ohlcv = torch.randn(batch_size, seq_len_ts, 5)
    indicators = torch.randn(batch_size, seq_len_ts, 20)

    # Test trading task
    print("Testing trading task...")
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ohlcv=ohlcv,
        indicators=indicators,
        task='trading'
    )

    print(f"✓ Trading logits shape: {outputs['trading_logits'].shape}")
    print(f"✓ Price prediction shape: {outputs['price_prediction'].shape}")

    # Test prediction method
    print("\nTesting prediction method...")
    predictions = model.predict_trading_action(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ohlcv=ohlcv,
        indicators=indicators
    )

    print(f"✓ Predicted actions: {predictions['action']}")
    print(f"✓ Action probabilities shape: {predictions['action_probs'].shape}")

    # Test time-series only
    print("\nTesting time-series only mode...")
    ts_only_outputs = model(
        ohlcv=ohlcv,
        indicators=indicators,
        task='trading'
    )
    print(f"✓ TS-only trading logits: {ts_only_outputs['trading_logits'].shape}")

    print("\n✓ All tests passed!")
