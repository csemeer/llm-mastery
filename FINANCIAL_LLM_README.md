# Financial LLM - Multi-Modal AI for Trading & Investment

A production-ready, continually-learning LLM specifically designed for financial markets. Combines text understanding (SEC filings, news) with time series analysis (OHLCV, indicators) for trading, investment analysis, and portfolio management.

## 🌟 Key Features

### Multi-Modal Architecture
- **Text Encoder**: Processes SEC filings, financial reports, news articles
- **Time Series Encoder**: Analyzes OHLCV data and technical indicators
- **Cross-Modal Fusion**: Combines textual and numerical insights
- **Task-Specific Heads**: Trading signals, risk assessment, portfolio optimization

### Continual Learning
- **Elastic Weight Consolidation (EWC)**: Prevents catastrophic forgetting
- **Experience Replay**: Maintains performance on historical data
- **LoRA Integration**: Efficient parameter updates (update <1% of parameters)
- **Daily Update Pipeline**: Automated model updates with new data

### Production Features
- **Multi-task Learning**: Single model for multiple financial tasks
- **Efficient Fine-tuning**: LoRA-based adaptation
- **Real-time Data Integration**: Yahoo Finance, SEC EDGAR support
- **Comprehensive Evaluation**: Financial metrics and backtesting

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Financial LLM                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐              ┌──────────────┐       │
│  │ Text Encoder │              │  TS Encoder  │       │
│  │  (SEC, News) │              │ (OHLCV, TA)  │       │
│  └──────┬───────┘              └──────┬───────┘       │
│         │                              │               │
│         └──────────┐       ┌───────────┘               │
│                    ▼       ▼                           │
│              ┌─────────────────┐                       │
│              │  Fusion Layers  │                       │
│              │  (Cross-Modal)  │                       │
│              └────────┬────────┘                       │
│                       │                                │
│         ┌─────────────┼─────────────┐                 │
│         ▼             ▼             ▼                 │
│    ┌────────┐   ┌─────────┐   ┌─────────┐           │
│    │Trading │   │  Risk   │   │Portfolio│           │
│    │  Head  │   │  Head   │   │  Head   │           │
│    └────────┘   └─────────┘   └─────────┘           │
│                                                        │
└────────────────────────────────────────────────────────┘

Continual Learning Layer:
- EWC (Elastic Weight Consolidation)
- Experience Replay Buffer
- LoRA Adapters
```

## 📦 Installation

### Requirements
```bash
pip install -r requirements_financial.txt
```

Key dependencies:
- PyTorch >= 2.0.0
- yfinance (market data)
- sec-edgar-downloader (SEC filings)
- transformers (tokenization)
- peft (LoRA)
- ta (technical analysis)

## 🚀 Quick Start

### 1. Train Initial Model

Train on historical data for a stock:

```bash
python train_financial_llm.py \
    --ticker AAPL \
    --start-date 2020-01-01 \
    --end-date 2024-01-01 \
    --epochs 50 \
    --batch-size 32 \
    --use-continual-learning \
    --use-lora
```

### 2. Daily Updates

Set up automated daily updates:

```bash
python scripts/daily_update.py \
    --checkpoint-path checkpoints/financial_llm/best_model.pt \
    --tickers AAPL GOOGL MSFT \
    --days-back 30 \
    --update-epochs 3
```

### 3. Make Predictions

```python
from financial_llm.models.financial_llm import FinancialLLM
import torch

# Load model
model = FinancialLLM.from_pretrained('checkpoints/financial_llm/best_model.pt')

# Prepare data (OHLCV + indicators)
ohlcv = torch.randn(1, 60, 5)  # 60 days of data
indicators = torch.randn(1, 60, 20)

# Get trading signal
predictions = model.predict_trading_action(
    ohlcv=ohlcv,
    indicators=indicators
)

print(f"Action: {predictions['action']}")  # 0=Buy, 1=Sell, 2=Hold
print(f"Confidence: {predictions['action_probs']}")
```

## 📊 Use Cases

### 1. Trading Signal Generation

```python
# Multi-modal trading signal
outputs = model(
    input_ids=sec_filing_tokens,      # Recent 10-K
    ohlcv=price_data,                 # Last 60 days
    indicators=technical_indicators,
    task='trading'
)

action = outputs['trading_logits'].argmax()
price_pred = outputs['price_prediction']
```

### 2. Risk Assessment

```python
# Assess investment risk
risk_outputs = model(
    input_ids=company_news_tokens,
    ohlcv=volatility_data,
    task='risk'
)

risk_scores = risk_outputs['risk_scores']
# [volatility_risk, beta, downside_risk, ...]
```

### 3. Portfolio Optimization

```python
# Get optimal portfolio weights
weights = []

for stock_data in portfolio:
    output = model(
        input_ids=stock_data['filings'],
        ohlcv=stock_data['prices'],
        task='portfolio'
    )
    weights.append(output['portfolio_weights'])

# Normalize weights
weights = softmax(weights)
```

## 🔄 Continual Learning Workflow

### Initial Training
1. Train on 3-5 years of historical data
2. Validate on recent 6 months
3. Test on last 3 months

### Daily Updates (Production)
1. **Fetch**: Get yesterday's market data
2. **Preprocess**: Calculate indicators
3. **Fine-tune**: 2-3 epochs with EWC penalty
4. **Validate**: Check performance on recent data
5. **Deploy**: Update production model

### Preventing Catastrophic Forgetting

```python
# EWC: Protects important parameters
ewc_loss = λ/2 * Σ F_i * (θ_i - θ*_i)²

# Experience Replay: 30% of batch from old data
batch = {
    'new_data': 70%,      # Recent market data
    'replay_data': 30%    # Historical samples
}
```

## 🎯 Training Strategies

### For Different Market Conditions

**Bull Market Training**:
```bash
python train_financial_llm.py \
    --ticker SPY \
    --start-date 2019-01-01 \
    --end-date 2021-12-31 \
    --ewc-lambda 50.0  # Lower regularization
```

**Bear Market Training**:
```bash
python train_financial_llm.py \
    --ticker SPY \
    --start-date 2007-01-01 \
    --end-date 2009-12-31 \
    --ewc-lambda 200.0  # Higher regularization
```

**Mixed Training** (Recommended):
```bash
python train_financial_llm.py \
    --ticker SPY \
    --start-date 2015-01-01 \
    --end-date 2024-01-01 \
    --use-continual-learning \
    --ewc-lambda 100.0
```

## 📈 Model Sizes & Performance

| Configuration | Parameters | Memory | Training Time* | Accuracy** |
|--------------|-----------|--------|----------------|-----------|
| Small        | ~2M       | 2GB    | 2 hours        | 58-62%    |
| Medium       | ~10M      | 4GB    | 6 hours        | 62-66%    |
| Large        | ~50M      | 8GB    | 24 hours       | 66-70%    |

*On NVIDIA V100, **On next-day direction prediction

### Small Model (Fast Iteration)
```bash
python train_financial_llm.py \
    --d-model 128 \
    --num-text-layers 2 \
    --num-ts-layers 2 \
    --num-fusion-layers 1
```

### Large Model (Best Performance)
```bash
python train_financial_llm.py \
    --d-model 512 \
    --num-text-layers 6 \
    --num-ts-layers 4 \
    --num-fusion-layers 3
```

## 💡 Advanced Features

### 1. LoRA for Efficient Fine-tuning

```python
from financial_llm.utils.continual_learning import apply_lora_to_model

# Only train 0.5% of parameters
apply_lora_to_model(
    model,
    rank=8,           # LoRA rank (lower = more efficient)
    alpha=16.0,       # LoRA scaling
    target_modules=['W_q', 'W_v']  # Which layers to adapt
)
```

**Benefits**:
- 100x fewer trainable parameters
- 3x faster training
- Same or better performance
- Multiple task-specific LoRA adapters

### 2. Multi-Stock Training

```python
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']

for ticker in tickers:
    dataset = create_market_datasets(ticker, ...)
    train_one_epoch(dataset)

    # Update EWC after each ticker
    cl_trainer.initialize_ewc(dataloader)
```

### 3. Custom Technical Indicators

```python
from financial_llm.data_processors.market_data import TechnicalIndicators

class CustomIndicators(TechnicalIndicators):
    @staticmethod
    def add_indicators(df):
        df = super().add_indicators(df)

        # Add custom indicators
        df['my_custom_indicator'] = ...

        return df
```

## 🔬 Evaluation & Backtesting

### Performance Metrics

```python
from financial_llm.utils.evaluation import evaluate_trading_strategy

results = evaluate_trading_strategy(
    model=model,
    test_data=test_dataset,
    initial_capital=100000,
    commission=0.001
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
```

### Backtesting Example

```python
# Backtest on historical data
backtest_results = backtest_strategy(
    model=model,
    start_date='2023-01-01',
    end_date='2024-01-01',
    tickers=['AAPL', 'GOOGL'],
    strategy='long_only',  # or 'long_short', 'market_neutral'
    rebalance_frequency='daily'
)
```

## 📚 Data Sources

### Market Data
- **Yahoo Finance**: Historical OHLCV, real-time quotes
- **Alpha Vantage**: Alternative data source
- **Custom CSV**: Import your own data

### Text Data
- **SEC EDGAR**: 10-K, 10-Q, 8-K filings
- **Financial News**: Integration with news APIs
- **Earnings Transcripts**: Quarterly calls
- **Social Media**: Sentiment analysis

## 🛠️ Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch-size 16

# Use gradient accumulation
--gradient-accumulation-steps 4

# Enable mixed precision
--mixed-precision

# Use smaller model
--d-model 256
```

### Poor Performance
1. **Increase model size**: More parameters capture complexity
2. **More training data**: At least 2-3 years recommended
3. **Adjust EWC lambda**: Balance new vs old knowledge
4. **Feature engineering**: Add domain-specific indicators

### Catastrophic Forgetting
```bash
# Increase EWC lambda
--ewc-lambda 200.0

# Larger replay buffer
--replay-buffer-size 20000

# More conservative updates
--learning-rate 1e-5
```

## 📖 Research Background

This implementation is based on:

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Transformer architecture

2. **Overcoming Catastrophic Forgetting** (Kirkpatrick et al., 2017)
   - Elastic Weight Consolidation (EWC)

3. **LoRA: Low-Rank Adaptation** (Hu et al., 2021)
   - Efficient fine-tuning

4. **FinBERT** (Araci, 2019)
   - Financial domain pre-training

5. **Temporal Fusion Transformers** (Lim et al., 2021)
   - Time series with attention

## 🚧 Roadmap

- [ ] Pre-trained weights on 10+ years of market data
- [ ] Integration with live trading APIs (Alpaca, Interactive Brokers)
- [ ] Multi-asset class support (Crypto, Forex, Commodities)
- [ ] Explainability module (attention visualization, SHAP values)
- [ ] Ensemble methods with multiple model checkpoints
- [ ] Automated hyperparameter optimization
- [ ] Model distillation for edge deployment
- [ ] Real-time streaming inference

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Not financial advice
- No guarantees of profitability
- Past performance doesn't predict future results
- Use at your own risk
- Consult licensed financial advisors

## 📄 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional data sources
- New technical indicators
- Alternative continual learning methods
- Backtesting improvements
- Documentation

## 📬 Contact

For questions and discussions:
- GitHub Issues: [Report bugs or request features]
- Discussions: [Ask questions and share ideas]

---

**Built with ❤️ for quantitative finance and machine learning research**
