# Web Application Guide

Complete guide for the Financial LLM Trading Bot web application.

## 🎯 Overview

The Financial LLM Trading Bot web application is a production-ready platform for:
- 🤖 Managing AI-powered trading bots
- 🎓 Fine-tuning financial LLM models
- 📊 Real-time trading monitoring
- 📈 Backtesting strategies
- 🔌 Broker integration management

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Web Browser                          │
│                   (React Frontend)                       │
└────────────┬───────────────────────────┬─────────────────┘
             │                           │
             │ HTTP/REST                 │ WebSocket
             │                           │
┌────────────▼───────────────────────────▼─────────────────┐
│                   Flask Backend                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │         API Routes + WebSocket Handlers         │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐     │
│  │   Bot    │  │ Training │  │    Monitoring     │     │
│  │ Manager  │  │ Manager  │  │     Service       │     │
│  └──────────┘  └──────────┘  └───────────────────┘     │
│  ┌─────────────────────────────────────────────────┐    │
│  │           SQLAlchemy ORM + SQLite DB            │    │
│  └─────────────────────────────────────────────────┘    │
└────────────┬───────────────────────────┬─────────────────┘
             │                           │
             │                           │
    ┌────────▼────────┐         ┌───────▼────────┐
    │  Broker APIs    │         │  Financial LLM  │
    │  (Alpaca, etc)  │         │     Models      │
    └─────────────────┘         └─────────────────┘
```

### Component Breakdown

#### Frontend (React)
- **Pages**: Dashboard, Bots, Training, Backtesting, Brokers
- **Components**: Reusable UI components (BotCard, Charts, etc.)
- **Contexts**: Authentication, WebSocket
- **Services**: API client, utilities

#### Backend (Flask)
- **API Routes**: RESTful endpoints for all operations
- **WebSocket**: Real-time updates via Socket.IO
- **Managers**: Bot lifecycle, Training jobs, Monitoring
- **Database**: SQLAlchemy ORM with SQLite

---

## 📊 Features Breakdown

### 1. Dashboard

**Purpose**: Overview of trading activity and performance

**Components**:
- Active bots counter with status indicators
- Real-time PnL display with percentage change
- Win rate visualization
- Recent trades table
- Performance chart (30-day equity curve)

**Real-time Updates**:
- Bot status changes (start/stop)
- New trades executed
- PnL updates every minute

**API Endpoints**:
```
GET /api/dashboard/stats
```

**Response**:
```json
{
  "bots": {
    "total": 5,
    "running": 2,
    "active_list": [...]
  },
  "trades": {
    "total": 125,
    "winning": 73,
    "win_rate": 0.584,
    "recent": [...]
  },
  "performance": {
    "total_pnl": 1250.45,
    "today_pnl": 45.20,
    "pnl_change": 3.75
  }
}
```

---

### 2. Bot Management

**Purpose**: Create, configure, and control trading bots

**Features**:
- ✅ Create new bots with custom configurations
- ⚙️ Configure strategy, symbols, and parameters
- ▶️ Start/Stop bots with one click
- 📊 View real-time performance metrics
- 🗑️ Delete inactive bots

**Bot Configuration Schema**:
```json
{
  "name": "AAPL Momentum Bot",
  "description": "Momentum strategy on Apple stock",
  "broker": "alpaca",
  "symbols": ["AAPL", "MSFT"],
  "strategy": "momentum",
  "config": {
    "timeframe": "1Hour",
    "lookback": 60,
    "threshold": 0.02,
    "max_position_size": 100,
    "risk_per_trade": 0.02
  }
}
```

**Supported Strategies**:
- `momentum` - Momentum-based trading
- `mean_reversion` - Mean reversion strategy
- `llm_direct` - Direct LLM predictions
- `hybrid` - Combined approach

**Bot Lifecycle**:
```
Created → Configured → Running → Paused → Stopped
                ↓
              Error
```

**API Endpoints**:
```
GET    /api/bots              # List all bots
POST   /api/bots              # Create new bot
GET    /api/bots/:id          # Get bot details
PUT    /api/bots/:id          # Update bot
DELETE /api/bots/:id          # Delete bot
POST   /api/bots/:id/start    # Start bot
POST   /api/bots/:id/stop     # Stop bot
GET    /api/bots/:id/performance  # Get performance
```

---

### 3. Model Training

**Purpose**: Fine-tune financial LLM models with custom data

**Features**:
- 🎓 Start new training jobs
- 📊 Real-time progress monitoring
- 📈 Loss and accuracy visualization
- ⏸️ Pause/Resume training
- ❌ Cancel running jobs
- 💾 Download trained models

**Training Configuration**:
```json
{
  "model_type": "fine_tune",
  "config": {
    "base_model": "checkpoints/base_model.pt",
    "tickers": ["AAPL", "GOOGL", "MSFT"],
    "date_range": {
      "start": "2020-01-01",
      "end": "2024-01-01"
    },
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "use_ewc": true,
    "use_lora": true
  }
}
```

**Training Progress Updates** (WebSocket):
```javascript
socket.on('training_progress', (data) => {
  // {
  //   job_id: 1,
  //   epoch: 25,
  //   total_epochs: 50,
  //   progress: 50.0,
  //   loss: 0.325,
  //   accuracy: 0.687
  // }
});
```

**API Endpoints**:
```
GET  /api/training/jobs           # List all jobs
POST /api/training/start          # Start training
GET  /api/training/jobs/:id       # Get job details
POST /api/training/jobs/:id/cancel  # Cancel job
```

---

### 4. Backtesting

**Purpose**: Test trading strategies on historical data

**Features**:
- 📅 Select date range
- 🎯 Choose strategy and symbols
- ⚙️ Configure parameters
- 📊 View performance metrics
- 📈 Visualize equity curve
- 💾 Save backtest results

**Backtest Request**:
```json
{
  "strategy": "momentum",
  "symbols": ["AAPL", "GOOGL"],
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "config": {
    "initial_capital": 100000,
    "timeframe": "1Day",
    "commission": 0.001
  }
}
```

**Backtest Results**:
```json
{
  "summary": {
    "total_return": 15.5,
    "sharpe_ratio": 1.24,
    "max_drawdown": -8.3,
    "win_rate": 0.582,
    "total_trades": 125,
    "avg_trade_duration": "3.2 days"
  },
  "equity_curve": [...],
  "trades": [...],
  "monthly_returns": [...]
}
```

**Performance Metrics**:
- Total Return (%)
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown (%)
- Win Rate (%)
- Profit Factor
- Calmar Ratio

**API Endpoint**:
```
POST /api/backtest
```

---

### 5. Broker Management

**Purpose**: Configure and test broker connections

**Features**:
- 🔌 Add broker credentials
- ✅ Test connections
- 📊 View account balances
- 🔐 Secure credential storage
- 🌐 Support for US and Indian markets

**Supported Brokers**:

**US Markets**:
- Alpaca (commission-free)
- Interactive Brokers (global)
- TD Ameritrade

**Indian Markets**:
- Zerodha Kite
- Angel One
- Upstox

**Broker Configuration**:
```json
{
  "broker": "alpaca",
  "credentials": {
    "api_key": "...",
    "api_secret": "...",
    "paper_trading": true
  }
}
```

**API Endpoints**:
```
GET  /api/brokers           # List supported brokers
POST /api/brokers/test      # Test connection
```

---

## 🔒 Security

### Authentication Flow

```
User Registration
     ↓
Password Hashed (Werkzeug)
     ↓
User Stored in DB
     ↓
User Login
     ↓
JWT Token Generated (7-day expiry)
     ↓
Token Stored in LocalStorage
     ↓
Token Sent with Each Request (Authorization Header)
     ↓
Backend Validates Token
     ↓
Request Processed or 401 Unauthorized
```

### Security Measures

✅ **Password Security**
- Werkzeug password hashing
- Salt and hash generation
- No plaintext storage

✅ **JWT Tokens**
- 7-day expiration
- Secure signing with SECRET_KEY
- Automatic refresh on login

✅ **API Security**
- Token-based authentication
- CORS protection
- SQL injection prevention (SQLAlchemy ORM)
- XSS protection (React escaping)

✅ **Data Security**
- Encrypted broker credentials
- Secure WebSocket connections
- Environment variables for secrets

### Production Security Checklist

- [ ] Change SECRET_KEY to random value
- [ ] Enable HTTPS
- [ ] Configure proper CORS origins
- [ ] Set up rate limiting
- [ ] Enable database encryption
- [ ] Implement audit logging
- [ ] Set up intrusion detection
- [ ] Configure firewall rules
- [ ] Use secure session management
- [ ] Implement 2FA (optional)

---

## 📡 Real-Time Updates

### WebSocket Architecture

```
Client (React) ←→ Socket.IO ←→ Flask-SocketIO ←→ Event Handlers
       ↓                                              ↓
   UI Updates                                   Bot Managers
```

### Event Types

**Bot Events**:
- `bot_status` - Bot started/stopped
- `bot_heartbeat` - Bot is alive (every 60s)
- `bot_error` - Bot encountered error
- `new_trade` - Trade executed

**Training Events**:
- `training_progress` - Training epoch completed
- `training_completed` - Training finished
- `training_failed` - Training error
- `training_cancelled` - Training cancelled

**Dashboard Events**:
- `dashboard_update` - Dashboard stats updated
- `market_data` - New market data available

### Client Implementation

```javascript
import io from 'socket.io-client';

const socket = io('http://localhost:5000');

// Subscribe to bot updates
socket.emit('subscribe_bot', { bot_id: 1 });

// Listen for updates
socket.on('bot_status', (data) => {
  console.log('Bot status:', data);
  // Update UI
});

socket.on('new_trade', (data) => {
  console.log('New trade:', data);
  // Show notification
});

// Unsubscribe when done
socket.emit('unsubscribe_bot', { bot_id: 1 });
```

---

## 🚀 Deployment

### Development

```bash
# Terminal 1 - Backend
cd webapp/backend
source venv/bin/activate
python app.py

# Terminal 2 - Frontend
cd webapp/frontend
npm start
```

### Production - Docker

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production - Manual

```bash
# Build frontend
cd webapp/frontend
npm run build

# Start backend with production server
cd webapp/backend
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 app:app
```

### Environment Variables

Production `.env` file:

```bash
# Flask
SECRET_KEY=<random-secret-key>
FLASK_ENV=production

# Database
DATABASE_URL=sqlite:///trading_bots.db

# CORS
CORS_ORIGINS=https://yourdomain.com

# Logging
LOG_LEVEL=INFO
```

---

## 📊 Database Management

### Migrations

```bash
# Create new migration
flask db migrate -m "Description"

# Apply migrations
flask db upgrade

# Rollback
flask db downgrade
```

### Backups

```bash
# Backup database
sqlite3 trading_bots.db ".backup backup_$(date +%Y%m%d).db"

# Restore database
cp backup_20240101.db trading_bots.db
```

### Database Schema Updates

When modifying models:
1. Update model in `models.py`
2. Create migration
3. Test migration locally
4. Apply to production

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: Database locked error

**Solution**:
```bash
# Stop all services
# Delete database
rm trading_bots.db
# Recreate database
python -c "from app import db, app; app.app_context().push(); db.create_all()"
```

**Issue**: WebSocket not connecting

**Solution**:
- Check CORS settings
- Verify backend is running
- Check browser console
- Try different transport: `{transports: ['polling', 'websocket']}`

**Issue**: Frontend build fails

**Solution**:
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm run build
```

---

## 📈 Performance Optimization

### Backend

- Use Redis for caching
- Implement request caching
- Optimize database queries
- Use connection pooling
- Enable gzip compression

### Frontend

- Code splitting
- Lazy loading
- Image optimization
- Bundle size optimization
- Service Workers for offline support

### Database

- Create indexes on frequently queried columns
- Optimize query patterns
- Regular VACUUM operations
- Monitor query performance

---

## 🧪 Testing

### Backend Tests

```bash
cd webapp/backend
pytest tests/ -v
```

### Frontend Tests

```bash
cd webapp/frontend
npm test
```

### E2E Tests

```bash
npm run test:e2e
```

---

## 📚 API Documentation

Complete API documentation available at:
- Development: `http://localhost:5000/api/docs`
- Production: `https://yourdomain.com/api/docs`

Or see [API.md](API.md) for full reference.

---

## 🎓 Best Practices

### Code Organization

- Keep components small and focused
- Use custom hooks for reusable logic
- Implement error boundaries
- Use TypeScript for type safety
- Follow consistent naming conventions

### State Management

- Use React Context for global state
- Keep state as local as possible
- Use memoization for expensive computations
- Implement proper loading states

### Error Handling

- Implement global error boundary
- Show user-friendly error messages
- Log errors to monitoring service
- Provide retry mechanisms

---

## 📞 Support & Resources

- **Documentation**: Full documentation in `/docs`
- **API Reference**: See `API.md`
- **GitHub Issues**: Report bugs and request features
- **Community**: Join our Discord/Slack

---

**Happy Trading! 🚀**
