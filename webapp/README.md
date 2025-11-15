# Financial LLM Trading Bot - Web Application

Production-ready web application for managing AI-powered trading bots, fine-tuning models, and monitoring trading activity in real-time.

## 🎯 Features

### Dashboard
- 📊 Real-time performance metrics
- 📈 PnL tracking and charts
- 🤖 Active bot monitoring
- 📋 Recent trades overview

### Bot Management
- ✅ Create and configure trading bots
- ▶️ Start/Stop bots with one click
- ⚙️ Configure strategies and parameters
- 📊 View bot performance metrics
- 🔄 Real-time status updates via WebSocket

### Model Training
- 🎓 Fine-tune LLM models
- 📊 Real-time training progress
- 📈 Loss and accuracy charts
- 💾 Save and manage trained models
- 🔄 Resume interrupted training

### Backtesting
- 📉 Visual backtesting interface
- 📊 Performance metrics (Sharpe, max drawdown, win rate)
- 📈 Equity curve visualization
- 🎯 Strategy comparison

### Broker Integration
- 🔌 Connect to multiple brokers (US & India)
- 🔐 Secure credential management
- ✅ Test connections before deployment
- 📊 View account balances and positions

---

## 🏗️ Architecture

```
webapp/
├── backend/                    # Flask API Server
│   ├── app.py                  # Main Flask application
│   ├── models.py               # Database models
│   ├── bot_manager.py          # Bot lifecycle management
│   ├── training_manager.py     # Training job management
│   └── monitoring.py           # Real-time monitoring
│
├── frontend/                   # React Application
│   ├── public/
│   ├── src/
│   │   ├── components/         # Reusable components
│   │   │   ├── Layout.js
│   │   │   ├── BotCard.js
│   │   │   ├── TrainingProgress.js
│   │   │   └── PerformanceChart.js
│   │   ├── pages/              # Page components
│   │   │   ├── Dashboard.js
│   │   │   ├── Bots.js
│   │   │   ├── BotDetail.js
│   │   │   ├── Training.js
│   │   │   ├── Backtesting.js
│   │   │   └── Brokers.js
│   │   ├── contexts/           # React contexts
│   │   │   ├── AuthContext.js
│   │   │   └── SocketContext.js
│   │   ├── services/
│   │   │   └── api.js          # Axios API client
│   │   └── App.js
│   └── package.json
│
└── README.md
```

### Technology Stack

**Backend:**
- Flask - Web framework
- Flask-SocketIO - WebSocket support
- SQLAlchemy - ORM
- JWT - Authentication
- SQLite - Database

**Frontend:**
- React 18 - UI framework
- Material-UI (MUI) - Component library
- Socket.IO Client - Real-time updates
- Axios - HTTP client
- Chart.js & Recharts - Data visualization
- React Router - Routing

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### 1. Install Backend Dependencies

```bash
cd webapp/backend
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd webapp/frontend
npm install
```

### 3. Start Development Servers

**Backend (Terminal 1):**
```bash
cd webapp/backend
python app.py
```

Server starts at: `http://localhost:5000`

**Frontend (Terminal 2):**
```bash
cd webapp/frontend
npm start
```

App opens at: `http://localhost:3000`

---

## 📦 Production Deployment

### Using Docker Compose

```bash
cd webapp
docker-compose up -d
```

The application will be available at `http://localhost:5000`

### Manual Deployment

**1. Build Frontend:**
```bash
cd webapp/frontend
npm run build
```

**2. Start Production Server:**
```bash
cd webapp/backend
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 app:app
```

---

## 🔐 Authentication

### Register New User

```bash
POST /api/auth/register
{
  "email": "user@example.com",
  "username": "trader1",
  "password": "secure_password"
}
```

### Login

```bash
POST /api/auth/login
{
  "email": "user@example.com",
  "password": "secure_password"
}
```

Returns JWT token for authenticated requests.

---

## 📡 API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | User login |
| GET | `/api/auth/me` | Get current user |

### Trading Bots

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/bots` | Get all bots |
| POST | `/api/bots` | Create new bot |
| GET | `/api/bots/:id` | Get bot details |
| PUT | `/api/bots/:id` | Update bot |
| DELETE | `/api/bots/:id` | Delete bot |
| POST | `/api/bots/:id/start` | Start bot |
| POST | `/api/bots/:id/stop` | Stop bot |
| GET | `/api/bots/:id/performance` | Get performance |

### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/training/jobs` | Get all training jobs |
| POST | `/api/training/start` | Start training |
| GET | `/api/training/jobs/:id` | Get job details |
| POST | `/api/training/jobs/:id/cancel` | Cancel job |

### Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/dashboard/stats` | Get dashboard stats |
| GET | `/api/trades` | Get trade history |
| POST | `/api/backtest` | Run backtest |

### Brokers

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/brokers` | Get supported brokers |
| POST | `/api/brokers/test` | Test broker connection |

---

## 🔌 WebSocket Events

### Client → Server

```javascript
// Subscribe to bot updates
socket.emit('subscribe_bot', { bot_id: 1 });

// Unsubscribe from bot updates
socket.emit('unsubscribe_bot', { bot_id: 1 });
```

### Server → Client

```javascript
// Bot status update
socket.on('bot_status', (data) => {
  // { bot_id: 1, status: 'running' }
});

// New trade executed
socket.on('new_trade', (data) => {
  // { bot_id: 1, trade: {...} }
});

// Training progress
socket.on('training_progress', (data) => {
  // { job_id: 1, epoch: 10, progress: 20, loss: 0.45 }
});

// Training completed
socket.on('training_completed', (data) => {
  // { job_id: 1, model_path: '...' }
});
```

---

## 🎨 Frontend Components

### Dashboard

```jsx
import Dashboard from './pages/Dashboard';

// Shows:
// - Active bots count
// - Total PnL
// - Win rate
// - Recent trades
// - Performance chart
```

### Bot Management

```jsx
import Bots from './pages/Bots';
import BotDetail from './pages/BotDetail';

// Features:
// - List all bots
// - Create new bot
// - Start/stop bots
// - View detailed performance
// - Configure strategy
```

### Training Interface

```jsx
import Training from './pages/Training';

// Features:
// - Start fine-tuning
// - Monitor training progress
// - View loss/accuracy charts
// - Cancel running jobs
// - Download trained models
```

### Backtesting

```jsx
import Backtesting from './pages/Backtesting';

// Features:
// - Select strategy and symbols
// - Set date range
// - Run backtest
// - View equity curve
// - Compare strategies
```

---

## 🗄️ Database Schema

### Users
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email VARCHAR(120) UNIQUE NOT NULL,
    username VARCHAR(80) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at DATETIME,
    last_login DATETIME
);
```

### Trading Bots
```sql
CREATE TABLE trading_bots (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    broker VARCHAR(50) NOT NULL,
    symbols TEXT NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    config TEXT,
    status VARCHAR(20),
    model_path VARCHAR(255),
    created_at DATETIME,
    updated_at DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### Training Jobs
```sql
CREATE TABLE training_jobs (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    config TEXT,
    status VARCHAR(20),
    progress FLOAT,
    current_epoch INTEGER,
    total_epochs INTEGER,
    loss FLOAT,
    accuracy FLOAT,
    model_path VARCHAR(255),
    created_at DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### Trades
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    bot_id INTEGER NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity FLOAT NOT NULL,
    entry_price FLOAT NOT NULL,
    exit_price FLOAT,
    pnl FLOAT,
    status VARCHAR(20),
    executed_at DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (bot_id) REFERENCES trading_bots(id)
);
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` file in `webapp/backend/`:

```bash
# Flask
SECRET_KEY=your-secret-key-change-in-production
FLASK_ENV=development

# Database
DATABASE_URL=sqlite:///trading_bots.db

# CORS
CORS_ORIGINS=http://localhost:3000

# WebSocket
SOCKETIO_ASYNC_MODE=eventlet
```

### Frontend Configuration

Edit `webapp/frontend/src/config.js`:

```javascript
export const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
export const WS_URL = process.env.REACT_APP_WS_URL || 'http://localhost:5000';
```

---

## 🧪 Testing

### Backend Tests

```bash
cd webapp/backend
pytest tests/
```

### Frontend Tests

```bash
cd webapp/frontend
npm test
```

---

## 📊 Monitoring & Logging

### Application Logs

Logs are stored in `webapp/backend/logs/`:
- `app.log` - Application logs
- `trading.log` - Trading activity
- `training.log` - Training jobs

### Real-time Monitoring

Access monitoring dashboard at: `http://localhost:5000/dashboard`

---

## 🛡️ Security

### Best Practices

✅ JWT authentication with expiration
✅ Password hashing with Werkzeug
✅ CORS protection
✅ SQL injection prevention (SQLAlchemy ORM)
✅ XSS protection (React escaping)
✅ Secure WebSocket connections

### Production Checklist

- [ ] Change `SECRET_KEY` in production
- [ ] Use HTTPS for deployment
- [ ] Enable rate limiting
- [ ] Set up database backups
- [ ] Configure proper CORS origins
- [ ] Use environment variables for secrets
- [ ] Enable logging and monitoring

---

## 🚀 Performance Optimization

### Backend

- Use Redis for session storage
- Implement caching for frequently accessed data
- Use Celery for background tasks
- Optimize database queries

### Frontend

- Code splitting with React.lazy()
- Memoization with React.memo()
- Virtual scrolling for large lists
- Optimize bundle size

---

## 📚 Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [React Documentation](https://react.dev/)
- [Material-UI Documentation](https://mui.com/)
- [Socket.IO Documentation](https://socket.io/)

---

## 🐛 Troubleshooting

### Backend won't start

```bash
# Check dependencies
pip install -r requirements.txt

# Check database
python
>>> from app import db, app
>>> with app.app_context():
>>>     db.create_all()
```

### Frontend won't start

```bash
# Clear cache
rm -rf node_modules package-lock.json
npm install

# Check Node version
node --version  # Should be 16+
```

### WebSocket not connecting

- Check CORS settings
- Verify backend is running
- Check browser console for errors
- Ensure port 5000 is not blocked

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📞 Support

- GitHub Issues: [Create an issue](https://github.com/yourusername/llm-mastery/issues)
- Documentation: [Full docs](https://docs.yourproject.com)
- Email: support@yourproject.com

---

**Built with ❤️ for algorithmic traders**
