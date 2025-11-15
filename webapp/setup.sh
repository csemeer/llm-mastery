#!/bin/bash

# Financial LLM Trading Bot - Web Application Setup Script
# This script sets up the complete web application environment

set -e

echo "========================================="
echo "Financial LLM Trading Bot - Setup"
echo "========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 found${NC}"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}✗ Node.js is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js found${NC}"

# Check npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}✗ npm is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ npm found${NC}"

echo ""
echo "========================================="
echo "Setting up Backend..."
echo "========================================="
echo ""

cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p data logs ../checkpoints

# Initialize database
echo "Initializing database..."
python3 << EOF
from app import app, db
with app.app_context():
    db.create_all()
    print("✓ Database initialized")
EOF

cd ..

echo ""
echo "========================================="
echo "Setting up Frontend..."
echo "========================================="
echo ""

cd frontend

# Install Node dependencies
echo "Installing Node.js dependencies..."
npm install

cd ..

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To start the application:"
echo ""
echo "Backend (Terminal 1):"
echo "  cd webapp/backend"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "Frontend (Terminal 2):"
echo "  cd webapp/frontend"
echo "  npm start"
echo ""
echo "Or use Docker:"
echo "  docker-compose up"
echo ""
echo "Access the application at: http://localhost:3000"
echo "API documentation: http://localhost:5000/api"
echo ""
echo -e "${GREEN}Happy Trading! 🚀${NC}"
echo ""
