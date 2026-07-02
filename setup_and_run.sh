#!/bin/bash

# --- 1. Environment Setup ---
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

echo "🔌 Activating environment..."
source .venv/bin/activate

echo "📥 Checking dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# --- 2. Frontend Setup ---
if [ -d "./frontend" ]; then
    echo "🌐 Setting up frontend..."
    if ! command -v pnpm &> /dev/null; then
        echo "📦 pnpm not found. Installing pnpm..."
        npm install -g pnpm
    fi
    pushd frontend > /dev/null
    echo "📥 Installing frontend dependencies..."
    pnpm install
    popd > /dev/null
    echo "✅ Frontend setup complete."
fi

# --- 3. Main Menu ---
echo "------------------------------------------------"
echo "✅ Setup Complete. What would you like to do?"
echo "1) Start API Server (api/main.py — port 8001)"
echo "2) Start Frontend Dev Server (port 3000)"
echo "3) Start All (API + Frontend)"
echo "4) Exit"
read -p "Select an option [1-4]: " choice

case $choice in
    1)
        echo "🚀 Starting API Server on http://0.0.0.0:8001..."
        python3 -m api.main
        ;;
    2)
        echo "🌐 Starting Frontend Dev Server on http://localhost:3000..."
        cd frontend && pnpm dev
        ;;
    3)
        echo "🚀 Starting API Server (background) on http://0.0.0.0:8001..."
        python3 -m api.main &
        API_PID=$!
        echo "   API PID: $API_PID"

        trap "echo ''; echo '🛑 Shutting down...'; kill $API_PID 2>/dev/null; exit 0" INT TERM

        echo "🌐 Starting Frontend Dev Server on http://localhost:3000..."
        echo "   Press Ctrl+C to stop both servers."
        cd frontend && pnpm dev

        kill $API_PID 2>/dev/null
        ;;
    4)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid option."
        ;;
esac
