#!/bin/bash

# --- 1. Environment Setup ---
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

echo "🔌 Activating environment..."
source .venv/bin/activate

# Check if pip is up to date and install requirements
echo "📥 Checking dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# --- 2. ClearML Initialization ---
if [ ! -f "clearml.conf" ]; then
    read -p "❓ ClearML config not found. Initialize now? (y/n): " init_clearml
    if [[ $init_clearml == "y" ]]; then
        clearml-init
    fi
fi

# --- 3. Data & Preprocessing Check ---
if [ ! -d "./data" ]; then
    read -p "⚠️ Raw data (GRID) not found. Download it now? (y/n): " dl_data
    if [[ $dl_data == "y" ]]; then
        python3 scripts/dataset/download_data.py
    fi
fi

if [ ! -d "./data/preprocessed" ]; then
    echo "⚠️ Preprocessed data not found."
    read -p "⚙️ Run preprocessing now? (This may take a while) (y/n): " run_pre
    if [[ $run_pre == "y" ]]; then
        python3 scripts/preprocess.py
    fi
fi

# --- 4. Frontend Setup ---
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

# --- 5. Main Menu ---
echo "------------------------------------------------"
echo "✅ Setup Complete. What would you like to do?"
echo "1) Train the model (train.py)"
echo "2) Evaluate model on test data (test.py)"
echo "3) Run Live Inference (inference.py)"
echo "4) Start Frontend Dev Server"
echo "5) Exit"
read -p "Select an option [1-5]: " choice

case $choice in
    1)
        echo "🚀 Starting Training..."
        python3 train.py
        ;;
    2) python3 test.py ;;
    3)
        read -p "Enter IP Webcam URL (e.g., http://192.168.0.69:8080): " ip_url
        echo "👁️ Starting Inference on $ip_url..."
        python3 inference.py --ip "$ip_url"
        ;;
    4)
        echo "🌐 Starting Frontend Dev Server..."
        cd frontend && pnpm dev
        ;;
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid option."
        ;;
esac
