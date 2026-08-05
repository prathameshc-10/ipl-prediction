#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🏏 Starting Third Umpire Setup Script..."

# 1. Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment 'venv'..."
    python -m venv venv
else
    echo "📦 Virtual environment 'venv' already exists."
fi

# 2. Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# 3. Upgrade pip and install requirements
echo "📥 Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Make python search path recognize current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 5. Run training script to generate model and visual plots
echo "⚙️ Running model training pipeline..."
python src/model_training.py

# 6. Run unit tests
echo "🧪 Running unit tests..."
python -m pytest

echo "🎉 Setup complete! The model has been trained and validated successfully."
echo "--------------------------------------------------------"
echo "To run the FastAPI backend server:"
echo "  source venv/bin/activate"
echo "  uvicorn backend.main:app --reload"
echo "--------------------------------------------------------"
echo "To run the Streamlit frontend:"
echo "  source venv/bin/activate"
echo "  streamlit run frontend/app.py"
echo "--------------------------------------------------------"
