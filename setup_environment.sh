#!/bin/bash
# Environment Setup for LLM Safety Research

echo "Creating Python environment and installing packages..."

# Create virtual environment
python3 -m venv llm_safety_env

# Activate environment
source llm_safety_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

echo "Environment setup complete!"
echo "To activate: source llm_safety_env/bin/activate"