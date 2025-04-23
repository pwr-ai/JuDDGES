#!/bin/bash
# JuDDGES setup script using UV

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    pip install uv
fi

# Create a virtual environment
echo "Creating virtual environment..."
uv venv .venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install the project in development mode
echo "Installing JuDDGES..."
uv pip install -e .

echo "Setup complete! JuDDGES environment is ready."
echo "To activate this environment in the future, run: source .venv/bin/activate" 