#!/bin/bash
# Setup script for Chatterbox TTS with MLX optimization on Apple Silicon
# Usage: bash setup_mlx.sh

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Chatterbox TTS - MLX Setup for Apple Silicon             ║"
echo "║  Optimized for M1/M2/M3/M4 Macs                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ Error: This script is for macOS only."
    echo "   For other platforms, use: pip install chatterbox-tts"
    exit 1
fi

# Check if running on Apple Silicon
arch=$(uname -m)
if [[ "$arch" != "arm64" ]]; then
    echo "⚠️  Warning: You're not running on Apple Silicon (detected: $arch)"
    echo "   MLX optimization works best on M1/M2/M3/M4 chips."
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
fi

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python $required_version or higher is required."
    echo "   You have Python $python_version"
    echo "   Please upgrade: brew install python@3.11"
    exit 1
fi

echo "✓ Python $python_version detected"
echo "✓ Running on macOS ($arch)"
echo ""

# Check if virtual environment is active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: No virtual environment detected."
    echo "   It's recommended to use a virtual environment."
    echo ""
    read -p "Create a new virtual environment? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo "Creating virtual environment 'venv'..."
        python3 -m venv venv
        echo "Activating virtual environment..."
        source venv/bin/activate
        echo "✓ Virtual environment created and activated"
        echo ""
    fi
fi

# Install MLX dependencies
echo "Step 1: Installing MLX and dependencies..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "requirements-mlx.txt" ]; then
    pip install -r requirements-mlx.txt
    echo "✓ MLX dependencies installed"
else
    echo "❌ Error: requirements-mlx.txt not found"
    echo "   Are you in the chatterbox directory?"
    exit 1
fi

echo ""

# Install Chatterbox
echo "Step 2: Installing Chatterbox TTS..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "pyproject.toml" ]; then
    pip install -e .
    echo "✓ Chatterbox installed in development mode"
elif [ -f "setup.py" ]; then
    pip install -e .
    echo "✓ Chatterbox installed in development mode"
else
    # Fallback to PyPI
    pip install chatterbox-tts
    echo "✓ Chatterbox installed from PyPI"
fi

echo ""

# Verify installation
echo "Step 3: Verifying installation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 -c "
import sys
try:
    import mlx.core as mx
    print('✓ MLX:', mx.__version__)
except ImportError:
    print('❌ MLX: Not found')
    sys.exit(1)

try:
    import torch
    print('✓ PyTorch:', torch.__version__)
except ImportError:
    print('❌ PyTorch: Not found')
    sys.exit(1)

try:
    from chatterbox.tts import ChatterboxTTS
    print('✓ Chatterbox TTS: Installed')
except ImportError:
    print('❌ Chatterbox TTS: Not found')
    sys.exit(1)

print('')
print('All components installed successfully!')
" || {
    echo ""
    echo "❌ Installation verification failed. Please check the errors above."
    exit 1
}

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Installation Complete! 🎉                                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo ""
echo "  1. Run the MLX example:"
echo "     $ python example_mlx.py"
echo ""
echo "  2. Or try it in Python:"
echo "     $ python3"
echo "     >>> from chatterbox.tts import ChatterboxTTS"
echo "     >>> model = ChatterboxTTS.from_pretrained(device='cpu')"
echo "     >>> wav = model.generate('Hello from Chatterbox!')"
echo ""
echo "  3. Read the MLX guide for advanced usage:"
echo "     $ cat README_MLX.md"
echo ""
echo "Need help? Join our Discord: https://discord.gg/rJq9cRJBJ6"
echo ""
