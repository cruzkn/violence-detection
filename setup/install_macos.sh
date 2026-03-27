#!/bin/bash
# Violence Detection System — macOS Setup
# Tested on macOS 13+ (Ventura/Sonoma), Apple Silicon & Intel
set -e
echo "======================================"
echo " Violence Detection — macOS Setup    "
echo "======================================"

python3 -c "import sys; exit(0 if sys.version_info>=(3,9) else 1)" || { echo "Python 3.9+ required"; exit 1; }
echo "✓ Python OK"

echo "→ Creating virtual environment..."
cd "$(dirname "$0")/.."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel -q

ARCH=$(uname -m)
echo "→ Architecture: $ARCH"
if [ "$ARCH" = "arm64" ]; then
    echo "→ Installing PyTorch for Apple Silicon (MPS)..."
else
    echo "→ Installing PyTorch for Intel Mac (CPU)..."
fi
pip install torch torchvision torchaudio -q
pip install -r requirements.txt -q

echo ""
echo "→ Verifying key packages..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python3 -c "import ultralytics; print('✓ YOLOv8 (Ultralytics) OK')"
python3 -c "import cv2; print(f'✓ OpenCV {cv2.__version__}')"
python3 -c "import flask; print(f'✓ Flask OK')"
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('✓ Apple MPS GPU available — training accelerated')
else:
    print('→ Using CPU (no MPS — functional but slower)')
"
echo ""
echo "======================================"
echo " Setup Complete!"
echo " Activate:  source venv/bin/activate"
echo " Next step: python data/download_datasets.py"
echo "======================================"
