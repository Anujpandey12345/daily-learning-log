#!/bin/bash

echo "🚀 Installing OCR & Computer Vision Dependencies..."

# Update packages
sudo apt update

# Install system dependencies
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-full \
    tesseract-ocr \
    libtesseract-dev \
    unzip

# Create virtual environment
python3 -m venv env

# Activate environment
source env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python packages
pip install \
    opencv-python \
    pytesseract \
    easyocr \
    boto3 \
    pillow \
    matplotlib \
    numpy \
    torch \
    torchvision

echo "✅ Setup Complete!"
echo "👉 Activate env using:"
echo "source env/bin/activate"