#!/bin/bash

python3 -m venv .venv

source .venv/bin/activate

if [ -f "requirements.txt" ]; then
    # Install packages using pip
    pip install -r requirements.txt
else
    echo "requirements.txt file not found."
fi

echo "Python packages installed successfully."
echo "Installing FFmpeg for animation saving."

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    # Install FFmpeg using apt-get
    if command -v apt-get &> /dev/null; then
        sudo apt-get install ffmpeg
    # Install FFmpeg using yum
    elif command -v yum &> /dev/null; then
        sudo yum install ffmpeg
    # Install FFmpeg using dnf
    elif command -v dnf &> /dev/null; then
        sudo dnf install ffmpeg
    # Install FFmpeg using pacman
    elif command -v pacman &> /dev/null; then
        sudo pacman -S ffmpeg
    # Install FFmpeg using brew
    elif command -v brew &> /dev/null; then
        brew install ffmpeg
    else
        echo "Unable to install FFmpeg. Please install it manually."
    fi
else
    echo "FFmpeg is already installed."
fi

echo "Virtual environment activated and packages installed successfully."
echo "To deactivate the virtual environment, run 'deactivate'."