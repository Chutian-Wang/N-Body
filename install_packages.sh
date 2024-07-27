#!/bin/bash

# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Check if requirements.txt file exists
if [ -f "requirements.txt" ]; then
    # Install packages using pip
    pip install -r requirements.txt
else
    echo "requirements.txt file not found."
fi

# Deactivate the virtual environment
deactivate