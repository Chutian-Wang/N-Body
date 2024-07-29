#!/bin/bash

python3 -m venv .venv

source .venv/bin/activate

if [ -f "requirements.txt" ]; then
    # Install packages using pip
    pip install -r requirements.txt
else
    echo "requirements.txt file not found."
fi

deactivate