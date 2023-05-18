#!/usr/bin/env bash

#activate virtual environment
source ./env/bin/activate

# run the code
python3 utils/preprocessing2.py
python3 src/log_regression.py
python3 src/neural_network.py

# deactive the venv
deactivate