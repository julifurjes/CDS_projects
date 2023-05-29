#!/usr/bin/env bash

#activate virtual environment
source ./env/bin/activate

# run the code
python3 utils/vectorizer_file.py
python3 src/logistic_regression_2.py
python3 src/neural_network_2.py

# deactive the venv
deactivate