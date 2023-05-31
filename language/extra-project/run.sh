#!/usr/bin/env bash

#activate virtual environment
source ./env/bin/activate

# run the code
python3 utils/classifier_utils.py
python3 src/assignment.py

# deactive the venv
deactivate