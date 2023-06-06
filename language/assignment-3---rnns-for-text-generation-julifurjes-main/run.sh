#!/usr/bin/env bash

#activate virtual environment
source ./env/bin/activate

# run the code
python3 src/predefined_functions.py
python3 src/assignment.py
python3 src/loading_model.py

# deactive the venv
deactivate