#!/usr/bin/env bash

bash setup.sh

#activate virtual environment
source ./env/bin/activate

# run the code
python3 src/assignment.py

# deactive the venv
deactivate