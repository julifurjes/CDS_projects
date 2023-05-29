#!/usr/bin/env bash

# create virtual environment
python -m venv env

#activate virtual environment
source ./env/bin/activate

python -m pip install -r requirements.txt