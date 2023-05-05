#!/usr/bin/env bash

# create virtual environment
python3 -m venv env

#activate virtual environment
source ./env/bin/activate

python3 -m pip install --upgrade pip
python -m pip install -r requirements.txt

pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm