#!/usr/bin/env bash
python3.9 -m pip install --upgrade pip
python3.9 -m pip install -r requirements.txt

pip install -U pip setuptools wheel
pip install -U spacy
python3.9 -m spacy download en_core_web_sm