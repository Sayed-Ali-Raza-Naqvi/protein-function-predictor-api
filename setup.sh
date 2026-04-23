#!/bin/bash

echo "Setting up Protein Function Predictor project..."

# Data folders
mkdir -p data/raw
mkdir -p data/processed

# Embeddings and models
mkdir -p embeddings
mkdir -p models

# Notebooks
mkdir -p notebooks

# Source code
mkdir -p src
touch src/__init__.py
touch src/preprocess.py
touch src/embed.py
touch src/train.py
touch src/visualize.py
touch src/predict.py

# API
mkdir -p api
touch api/__init__.py
touch api/main.py

# Root files
touch requirements.txt
touch .gitignore
touch README.md
touch Dockerfile

echo "Directory structure created successfully!"