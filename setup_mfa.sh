#!/bin/bash

eval "$(conda shell.bash hook)"

# Check if the "aligner" environment exists
if ! conda env list | grep -q "aligner"; then
  echo "Creating MFA Conda environment..."
  conda create -n aligner -c conda-forge montreal-forced-aligner -y
  echo "MFA environment created."
else
  echo "MFA environment already exists."
fi

# Activate the environment and download models
conda activate aligner  # Required once per terminal session

# Check if the English acoustic model is installed
if ! mfa model list acoustic | grep -q "english_us_arpa"; then
  echo "Downloading English acoustic model..."
  mfa model download acoustic english_us_arpa
else
  echo "English acoustic model already exists."
fi

# Check if the English dictionary is installed
if ! mfa model list dictionary | grep -q "english_us_arpa"; then
  echo "Downloading English dictionary..."
  mfa model download dictionary english_us_arpa
else
  echo "English dictionary already exists."
fi

rm -rf ~/Documents/MFA/temp/*
mfa align ./audio/ english_us_arpa english_us_arpa ./audio/ --clean # Run for each new dataset
echo "Alignment complete."