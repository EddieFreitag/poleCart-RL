#!/bin/bash

# Exit on error
set -e

ENV_NAME="cartpole"

echo "Creating conda environment: $ENV_NAME"

# Create environment from environment.yml
conda env create -f environment.yml

echo "Environment created!"

echo "Activating environment..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Installing extra pip packages if necessary..."
pip install pygame --upgrade

echo
echo "Setup complete!"
echo "To activate the environment later, run:"
echo "conda activate $ENV_NAME"
