$ENV_NAME = "cartpole"

Write-Host "Creating conda environment: $ENV_NAME"

conda env create -f environment.yml

Write-Host "Environment created!"

Write-Host "Activating environment..."
conda activate $ENV_NAME

Write-Host "Installing extra pip packages..."
pip install pygame --upgrade

Write-Host ""
Write-Host "Setup complete!"
Write-Host "To activate the environment later, run:"
Write-Host "conda activate $ENV_NAME"
