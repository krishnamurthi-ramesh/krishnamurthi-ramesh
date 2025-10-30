# Allow script execution for current process
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Create virtual environment if it doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies..."
python -m pip install -r requirements.txt

Write-Host "Setup complete! Run 'streamlit run app/streamlit_app.py' to start the application."