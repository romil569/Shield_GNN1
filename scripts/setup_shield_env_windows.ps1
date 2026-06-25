$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

function Test-CommandAvailable {
    param([string]$Command)
    return $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

function Test-PyVersion {
    param([string]$Version)
    if (-not (Test-CommandAvailable "py")) {
        return $false
    }

    & py "-$Version" --version *> $null
    return $LASTEXITCODE -eq 0
}

function Get-PythonCommand {
    if (Test-PyVersion "3.11") {
        return "py -3.11"
    }
    if (Test-PyVersion "3.10") {
        return "py -3.10"
    }
    if (Test-CommandAvailable "python") {
        return "python"
    }

    $bundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path $bundledPython) {
        return "`"$bundledPython`""
    }

    throw "No suitable Python interpreter found. Install Python 3.11 or 3.10, then rerun this script."
}

$pythonCommand = Get-PythonCommand

if (-not (Test-Path "shield_env")) {
    Invoke-Expression "$pythonCommand -m venv shield_env"
}

.\shield_env\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name shield-gnn-env --display-name "SHIELD-GNN (shield_env)"
python scripts\check_dataset_files.py
python scripts\verify_setup.py

Write-Host "SHIELD-GNN shield_env setup and verification complete."
