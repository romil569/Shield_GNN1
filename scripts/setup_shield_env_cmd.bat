@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_CMD="

py -3.11 --version >nul 2>&1
if "%ERRORLEVEL%"=="0" set "PYTHON_CMD=py -3.11"

if not defined PYTHON_CMD (
    py -3.10 --version >nul 2>&1
    if "%ERRORLEVEL%"=="0" set "PYTHON_CMD=py -3.10"
)

if not defined PYTHON_CMD (
    where python >nul 2>&1
    if "%ERRORLEVEL%"=="0" set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
        set "PYTHON_CMD=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    )
)

if not defined PYTHON_CMD (
    echo No suitable Python interpreter found. Install Python 3.11 or 3.10, then rerun this script.
    exit /b 1
)

if not exist "shield_env" (
    %PYTHON_CMD% -m venv shield_env
)

call shield_env\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name shield-gnn-env --display-name "SHIELD-GNN (shield_env)"
python scripts\check_dataset_files.py
python scripts\verify_setup.py

echo SHIELD-GNN shield_env setup and verification complete.
endlocal
