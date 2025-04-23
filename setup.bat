@echo off
REM JuDDGES setup script using UV for Windows

REM Check if UV is installed
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo UV is not installed. Installing UV...
    pip install uv
)

REM Create a virtual environment
echo Creating virtual environment...
uv venv .venv

REM Activate the virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install the project in development mode
echo Installing JuDDGES...
uv pip install -e .

echo Setup complete! JuDDGES environment is ready.
echo To activate this environment in the future, run: .venv\Scripts\activate.bat 