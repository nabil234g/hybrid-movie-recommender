@echo off
title Hybrid Movie Recommendation System

echo.
echo  Setting up environment...
echo.

where py >nul 2>&1
if %errorlevel% neq 0 (
    echo  Python Launcher not found. Installing Python 3.12...
    winget install Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
)

if not exist "%~dp0venv\" (
    echo  Creating virtual environment...
    py -3.12 -m venv "%~dp0venv"
    echo  Installing packages (this takes 2-3 minutes on first run)...
    "%~dp0venv\Scripts\pip.exe" install streamlit pandas "numpy<2" scikit-learn scikit-surprise
)

echo.
echo  Launching app...
echo.
"%~dp0venv\Scripts\streamlit.exe" run "%~dp0app.py"

pause
