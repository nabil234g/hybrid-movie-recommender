@echo off
title Hybrid Movie Recommendation System
cd /d "%~dp0"

echo.
echo  ============================================
echo   Hybrid Movie Recommendation System
echo  ============================================
echo.

where py >nul 2>&1
if %errorlevel% neq 0 (
    echo  Python not found. Installing Python 3.12...
    winget install Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
    echo.
    echo  Python installed. Close this window and double-click run.bat again.
    pause
    exit /b
)

if not exist "venv\Scripts\streamlit.exe" (
    echo  First-time setup. This takes 2-3 minutes...
    echo.

    if exist "venv\" (
        rmdir /s /q venv
    )

    py -3.12 -m venv venv
    if %errorlevel% neq 0 (
        echo.
        echo  ERROR: Could not create virtual environment.
        echo  Make sure Python 3.12 is installed by running: py -0
        echo.
        pause
        exit /b
    )

    echo  Installing packages...
    venv\Scripts\pip.exe install streamlit pandas "numpy<2" scikit-learn scikit-surprise
    if %errorlevel% neq 0 (
        echo.
        echo  ERROR: Package installation failed. Check your internet connection.
        echo.
        pause
        exit /b
    )

    echo.
    echo  Setup complete.
)

echo  Opening app at http://localhost:8501
echo  Press Ctrl+C in this window to stop the app.
echo.
venv\Scripts\streamlit.exe run app.py

pause
