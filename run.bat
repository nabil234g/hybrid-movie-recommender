@echo off
title Hybrid Movie Recommendation System
cd /d "%~dp0"

echo.
echo  ============================================
echo   Hybrid Movie Recommendation System
echo  ============================================
echo.

py -3.12 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  Python 3.12 not found. Installing now...
    winget install Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
    echo.
    echo  Done. Close this window and double-click run.bat again.
    pause
    exit /b
)

if not exist "venv\Scripts\streamlit.exe" (
    echo  First-time setup in progress...
    echo.

    if exist "venv\" (
        rmdir /s /q venv
    )

    py -3.12 -m venv venv
    if %errorlevel% neq 0 (
        echo  ERROR: Could not create virtual environment.
        pause
        exit /b
    )

    echo  Installing packages (no compilation needed)...
    venv\Scripts\pip.exe install wheels\scikit_surprise-1.1.4-cp312-cp312-win_amd64.whl
    venv\Scripts\pip.exe install streamlit pandas "numpy<2" scikit-learn

    if not exist "venv\Scripts\streamlit.exe" (
        echo.
        echo  ERROR: Installation failed. Check your internet connection and try again.
        pause
        exit /b
    )

    echo.
    echo  Setup complete.
    echo.
)

echo  Opening app at http://localhost:8501
echo  Press Ctrl+C in this window to stop the app.
echo.
venv\Scripts\streamlit.exe run app.py

pause
