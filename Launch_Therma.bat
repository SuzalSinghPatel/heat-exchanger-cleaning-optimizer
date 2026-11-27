@echo off
setlocal
cd /d "%~dp0"
title Therma - Predictive Cleaning Optimiser

echo ========================================================
echo   THERMA - Predictive Cleaning Optimiser
echo ========================================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
    echo During installation, tick "Add Python to PATH".
    pause
    exit /b 1
  )
  set "PY=python"
)

echo Checking required components. This may take a few minutes on first launch...
%PY% -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo.
  echo Dependency installation failed. Check your internet connection and Python installation.
  pause
  exit /b 1
)

echo.
echo Starting Therma...
where pyw >nul 2>nul
if %errorlevel%==0 (
  start "Therma" pyw -3 app\desktop_app.py
) else (
  where pythonw >nul 2>nul
  if %errorlevel%==0 (
    start "Therma" pythonw app\desktop_app.py
  ) else (
    echo Python GUI launcher was not found. Try Run_Therma_with_console.bat.
    pause
  )
)
endlocal
