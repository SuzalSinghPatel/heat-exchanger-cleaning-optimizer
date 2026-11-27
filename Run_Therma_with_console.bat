@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (py -3 app\desktop_app.py) else (python app\desktop_app.py)
pause
