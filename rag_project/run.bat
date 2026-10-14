@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Project environment not found. Running setup first...
    call setup.bat /no-pause
    if errorlevel 1 goto :error
)

".venv\Scripts\python.exe" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Streamlit is missing from the project environment. Repairing setup...
    call setup.bat /no-pause
    if errorlevel 1 goto :error
)

if not exist ".env" (
    copy ".env.example" ".env" >nul
    if errorlevel 1 goto :error
)

echo Starting StudyPilot AI. Keep this window open while using the app.
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless false
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo StudyPilot did not start. Run setup.bat to install or repair dependencies.
pause
exit /b 1
