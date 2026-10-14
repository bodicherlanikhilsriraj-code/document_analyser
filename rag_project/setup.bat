@echo off
setlocal
cd /d "%~dp0"

set "PYTHON="
py -3.11 --version >nul 2>&1
if not errorlevel 1 set "PYTHON=py -3.11"

if not defined PYTHON (
    set "PYTHON=py -3"
    call :check_python
    if errorlevel 1 (
        set "PYTHON=python"
        call :check_python
        if errorlevel 1 goto :python_error
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the project virtual environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 goto :error
)

if not exist ".env.example" (
    echo The example configuration file .env.example is missing.
    goto :error
)

echo Installing project dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

if not exist ".env" (
    copy ".env.example" ".env" >nul
    if errorlevel 1 goto :error
)

echo.
echo Setup complete. Start StudyPilot by double-clicking run.bat.
if /i not "%~1"=="/no-pause" pause
exit /b 0

:error
echo.
echo Setup failed. Check the message above, then run setup.bat again.
if /i not "%~1"=="/no-pause" pause
exit /b 1

:python_error
echo Python 3.10 or newer was not found.
echo Install Python 3.11 from https://www.python.org/downloads/ and select "Add Python to PATH".
if /i not "%~1"=="/no-pause" pause
exit /b 1

:check_python
%PYTHON% -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>&1
exit /b %errorlevel%
