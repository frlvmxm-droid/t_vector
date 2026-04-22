@echo off
REM ===================================================================
REM  BankReasonTrainer -- launch Voila web-UI on localhost (Windows).
REM
REM  Works on Windows 10/11. Web-UI runs in the browser over
REM  ipywidgets + Voila; no Tkinter / CTk required.
REM
REM  Environment overrides (optional):
REM    BRT_PORT     -- port to bind Voila (default: 8866)
REM    BRT_HOST     -- host to bind Voila (default: 127.0.0.1)
REM    BRT_NO_OPEN  -- set to 1 to skip auto-opening the browser
REM ===================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

if "%BRT_PORT%"=="" (set "PORT=8866") else (set "PORT=%BRT_PORT%")
if "%BRT_HOST%"=="" (set "HOST=127.0.0.1") else (set "HOST=%BRT_HOST%")

REM --- pick Python interpreter -------------------------------------
set "PY="

REM Prefer the official Python launcher with an explicit 3.11+ hint.
where py >nul 2>&1
if %ERRORLEVEL%==0 (
    py -3.11 -c "import sys" >nul 2>&1
    if !ERRORLEVEL!==0 (
        set "PY=py -3.11"
    ) else (
        py -3 -c "import sys; sys.exit(0 if sys.version_info>=(3,11) else 1)" >nul 2>&1
        if !ERRORLEVEL!==0 set "PY=py -3"
    )
)

REM Fall back to python.exe on PATH -- but reject the Microsoft Store stub.
if "!PY!"=="" (
    where python >nul 2>&1
    if !ERRORLEVEL!==0 (
        python -c "import sys, os; sys.exit(1 if 'WindowsApps' in os.path.dirname(sys.executable) else 0)" >nul 2>&1
        if !ERRORLEVEL!==0 (
            python -c "import sys; sys.exit(0 if sys.version_info>=(3,11) else 1)" >nul 2>&1
            if !ERRORLEVEL!==0 set "PY=python"
        )
    )
)

if "!PY!"=="" (
    echo ERROR: Python 3.11+ not found.
    echo Install from https://python.org/downloads/  (NOT the Microsoft Store stub^).
    pause
    exit /b 1
)

for /f "tokens=*" %%V in ('!PY! --version 2^>^&1') do set "PYVER=%%V"
echo [run_web] using: !PYVER!  (!PY!^)

REM --- ensure ui deps ----------------------------------------------
!PY! -c "import voila, ipywidgets" >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo [run_web] installing web-UI dependencies ^(this can take a minute on first run^)...
    !PY! -m pip install --quiet --disable-pip-version-check -e ".[ui]"
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo ERROR: pip install failed. Try manually:
        echo          !PY! -m pip install "ipywidgets>=8.0" "voila>=0.5"
        pause
        exit /b 1
    )
)

REM --- check port is free -----------------------------------------
!PY! -c "import socket,sys; s=socket.socket(); sys.exit(0) if s.connect_ex(('%HOST%',%PORT%)) else sys.exit(1)" >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: port %PORT% on %HOST% is already in use.
    echo        Pick another one:  set BRT_PORT=8867 ^&^& run_web.bat
    pause
    exit /b 1
)

REM --- open browser (best-effort) ---------------------------------
if NOT "%BRT_NO_OPEN%"=="1" (
    start "" "http://%HOST%:%PORT%/"
)

echo [run_web] serving on http://%HOST%:%PORT%/  (Ctrl+C to stop^)
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
!PY! -m voila notebooks\ui.ipynb --port=%PORT% --Voila.ip=%HOST% --no-browser

endlocal
