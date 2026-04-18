@echo off
setlocal
:: Go to the directory of this .bat file so python can find bootstrap_run.py
cd /d "%~dp0"

echo.
echo ============================================================
echo  BankReasonTrainer - Start (Windows)
echo ============================================================
echo.

:: --- Find Python: prefer py launcher (never points to MS Store stub) -------
set PYCMD=

where py >nul 2>&1
if not errorlevel 1 (
    py -3 --version >nul 2>&1
    if not errorlevel 1 (
        set PYCMD=py -3
        goto :found_python
    )
)

:: Check that "python" is not the Microsoft Store stub (WindowsApps path)
where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys,pathlib; exit(1 if 'WindowsApps' in str(pathlib.Path(sys.executable)) else 0)" >nul 2>&1
    if not errorlevel 1 (
        set PYCMD=python
        goto :found_python
    ) else (
        echo [WARNING] "python" points to Microsoft Store stub - skipping.
        echo.
    )
)

echo [ERROR] Python 3 not found in PATH.
echo.
echo   Install Python 3.9+ from the official site:
echo     https://www.python.org/downloads/
echo.
echo   During installation check both:
echo     "Add Python to PATH"  and  "py launcher"
echo.
pause
exit /b 1

:found_python
:: --- Show Python version ---------------------------------------------------
for /f "tokens=*" %%v in ('%PYCMD% --version 2^>^&1') do set PY_VER=%%v
echo Found: %PY_VER%
echo.

:: --- Run bootstrap (checks tkinter + deps, then launches GUI) --------------
%PYCMD% bootstrap_run.py
set EXIT_CODE=%errorlevel%

if %EXIT_CODE% neq 0 (
    echo.
    echo [ERROR] Application exited with code %EXIT_CODE%.
    echo         See messages above or run_app.log in the program folder.
    echo.
    pause
    exit /b %EXIT_CODE%
)

endlocal
