@echo off
setlocal

echo ============================================================
echo  BankReasonTrainer - Windows EXE Build
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Install Python 3.10+ from https://python.org
    echo         Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

python --version
echo.

:: Install app dependencies
echo [1/4] Installing app dependencies...
pip install --upgrade -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: Install PyInstaller
echo [2/4] Installing PyInstaller...
pip install --upgrade pyinstaller --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install PyInstaller.
    pause
    exit /b 1
)

:: Check UPX (optional, reduces exe size ~30%)
upx --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] UPX not found - building without compression ^(larger output^).
) else (
    echo [INFO] UPX found - compression enabled.
)
echo.

:: Clean previous build
echo [3/4] Cleaning previous build...
if exist "dist\BankReasonTrainer" rmdir /s /q "dist\BankReasonTrainer"
if exist "build"                  rmdir /s /q "build"

:: Run PyInstaller
echo [4/4] Building EXE...
echo.
pyinstaller bank_reason_trainer.spec --clean --noconfirm
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed. Common causes:
    echo   - Missing module: try  pip install ^<name^>
    echo   - Version conflict: try  pip install --upgrade pyinstaller
    pause
    exit /b 1
)

:: Create zip archive
echo.
echo Creating archive...
powershell -Command "Compress-Archive -Path 'dist\BankReasonTrainer' -DestinationPath 'dist\BankReasonTrainer-Windows.zip' -Force" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Archive skipped - use dist\BankReasonTrainer\ folder directly.
) else (
    echo [OK]  Archive: dist\BankReasonTrainer-Windows.zip
)

echo.
echo ============================================================
echo  Done!
echo  EXE: dist\BankReasonTrainer\BankReasonTrainer.exe
echo  ZIP: dist\BankReasonTrainer-Windows.zip
echo ============================================================
echo.
echo To run: double-click BankReasonTrainer.exe in dist\BankReasonTrainer\
echo.
pause
endlocal
