@echo off
echo ============================================================
echo  Download SBERT models to sbert_models\
echo ============================================================
echo.
echo Usage:
echo   download_sbert.bat          -- all models  (~2 GB)
echo   download_sbert.bat tiny     -- tiny only   (~45 MB)
echo   download_sbert.bat large    -- large only  (~1.3 GB)
echo.
if "%1"=="" (
    python download_sbert_models.py all
) else (
    python download_sbert_models.py %1
)
echo.
pause
