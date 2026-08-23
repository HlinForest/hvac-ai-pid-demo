@echo off
setlocal
cd /d "%~dp0"
python main.py --quick
if errorlevel 1 (
  echo.
  echo Demo failed. Install dependencies with: python -m pip install -r requirements.txt
  exit /b 1
)
echo.
echo Finished. Open outputs\engineering_report.html for the visual report.
