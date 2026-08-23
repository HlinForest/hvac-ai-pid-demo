@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0modelica\run_openmodelica_validation.ps1"
if errorlevel 1 (
  echo.
  echo OpenModelica validation failed. See the messages above.
  pause
  exit /b 1
)
echo.
echo OpenModelica validation and report refresh completed.
pause
