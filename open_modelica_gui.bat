@echo off
setlocal EnableExtensions
set "PROJECT_DIR=%~dp0"
if defined OPENMODELICAHOME (
  set "OMEDIT_EXE=%OPENMODELICAHOME%\bin\OMEdit.exe"
) else if defined MODELICA_OMEDIT (
  set "OMEDIT_EXE=%MODELICA_OMEDIT%"
) else (
  set "OMEDIT_EXE=OMEdit.exe"
)

if not exist "%OMEDIT_EXE%" (
  echo OMEdit was not found at "%OMEDIT_EXE%".
  echo Edit OMEDIT_EXE in this file if OpenModelica is installed elsewhere.
  pause
  exit /b 1
)

if exist N:\ (
  echo Drive N: is already in use. Close this window and free N: first.
  pause
  exit /b 1
)

subst N: "%PROJECT_DIR:~0,-1%"
set "OMEDIT_APPDATA=N:\modelica_runtime\omedit_appdata"
if exist "%OMEDIT_APPDATA%\.openmodelica\libraries" mkdir "%OMEDIT_APPDATA%\.openmodelica\libraries"
if defined OPENMODELICAHOME if exist "%OPENMODELICAHOME%\share\omlibrary\cache\index.json" (
  copy /Y "%OPENMODELICAHOME%\share\omlibrary\cache\index.json" "%OMEDIT_APPDATA%\.openmodelica\libraries\index.json" >nul
)
set "APPDATA=%OMEDIT_APPDATA%"
set "OPENMODELICALIBRARY=N:/modelica_runtime/omlibrary;N:/modelica_runtime/omedit_appdata/.openmodelica/libraries"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
echo Opening HVACAI.PrecisionCabinetCooling in OMEdit...
echo OMEdit user-library data is isolated in an ASCII-only project directory.
echo Keep this command window open until OMEdit is closed.
start "" /wait "%OMEDIT_EXE%" "N:/modelica/HVACAI/package.mo"
set "OMEDIT_EXIT=%ERRORLEVEL%"
subst N: /d
exit /b %OMEDIT_EXIT%
