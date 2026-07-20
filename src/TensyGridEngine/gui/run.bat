@echo off
setlocal enabledelayedexpansion

:: Directori del script
set "SCRIPT_DIR=%~dp0"

:: PROJECT_ROOT: 3 nivells cap enrere des de gui/
pushd "%SCRIPT_DIR%..\..\.."
set "PROJECT_ROOT=%CD%"
popd

:: Busca el venv en diverses ubicacions possibles
set "VENV_PYTHON="

:: 1) PROJECT_ROOT\venv  (ubicació estàndard)
if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    set "VENV_PYTHON=%PROJECT_ROOT%\venv\Scripts\python.exe"
    goto :found
)

:: 2) Un nivell per sobre de PROJECT_ROOT
pushd "%PROJECT_ROOT%\.."
set "PARENT=%CD%"
popd
if exist "%PARENT%\venv\Scripts\python.exe" (
    set "VENV_PYTHON=%PARENT%\venv\Scripts\python.exe"
    goto :found
)

:: 3) Dins de PROJECT_ROOT (el propi PROJECT_ROOT és el venv)
if exist "%PROJECT_ROOT%\Scripts\python.exe" (
    set "VENV_PYTHON=%PROJECT_ROOT%\Scripts\python.exe"
    goto :found
)

:: 4) Qualsevol subdirectori de PROJECT_ROOT que es digui *venv*
for /d %%D in ("%PROJECT_ROOT%\*venv*") do (
    if exist "%%D\Scripts\python.exe" (
        set "VENV_PYTHON=%%D\Scripts\python.exe"
        goto :found
    )
)

echo Error: no s'ha trobat cap venv. Ubicacions cercades:
echo   %PROJECT_ROOT%\venv\Scripts\python.exe
echo   %PARENT%\venv\Scripts\python.exe
echo   %PROJECT_ROOT%\Scripts\python.exe
echo   %PROJECT_ROOT%\*venv*\Scripts\python.exe
echo.
echo Crea un entorn virtual amb: python -m venv venv
exit /b 1

:found
echo Usant Python: %VENV_PYTHON%

:: Comprova si fastapi esta instal·lat
"%VENV_PYTHON%" -c "import fastapi" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Instal·lant dependències GUI...
    "%VENV_PYTHON%" -m pip install fastapi uvicorn
)

echo.
echo Iniciant VeraGrid GUI a http://localhost:8550
"%VENV_PYTHON%" "%SCRIPT_DIR%dashboard.py"

pause
