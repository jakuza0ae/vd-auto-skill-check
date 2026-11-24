@echo off
setlocal
set "VD_PYTHON=%~dp0python\python.exe"
if not exist "%VD_PYTHON%" (
    echo Bundled Python was not found. Extract the complete Windows release ZIP.
    exit /b 1
)
if "%~1"=="" (
    "%VD_PYTHON%" -B "%~dp0run.py" preflight
) else (
    "%VD_PYTHON%" -B "%~dp0run.py" %*
)
exit /b %errorlevel%
