@echo off
setlocal
where cl >nul 2>&1
if errorlevel 1 (
    echo Run this script from an x64 Visual Studio Developer Command Prompt. 1>&2
    exit /b 1
)
pushd "%~dp0" || exit /b 1
set "vd_build=..\build\windows-capture"
if not exist "%vd_build%" mkdir "%vd_build%"
if not exist "%vd_build%" (
    popd
    exit /b 1
)
cl /nologo /std:c++17 /O2 /EHsc /W4 /MT windows_capture.cpp /Fo:"%vd_build%\windows_capture.obj" /Fe:"%vd_build%\vd-windows-capture.exe" /link /MACHINE:X64 d3d11.lib dxgi.lib user32.lib
set "vd_result=%errorlevel%"
popd
exit /b %vd_result%
