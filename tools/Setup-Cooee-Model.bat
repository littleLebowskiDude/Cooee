@echo off
setlocal enabledelayedexpansion

rem  Setup-Cooee-Model.bat
rem
rem  Cooee ships without a speech model. This downloads one and points Cooee
rem  at it, so you don't have to clone the repository or go hunting through
rem  folders in the settings window.
rem
rem  Why isn't the model just inside the installer? Because Cooee has no
rem  internet code in it at all - that's the point of the app, and there's a
rem  test in the repo that fails the build if anyone adds any. Downloading
rem  lives out here in a script you can read instead.
rem
rem  Everything below is plain, readable commands. Nothing is hidden or
rem  encoded. Read it before you run it.

title Cooee - model setup

echo.
echo   ===========================================================
echo     Cooee - speech model setup
echo   ===========================================================
echo.

rem ---- Check this is an ARM64 machine ------------------------------------
rem  Cooee only runs on Snapdragon / Windows-on-ARM laptops. On anything else
rem  it cannot work at all, so say so now rather than after a 925 MB download.
rem
rem  Read the architecture carefully. A 32-bit command prompt on an ARM64
rem  machine reports plain "ARM" and puts the real answer in ARCHITEW6432.
rem  Wrongly telling someone their laptop is unsupported is the worst thing
rem  this file could do, so match on ARM rather than on ARM64 exactly.

set "ARCH=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "ARCH=%PROCESSOR_ARCHITEW6432%"

echo %ARCH% | find /i "ARM" >nul
if errorlevel 1 (
    echo   This is not an ARM64 machine.
    echo.
    echo   Cooee only runs on a Snapdragon / Copilot+ laptop. It uses the
    echo   NPU chip that those machines have and other laptops don't.
    echo.
    echo   Nothing has been downloaded. You can close this window.
    echo.
    pause
    exit /b 1
)

rem ---- Which model ------------------------------------------------------
rem  small.en is the one to use. base.en is a smaller, less accurate download
rem  if you just want a quick look: run this file with  base  after it.

set "SIZE=small"
if /i "%~1"=="base" set "SIZE=base"

if "%SIZE%"=="small" (
    set "TOTALMB=925"
    set "MIN_ENCODER=300000000"
    set "MIN_DECODER=500000000"
) else (
    set "TOTALMB=280"
    set "MIN_ENCODER=60000000"
    set "MIN_DECODER=170000000"
)

set "MODELROOT=%LOCALAPPDATA%\Cooee\models"
set "MODELDIR=%MODELROOT%\whisper-%SIZE%.en-onnx"
set "BASEURL=https://huggingface.co/onnx-community/whisper-%SIZE%.en/resolve/main"

echo   Model      whisper-%SIZE%.en
echo   From       huggingface.co
echo   Into       %MODELDIR%
echo   Size       about %TOTALMB% MB
echo.
echo   This is a big download. On a normal connection expect a few minutes.
echo   You can leave it running and do something else.
echo.
pause
echo.

rem ---- Check curl exists -------------------------------------------------
rem  curl.exe is part of Windows 11, so this should never fire.

where curl.exe >nul 2>&1
if errorlevel 1 (
    echo   Could not find curl.exe, which is normally part of Windows.
    echo   Please send this message to Angus.
    echo.
    pause
    exit /b 1
)

if not exist "%MODELDIR%\onnx" mkdir "%MODELDIR%\onnx" 2>nul
if not exist "%MODELDIR%\onnx" (
    echo   Could not create the folder %MODELDIR%
    echo   Check you have space on your C: drive.
    echo.
    pause
    exit /b 1
)

rem ---- Download ----------------------------------------------------------
rem  Small text files first, then the two big ones. Anything already
rem  downloaded is skipped, so if this stops halfway you can just run it
rem  again and it picks up where it left off.

call :get "config.json"                    config.json                  1000
if errorlevel 1 goto :failed
call :get "generation_config.json"         generation_config.json       1000
if errorlevel 1 goto :failed
call :get "vocab.json"                     vocab.json                   100000
if errorlevel 1 goto :failed
call :get "merges.txt"                     merges.txt                   100000
if errorlevel 1 goto :failed
call :get "added_tokens.json"              added_tokens.json            1000
if errorlevel 1 goto :failed

echo.
echo   The next two files are the big ones.
echo.
call :get "onnx/encoder_model.onnx"        onnx\encoder_model.onnx        %MIN_ENCODER%
if errorlevel 1 goto :failed
call :get "onnx/decoder_model_merged.onnx" onnx\decoder_model_merged.onnx %MIN_DECODER%
if errorlevel 1 goto :failed

rem ---- Point Cooee at the model -----------------------------------------
rem  Cooee keeps its settings in a plain JSON file. This sets the model
rem  folder in that file and leaves every other setting exactly as it was,
rem  so it's safe to run even if you've already customised things.
rem
rem  The folder path is handed over as an environment variable so there are
rem  no fiddly nested quotes to get wrong.

echo.
echo   Pointing Cooee at the model...

set "COOEE_MODEL_DIR=%MODELDIR%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $dir = $env:COOEE_MODEL_DIR; $folder = Join-Path $env:APPDATA 'cooee'; $cfg = Join-Path $folder 'config.json'; New-Item -ItemType Directory -Force -Path $folder | Out-Null; if (Test-Path $cfg) { $o = Get-Content $cfg -Raw | ConvertFrom-Json } else { $o = New-Object psobject }; $o | Add-Member -MemberType NoteProperty -Name model_path -Value $dir -Force; $o | ConvertTo-Json -Depth 20 | Set-Content -Path $cfg -Encoding UTF8"

if errorlevel 1 (
    echo.
    echo   The model downloaded fine, but Cooee's settings file could not be
    echo   updated automatically. You can set it by hand instead:
    echo.
    echo     Cooee tray icon, then the cog, then Model, then Folder...
    echo     and choose this folder:
    echo.
    echo     %MODELDIR%
    echo.
    pause
    exit /b 1
)

rem ---- Is Cooee running right now? --------------------------------------
rem  Settings are read when Cooee starts, so a running copy won't notice
rem  what we just wrote until it's restarted.

set "RUNNING="
for /f "delims=" %%P in ('tasklist /fi "imagename eq Cooee.exe" /nh 2^>nul ^| find /i "Cooee.exe"') do set "RUNNING=1"

echo.
echo   ===========================================================
echo     Done. The model is installed and Cooee knows where it is.
echo   ===========================================================
echo.

if defined RUNNING (
    echo   Cooee is running at the moment. Quit it and start it again so it
    echo   picks up the model:  right-click the tray icon, Quit, then open
    echo   Cooee from the Start menu.
) else (
    echo   Open Cooee from the Start menu.
)

echo.
echo   The very first time it loads the model it takes about 20 seconds
echo   while it gets it ready for the NPU chip. That happens once.
echo.
echo   Then hold  Ctrl + Windows  , say something, and let go.
echo   Your words land in whatever you were typing in.
echo.
pause
exit /b 0


:failed
echo.
echo   ===========================================================
echo     The download didn't finish.
echo   ===========================================================
echo.
echo   Nothing is broken. The most likely cause is the connection
echo   dropping partway through a large file.
echo.
echo   Just run this file again - everything already downloaded is
echo   kept, so it carries on rather than starting over.
echo.
pause
exit /b 1


rem ---- :get  <url path>  <local path>  <minimum expected bytes> ----------
rem  Downloads one file, unless it's already here and a sensible size.
rem  The size check matters: a download cut off halfway leaves a file that
rem  looks present but makes Cooee fail later with a confusing error.

:get
set "REL=%~1"
set "LOCAL=%MODELDIR%\%~2"
set "MINBYTES=%~3"

if exist "%LOCAL%" (
    for %%A in ("%LOCAL%") do set "HAVE=%%~zA"
    if !HAVE! GEQ %MINBYTES% (
        echo     already have   %~2
        exit /b 0
    )
    echo     incomplete, fetching again   %~2
    del /f /q "%LOCAL%" 2>nul
)

echo     downloading    %~2
curl.exe -L --fail --progress-bar -o "%LOCAL%" "%BASEURL%/%REL%"
if errorlevel 1 (
    if exist "%LOCAL%" del /f /q "%LOCAL%" 2>nul
    exit /b 1
)

for %%A in ("%LOCAL%") do set "GOT=%%~zA"
if !GOT! LSS %MINBYTES% (
    echo     that file arrived incomplete.
    del /f /q "%LOCAL%" 2>nul
    exit /b 1
)
exit /b 0
