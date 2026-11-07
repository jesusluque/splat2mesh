@echo off
rem splat2mesh: everything, unattended, on this PC (Windows + NVIDIA GPU).
rem   install_all.bat [vae: klein|dev] [iris: cpu-encoder|gpu|space]
rem Optional environment: HF_TOKEN (gated DINOv3 download only, not stored), DEEPSEEK_API_KEY (saved for the app).
setlocal
set ROOT=%~dp0..
set VAE=%1
if "%VAE%"=="" set VAE=klein
set IRIS=%2
if "%IRIS%"=="" (
  for /f "tokens=1" %%m in ('nvidia-smi --query-gpu^=memory.total --format^=csv^,noheader^,nounits 2^>nul') do (
    if %%m GEQ 15000 (set IRIS=gpu) else (set IRIS=cpu-encoder)
  )
)
if "%IRIS%"=="" set IRIS=space
echo [1/3] core
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\installer\bootstrap.ps1" -Root "%ROOT%" || goto :fail
if defined DEEPSEEK_API_KEY "%ROOT%\.venv\Scripts\python.exe" -m splat2mesh.winsetup options --deepseek-key-env DEEPSEEK_API_KEY
echo [2/3] components (athenea, Idol, ENEAS, Iris-3B: %IRIS%)
"%ROOT%\.venv\Scripts\python.exe" -u -m splat2mesh.winsetup install all --vae %VAE% --iris-mode %IRIS% || goto :fail
echo [3/3] status
"%ROOT%\.venv\Scripts\python.exe" -m splat2mesh.winsetup status
echo Done. Start the app with splat2mesh.bat
exit /b 0
:fail
echo Installation failed: see the messages above. Run this file again to resume: downloads and installed packages are reused.
exit /b 1
