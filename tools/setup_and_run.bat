@echo off
echo Checking environment...
d:\projects\cable_torch_env\Scripts\python.exe verify_environment.py
if %errorlevel% neq 0 (
    echo Environment check failed.
    echo If the error is about "Microsoft Visual C++ Redistributable", please install it from:
    echo https://aka.ms/vs/16/release/vc_redist.x64.exe
    pause
    exit /b 1
)

echo Starting training...
d:\projects\cable_torch_env\Scripts\python.exe train_cable_failure_model.py
pause
