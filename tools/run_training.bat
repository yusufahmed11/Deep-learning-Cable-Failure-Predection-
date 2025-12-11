@echo off
echo Starting training using cable_torch_env...
d:\projects\cable_torch_env\Scripts\python.exe train_cable_failure_model.py
if %errorlevel% neq 0 (
    echo.
    echo CRITICAL ERROR: Training failed.
    echo If you see "Microsoft Visual C++ Redistributable is not installed", please download and install it from:
    echo https://aka.ms/vs/16/release/vc_redist.x64.exe
    echo.
    pause
) else (
    echo Training completed successfully.
    pause
)
