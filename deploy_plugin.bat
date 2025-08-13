@echo off
echo CanRun G-Assist Plugin Deployment Script
echo ==========================================
echo This script requires Administrator privileges to deploy the plugin.
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: This script must be run as Administrator!
    echo.
    echo SOLUTION: Please follow these steps:
    echo 1. Close this window
    echo 2. Right-click 'deploy_plugin.bat' in File Explorer
    echo 3. Select "Run as administrator"
    echo 4. Click "Yes" when prompted by Windows UAC
    echo.
    echo ALTERNATIVE: Manual deployment instructions:
    echo 1. Open File Explorer as Administrator
    echo 2. Navigate to: C:\Program Files\NVIDIA Corporation\NVIDIA app\plugins\canrun\
    echo 3. Copy these files from this directory:
    echo    - manifest.json
    echo    - plugin.py
    echo    - g-assist-plugin-canrun.exe
    echo.
    pause
    exit /b 1
)

echo Creating plugin directory...
mkdir "C:\Program Files\NVIDIA Corporation\NVIDIA app\plugins\canrun" 2>nul

echo Copying plugin files...
echo Source files located in: %cd%
echo Target directory: C:\Program Files\NVIDIA Corporation\NVIDIA app\plugins\canrun\
echo.

copy "manifest.json" "C:\Program Files\NVIDIA Corporation\NVIDIA app\plugins\canrun\"
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy manifest.json - Administrator privileges required!
    goto :deployment_failed
)

copy "plugin.py" "C:\Program Files\NVIDIA Corporation\NVIDIA app\plugins\canrun\"
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy plugin.py - Administrator privileges required!
    goto :deployment_failed
)

copy "g-assist-plugin-canrun.exe" "C:\Program Files\NVIDIA Corporation\NVIDIA app\plugins\canrun\"
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy g-assist-plugin-canrun.exe - Administrator privileges required!
    goto :deployment_failed
)

echo.
echo Force killing NVIDIA processes to ensure clean restart...
taskkill /f /im "NVIDIA app.exe" 2>nul
taskkill /f /im "nvcontainer.exe" 2>nul
timeout /t 3 /nobreak >nul

echo Restarting NVIDIA App...
start "" "C:\Program Files\NVIDIA Corporation\NVIDIA app\nvidia_app.exe"

echo.
echo Verifying deployment...
if exist "C:\Program Files\NVIDIA Corporation\NVIDIA app\plugins\canrun\manifest.json" (
    echo ✓ manifest.json deployed successfully
) else (
    echo ✗ Failed to deploy manifest.json
)

if exist "C:\Program Files\NVIDIA Corporation\NVIDIA app\plugins\canrun\plugin.py" (
    echo ✓ plugin.py deployed successfully
) else (
    echo ✗ Failed to deploy plugin.py
)

if exist "C:\Program Files\NVIDIA Corporation\NVIDIA app\plugins\canrun\g-assist-plugin-canrun.exe" (
    echo ✓ g-assist-plugin-canrun.exe deployed successfully
) else (
    echo ✗ Failed to deploy g-assist-plugin-canrun.exe
)

echo.
echo ==========================================
echo CanRun Plugin Deployment Status:
echo ==========================================
echo.
echo Plugin Features:
echo • Smart Adaptive Stacking ML Model v6.1
echo • 94.8%% R² Accuracy, 23.2 FPS MAE
echo • RTX 50 Series Support
echo • 4K Gaming Performance Predictions
echo • S-Tier Performance Scoring
echo.
echo To use the plugin:
echo 1. Restart NVIDIA App
echo 2. Open G-Assist
echo 3. Ask "Can I run [game name]?"
echo.
echo The plugin will analyze your system and provide:
echo • Compatibility assessment
echo • Performance predictions
echo • Optimization recommendations
echo • S-Tier scoring system
echo.
echo ==========================================
pause