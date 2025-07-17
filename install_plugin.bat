@echo off
echo ========================================
echo CanRun G-Assist Plugin Installer v5.0
echo ========================================
echo.

echo Creating plugin directory...
mkdir "%USERPROFILE%\canrun" 2>nul

echo Checking required files...
if not exist "manifest.json" (
    echo ERROR: manifest.json not found in current directory
    pause
    exit /b 1
)
if not exist "config.json" (
    echo ERROR: config.json not found in current directory
    pause
    exit /b 1
)
if not exist "canrun-g-assist-plugin.exe" (
    echo ERROR: canrun-g-assist-plugin.exe not found in current directory
    pause
    exit /b 1
)

echo Copying essential files...
copy manifest.json "%USERPROFILE%\canrun\" >nul
if errorlevel 1 (
    echo ERROR: Failed to copy manifest.json
    pause
    exit /b 1
)

copy config.json "%USERPROFILE%\canrun\" >nul
if errorlevel 1 (
    echo ERROR: Failed to copy config.json
    pause
    exit /b 1
)

copy canrun-g-assist-plugin.exe "%USERPROFILE%\canrun\" >nul
if errorlevel 1 (
    echo ERROR: Failed to copy executable
    echo Make sure you have write permissions to %USERPROFILE%\canrun\
    pause
    exit /b 1
)

echo Copying data files...
xcopy data "%USERPROFILE%\canrun\data" /E /I /Q >nul
if errorlevel 1 (
    echo WARNING: Failed to copy data directory
)

xcopy src "%USERPROFILE%\canrun\src" /E /I /Q >nul
if errorlevel 1 (
    echo WARNING: Failed to copy src directory
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Plugin installed to: %USERPROFILE%\canrun\
echo.
echo Next steps:
echo 1. Restart NVIDIA GeForce Experience
echo 2. Press Alt+G to activate G-Assist
echo 3. Try: "Hey canrun, can I run Diablo 4?"
echo.
echo Testing plugin...
cd /d "%USERPROFILE%\canrun" 2>nul
if errorlevel 1 (
    echo WARNING: Could not change to plugin directory
    echo Testing from current location...
    "%USERPROFILE%\canrun\canrun-g-assist-plugin.exe" --function check_compatibility --game "Test Game"
) else (
    canrun-g-assist-plugin.exe --function check_compatibility --game "Test Game"
)
echo.
echo If you see JSON output above, the plugin is working!
echo.
pause