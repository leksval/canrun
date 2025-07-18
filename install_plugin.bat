@echo off
echo ========================================
echo CanRun G-Assist Plugin Installer v5.0
echo ========================================
echo.

REM Change to the directory where this batch file is located
cd /d "%~dp0"

set PLUGIN_NAME=canrun
set GASSIST_PLUGINS_DIR=%PROGRAMDATA%\NVIDIA Corporation\nvtopps\rise\plugins
set PLUGIN_DIR=%GASSIST_PLUGINS_DIR%\%PLUGIN_NAME%

echo Checking G-Assist installation...
if not exist "%GASSIST_PLUGINS_DIR%" (
    echo ERROR: G-Assist plugins directory not found!
    echo Expected location: %GASSIST_PLUGINS_DIR%
    echo.
    echo Please ensure that:
    echo 1. NVIDIA GeForce Experience is installed
    echo 2. G-Assist feature is enabled
    echo 3. You have administrator privileges
    echo.
    pause
    exit /b 1
)

echo Success: G-Assist plugins directory found: %GASSIST_PLUGINS_DIR%
echo.

echo Checking required files...
echo Current directory: %CD%
echo Listing files:
dir *.json *.exe
echo.
if not exist manifest.json (
    echo ERROR: manifest.json not found in current directory
    echo Current working directory: %CD%
    echo Batch file location: %~dp0
    pause
    exit /b 1
)
if not exist config.json (
    echo ERROR: config.json not found in current directory
    pause
    exit /b 1
)
if not exist canrun-g-assist-plugin.exe (
    echo ERROR: canrun-g-assist-plugin.exe not found in current directory
    echo Make sure the executable name matches the manifest.json declaration
    pause
    exit /b 1
)

echo Success: All required files found
echo.

echo Creating plugin directory...
mkdir "%PLUGIN_DIR%" 2>nul
if not exist "%PLUGIN_DIR%" (
    echo ERROR: Failed to create plugin directory
    echo Location: %PLUGIN_DIR%
    echo Please run this installer as Administrator
    pause
    exit /b 1
)

echo Success: Plugin directory created: %PLUGIN_DIR%
echo.

echo Installing plugin files...

echo Copying manifest.json...
copy manifest.json "%PLUGIN_DIR%\" >nul
if errorlevel 1 (
    echo ERROR: Failed to copy manifest.json
    pause
    exit /b 1
)

echo Copying config.json...
copy config.json "%PLUGIN_DIR%\" >nul
if errorlevel 1 (
    echo ERROR: Failed to copy config.json
    pause
    exit /b 1
)

echo Copying canrun-g-assist-plugin.exe...
copy canrun-g-assist-plugin.exe "%PLUGIN_DIR%\" >nul
if errorlevel 1 (
    echo ERROR: Failed to copy executable
    echo Make sure you have write permissions to %PLUGIN_DIR%
    pause
    exit /b 1
)

echo Copying data files...
if exist data (
    xcopy data "%PLUGIN_DIR%\data" /E /I /Q >nul
    if errorlevel 1 (
        echo WARNING: Failed to copy data directory
    ) else (
        echo Success: Data directory copied
    )
) else (
    echo WARNING: Data directory not found, skipping...
)

if exist src (
    xcopy src "%PLUGIN_DIR%\src" /E /I /Q >nul
    if errorlevel 1 (
        echo WARNING: Failed to copy src directory
    ) else (
        echo Success: Source directory copied
    )
) else (
    echo WARNING: Source directory not found, skipping...
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Plugin installed to: %PLUGIN_DIR%
echo.
echo Plugin directory structure:
echo %PLUGIN_DIR%\
echo - canrun-g-assist-plugin.exe
echo - manifest.json
echo - config.json
echo - data\ (if present)
echo - src\ (if present)
echo.
echo Next steps:
echo 1. Restart NVIDIA GeForce Experience
echo 2. Press Alt+G to activate G-Assist
echo 3. Try: "Hey canrun, can I run Diablo 4?"
echo.
echo Testing plugin installation...
cd /d "%PLUGIN_DIR%" 2>nul
if errorlevel 1 (
    echo WARNING: Could not change to plugin directory
    echo Testing from current location...
    "%PLUGIN_DIR%\canrun-g-assist-plugin.exe" --function detect_hardware
) else (
    canrun-g-assist-plugin.exe --function detect_hardware
)
echo.
echo If you see JSON output above, the plugin is working!
echo.
echo Plugin discovery logging will be written to:
echo %USERPROFILE%\canrun-enhanced-plugin.log
echo.
pause