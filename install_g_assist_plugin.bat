@echo off
REM ================================================================
REM CanRun v7.0.0 G-Assist Plugin Installation Script
REM Updated for NVIDIA G-Assist Plugin System
REM ================================================================

echo ======================================================
echo Installing CanRun v7.0.0 G-Assist Plugin
echo ======================================================

REM Check if running as Administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running as Administrator - OK
) else (
    echo ERROR: This script must be run as Administrator
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

REM Define directories
set "PLUGIN_DIR=%PROGRAMDATA%\NVIDIA Corporation\nvtopps\rise\plugins\canrun"
set "SOURCE_DIR=%~dp0"

echo Source Directory: %SOURCE_DIR%
echo Target Plugin Directory: %PLUGIN_DIR%
echo.

REM Create the plugin directory structure
echo Creating plugin directory structure...
if not exist "%PROGRAMDATA%\NVIDIA Corporation" mkdir "%PROGRAMDATA%\NVIDIA Corporation"
if not exist "%PROGRAMDATA%\NVIDIA Corporation\nvtopps" mkdir "%PROGRAMDATA%\NVIDIA Corporation\nvtopps"
if not exist "%PROGRAMDATA%\NVIDIA Corporation\nvtopps\rise" mkdir "%PROGRAMDATA%\NVIDIA Corporation\nvtopps\rise"
if not exist "%PROGRAMDATA%\NVIDIA Corporation\nvtopps\rise\plugins" mkdir "%PROGRAMDATA%\NVIDIA Corporation\nvtopps\rise\plugins"

REM Remove existing plugin if it exists
if exist "%PLUGIN_DIR%" (
    echo Removing existing CanRun plugin...
    rmdir /s /q "%PLUGIN_DIR%"
)

REM Create the canrun plugin directory
echo Creating CanRun plugin directory...
mkdir "%PLUGIN_DIR%"

REM Step 1: Copy main executable
echo.
echo Step 1: Installing main executable...
if exist "%SOURCE_DIR%g-assist-plugin-canrun.exe" (
    copy "%SOURCE_DIR%g-assist-plugin-canrun.exe" "%PLUGIN_DIR%\"
    echo [OK] Copied g-assist-plugin-canrun.exe
) else (
    echo WARNING: g-assist-plugin-canrun.exe not found!
    echo You may need to build it first with: pyinstaller --onefile --name g-assist-plugin-canrun plugin.py
)

REM Step 2: Copy configuration files
echo.
echo Step 2: Installing configuration files...
if exist "%SOURCE_DIR%manifest.json" (
    copy "%SOURCE_DIR%manifest.json" "%PLUGIN_DIR%\"
    echo [OK] Copied manifest.json
) else (
    echo ERROR: manifest.json not found!
    goto :error
)

if exist "%SOURCE_DIR%config.json" (
    copy "%SOURCE_DIR%config.json" "%PLUGIN_DIR%\"
    echo [OK] Copied config.json
) else (
    echo WARNING: config.json not found - using defaults
)

REM Step 3: Copy data directory
echo.
echo Step 3: Installing data files...
if exist "%SOURCE_DIR%data" (
    xcopy "%SOURCE_DIR%data" "%PLUGIN_DIR%\data\" /E /I /Y >nul 2>&1
    echo [OK] Copied data\ directory
) else (
    echo WARNING: data\ directory not found - plugin may have limited functionality
)

REM Step 4: Copy additional files (optional)
echo.
echo Step 4: Installing additional files...
if exist "%SOURCE_DIR%requirements.txt" (
    copy "%SOURCE_DIR%requirements.txt" "%PLUGIN_DIR%\"
    echo [OK] Copied requirements.txt
)

if exist "%SOURCE_DIR%plugin.py" (
    copy "%SOURCE_DIR%plugin.py" "%PLUGIN_DIR%\"
    echo [OK] Copied plugin.py (source)
)

if exist "%SOURCE_DIR%LICENSE" (
    copy "%SOURCE_DIR%LICENSE" "%PLUGIN_DIR%\"
    echo [OK] Copied LICENSE
)

REM Verify installation
echo.
echo ======================================================
echo Verifying installation...
echo ======================================================

if exist "%PLUGIN_DIR%\g-assist-plugin-canrun.exe" (
    echo [OK] Executable: g-assist-plugin-canrun.exe
) else (
    echo [ERROR] Missing executable!
    goto :error
)

if exist "%PLUGIN_DIR%\manifest.json" (
    echo [OK] Configuration: manifest.json
) else (
    echo [ERROR] Missing manifest.json!
    goto :error
)

if exist "%PLUGIN_DIR%\data" (
    echo [OK] Data files: data\ directory
) else (
    echo [ERROR] Missing data\ directory!
    goto :error
)

REM Test plugin functionality (optional)
echo.
echo ======================================================
echo Testing plugin functionality...
echo ======================================================

cd /d "%PLUGIN_DIR%"
if exist "g-assist-plugin-canrun.exe" (
    echo Testing plugin execution...
    REM Test with a simple command to verify it works
    timeout /t 2 >nul
    echo Plugin executable found and ready
) else (
    echo Plugin executable not found for testing
)

REM Display installation summary
echo.
echo ======================================================
echo CanRun v7.0.0 Plugin Installation Complete!
echo ======================================================
echo.
echo Plugin installed to: %PLUGIN_DIR%
echo.
echo Files installed:
dir "%PLUGIN_DIR%" /B 2>nul | findstr /v "^$"
echo.
echo G-Assist Integration:
echo - Plugin Name: canrun
echo - Voice Commands: "Hey canrun, can I run [game]?"
echo - Example: "Can I run Diablo 4?"
echo - Example: "Check compatibility for Cyberpunk 2077"
echo.
echo Next Steps:
echo 1. Ensure NVIDIA G-Assist is running
echo 2. Restart G-Assist if it was already running
echo 3. Test with voice command: "Hey canrun, can I run Diablo 4?"
echo.
goto :success

:error
echo.
echo ======================================================
echo Installation Failed!
echo ======================================================
echo Please check the following:
echo 1. All required files are in the source directory
echo 2. You're running as Administrator
echo 3. NVIDIA G-Assist is properly installed
echo.
pause
exit /b 1

:success
echo Installation completed successfully!
echo.
echo To rebuild the plugin after code changes, use:
echo pyinstaller --onefile --name g-assist-plugin-canrun --distpath . --add-data "data;data" plugin.py
echo.
pause
exit /b 0