@echo off
REM ================================================================
REM CanRun v10.0 G-Assist Plugin Installation Script
REM Protocol V2 (gassist_sdk 3.1.0) - executable is plugin.py
REM ================================================================

echo ======================================================
echo Installing CanRun v10.0 G-Assist Plugin
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

REM Step 1: Copy plugin sources and configuration
echo.
echo Step 1: Installing plugin sources...
for %%F in (
    plugin.py
    canrun_engine.py
    canrun_hardware_detector.py
    canrun_game_fetcher.py
    canrun_game_matcher.py
    canrun_compatibility_analyzer.py
    canrun_ml_predictor.py
    canrun_model_loader.py
    canrun_steam_fps.py
    g_assist_response_fixer.py
    manifest.json
    config.json
    requirements.txt
) do (
    if not exist "%SOURCE_DIR%%%F" (
        echo ERROR: Missing %%F
        goto :error
    )
    copy "%SOURCE_DIR%%%F" "%PLUGIN_DIR%\" >nul
    echo [OK] Copied %%F
)

REM Step 2: Copy data and vendored SDK
echo.
echo Step 2: Installing data and gassist_sdk...
if exist "%SOURCE_DIR%data" (
    xcopy "%SOURCE_DIR%data" "%PLUGIN_DIR%\data\" /E /I /Y >nul
    echo [OK] Copied data\
) else (
    echo ERROR: data\ directory not found
    goto :error
)

if exist "%SOURCE_DIR%libs\gassist_sdk" (
    xcopy "%SOURCE_DIR%libs" "%PLUGIN_DIR%\libs\" /E /I /Y >nul
    echo [OK] Copied libs\gassist_sdk
) else (
    echo ERROR: libs\gassist_sdk not found
    goto :error
)

if exist "%SOURCE_DIR%LICENSE" (
    copy "%SOURCE_DIR%LICENSE" "%PLUGIN_DIR%\" >nul
    echo [OK] Copied LICENSE
)

REM Step 3: Install Python dependencies into libs\ so G-Assist can import them
echo.
echo Step 3: Installing Python dependencies into libs\...
py -3 -m pip install -r "%PLUGIN_DIR%\requirements.txt" -t "%PLUGIN_DIR%\libs"
if errorlevel 1 (
    python -m pip install -r "%PLUGIN_DIR%\requirements.txt" -t "%PLUGIN_DIR%\libs"
    if errorlevel 1 (
        echo ERROR: pip install failed
        goto :error
    )
)
echo [OK] Dependencies installed

REM Verify installation
echo.
echo ======================================================
echo Verifying installation...
echo ======================================================

if exist "%PLUGIN_DIR%\plugin.py" (
    echo [OK] Executable: plugin.py
) else (
    echo [ERROR] Missing plugin.py
    goto :error
)

if exist "%PLUGIN_DIR%\manifest.json" (
    echo [OK] Configuration: manifest.json
) else (
    echo [ERROR] Missing manifest.json
    goto :error
)

if exist "%PLUGIN_DIR%\config.json" (
    echo [OK] Configuration: config.json
) else (
    echo [ERROR] Missing config.json
    goto :error
)

if exist "%PLUGIN_DIR%\libs\gassist_sdk\__init__.py" (
    echo [OK] SDK: libs\gassist_sdk
) else (
    echo [ERROR] Missing libs\gassist_sdk
    goto :error
)

if exist "%PLUGIN_DIR%\data" (
    echo [OK] Data files: data\
) else (
    echo [ERROR] Missing data\
    goto :error
)

REM Display installation summary
echo.
echo ======================================================
echo CanRun v10.0 Plugin Installation Complete!
echo ======================================================
echo.
echo Plugin installed to: %PLUGIN_DIR%
echo.
echo Files installed:
dir "%PLUGIN_DIR%" /B 2>nul | findstr /v "^$"
echo.
echo G-Assist Integration:
echo - Plugin Name: canrun
echo - Protocol: V2 (JSON-RPC 2.0, gassist_sdk 3.1.0)
echo - Voice commands: "Can I run [game]?"
echo - Example: "Can I run Diablo 4?"
echo - Example: "Check compatibility for Cyberpunk 2077"
echo.
echo Next Steps:
echo 1. Ensure NVIDIA G-Assist is running
echo 2. Restart G-Assist if it was already running
echo 3. Test with voice command: "Can I run Diablo 4?"
echo.
echo CLI check from the plugin directory:
echo python plugin.py canrun "cyberpunk 2077" --json
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
echo 4. Python 3.12+ is on PATH so pip can fill libs\
echo.
pause
exit /b 1

:success
echo Installation completed successfully!
echo.
pause
exit /b 0
