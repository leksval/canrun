@echo off
echo ====================================
echo CanRun G-Assist Plugin Setup
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

echo Python detected. Installing dependencies...
echo.

REM Install requirements
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo ====================================
echo Setup completed successfully!
echo ====================================
echo.
echo Next steps:
echo 1. Copy the plugin directory to your G-Assist plugins folder
echo 2. Run 'test.bat' to verify the plugin works
echo 3. Restart G-Assist to load the plugin
echo.
pause