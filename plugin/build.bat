@echo off
echo ====================================
echo CanRun G-Assist Plugin Build
echo ====================================
echo.

REM Set build directory
set BUILD_DIR=build
set PLUGIN_NAME=canrun-g-assist-plugin
set VERSION=1.0.0

REM Clean previous build
if exist "%BUILD_DIR%" (
    echo Cleaning previous build...
    rmdir /s /q "%BUILD_DIR%"
)

REM Create build directory
echo Creating build directory...
mkdir "%BUILD_DIR%"
mkdir "%BUILD_DIR%\%PLUGIN_NAME%"

REM Copy plugin files
echo Copying plugin files...
copy "g-assist-plugin-canrun.py" "%BUILD_DIR%\%PLUGIN_NAME%\"
copy "manifest.json" "%BUILD_DIR%\%PLUGIN_NAME%\"
copy "requirements.txt" "%BUILD_DIR%\%PLUGIN_NAME%\"
copy "setup.bat" "%BUILD_DIR%\%PLUGIN_NAME%\"
copy "test.bat" "%BUILD_DIR%\%PLUGIN_NAME%\"

REM Copy source files
echo Copying source files...
mkdir "%BUILD_DIR%\%PLUGIN_NAME%\src"
copy "..\src\*.py" "%BUILD_DIR%\%PLUGIN_NAME%\src\"

REM Copy data files
echo Copying data files...
mkdir "%BUILD_DIR%\%PLUGIN_NAME%\data"
copy "..\data\*.json" "%BUILD_DIR%\%PLUGIN_NAME%\data\"

REM Create README for distribution
echo Creating distribution README...
echo # CanRun G-Assist Plugin v%VERSION% > "%BUILD_DIR%\%PLUGIN_NAME%\README.txt"
echo. >> "%BUILD_DIR%\%PLUGIN_NAME%\README.txt"
echo Installation: >> "%BUILD_DIR%\%PLUGIN_NAME%\README.txt"
echo 1. Run setup.bat to install dependencies >> "%BUILD_DIR%\%PLUGIN_NAME%\README.txt"
echo 2. Run test.bat to verify functionality >> "%BUILD_DIR%\%PLUGIN_NAME%\README.txt"
echo 3. Copy this folder to your G-Assist plugins directory >> "%BUILD_DIR%\%PLUGIN_NAME%\README.txt"
echo 4. Restart G-Assist to load the plugin >> "%BUILD_DIR%\%PLUGIN_NAME%\README.txt"

REM Create ZIP package
echo Creating ZIP package...
cd "%BUILD_DIR%"
powershell -Command "Compress-Archive -Path '%PLUGIN_NAME%' -DestinationPath '%PLUGIN_NAME%-v%VERSION%.zip'"
cd ..

echo.
echo ====================================
echo Build completed successfully!
echo ====================================
echo.
echo Package created: %BUILD_DIR%\%PLUGIN_NAME%-v%VERSION%.zip
echo.
pause