@echo off
echo ====================================
echo CanRun G-Assist Plugin Test
echo ====================================
echo.

REM Test basic functionality
echo Testing basic plugin functionality...
echo.

REM Test hardware detection
echo [TEST 1] Hardware Detection:
python g-assist-plugin-canrun.py --function detect_hardware
if %errorlevel% neq 0 (
    echo ERROR: Hardware detection failed
    pause
    exit /b 1
)
echo.

REM Test game compatibility check
echo [TEST 2] Game Compatibility Check:
python g-assist-plugin-canrun.py --function check_compatibility --game "Cyberpunk 2077"
if %errorlevel% neq 0 (
    echo ERROR: Game compatibility check failed
    pause
    exit /b 1
)
echo.

REM Test performance prediction
echo [TEST 3] Performance Prediction:
python g-assist-plugin-canrun.py --function predict_performance --game "Cyberpunk 2077"
if %errorlevel% neq 0 (
    echo ERROR: Performance prediction failed
    pause
    exit /b 1
)
echo.

REM Test optimization suggestions
echo [TEST 4] Optimization Suggestions:
python g-assist-plugin-canrun.py --function get_optimization_suggestions --game "Cyberpunk 2077"
if %errorlevel% neq 0 (
    echo ERROR: Optimization suggestions failed
    pause
    exit /b 1
)
echo.

echo ====================================
echo All tests completed successfully!
echo ====================================
echo.
echo The plugin is ready for use with G-Assist.
echo.
pause