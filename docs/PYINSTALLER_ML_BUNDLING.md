# PyInstaller ML Model Bundling Guide

## Overview

This document provides comprehensive instructions for building CanRun G-Assist plugin executables with proper ML model data file bundling using PyInstaller.

## Problem Statement

The CanRun G-Assist plugin relies on multiple ML model data files for accurate FPS prediction. When building executables with PyInstaller, these data files must be explicitly bundled or the compiled executable will fall back to simplified PassMark calculations, resulting in constant FPS predictions instead of variable, game-specific predictions.

## Required Data Files

The following 9 data files are essential for proper ML model functionality:

### Core ML Models (JSON)
- `canrun/src/ml_fps_model.json` - Primary ML model data
- `canrun/data/ml_fps_model.json` - Secondary ML model data

### Hardware Databases
- `canrun/data/gpu_hierarchy.json` - GPU hierarchy and performance data
- `canrun/data/cpu_benchmarks.json` - CPU benchmark database

### Advanced ML Files (PKL)
- `canrun/data/fps_predictor_model.pkl` - Advanced ML predictor model
- `canrun/data/stacking_ensemble_model.pkl` - Stacking ensemble model
- `canrun/data/gpu_performance_model.pkl` - GPU performance model

### Supplementary Data Files
- `canrun/data/training_benchmarks.json` - Training benchmark data
- `canrun/data/game_specific_optimizations.json` - Game-specific optimization data

## PyInstaller Configuration

### Spec File Location
The PyInstaller specification file is located at:
```
canrun/g-assist-plugin-canrun.spec
```

### Data Files Configuration
The `datas` section in the spec file must include all required files:

```python
datas = [
    # Core ML model files
    ('canrun/src/ml_fps_model.json', 'canrun/src'),
    ('canrun/data/ml_fps_model.json', 'canrun/data'),
    
    # Hardware databases
    ('canrun/data/gpu_hierarchy.json', 'canrun/data'),
    ('canrun/data/cpu_benchmarks.json', 'canrun/data'),
    
    # Advanced ML files
    ('canrun/data/fps_predictor_model.pkl', 'canrun/data'),
    ('canrun/data/stacking_ensemble_model.pkl', 'canrun/data'),
    ('canrun/data/gpu_performance_model.pkl', 'canrun/data'),
    
    # Supplementary data files
    ('canrun/data/training_benchmarks.json', 'canrun/data'),
    ('canrun/data/game_specific_optimizations.json', 'canrun/data'),
],
```

## Build Process

### Prerequisites
- Python environment with all CanRun dependencies installed
- PyInstaller package installed
- All 9 required data files present in their respective locations

### Build Commands
```bash
# Navigate to project directory
cd canrun

# Build executable using the spec file
pyinstaller g-assist-plugin-canrun.spec

# Alternative: Build using uv (recommended)
uv run pyinstaller g-assist-plugin-canrun.spec
```

### Output Location
The compiled executable will be created at:
```
canrun/dist/canrun/canrun.exe
```

## Validation

### Testing ML Model Functionality
After building, validate that the ML model is working correctly:

```bash
# Test with various games to ensure variable FPS predictions
cd canrun
uv run python test/test_detailed_output_format.py

# Manual testing with specific games
uv run python plugin.py "Counter-Strike 2"
uv run python plugin.py "Cyberpunk 2077"
uv run python plugin.py "Diablo I"
```

### Expected Results
- **Variable FPS Predictions**: Different games should show different FPS values
- **No Constant Values**: Avoid consistent values like "77 always 77" or "93 FPS for all games"
- **Game-Specific Ranges**: Expect ranges like Counter-Strike 2: 125 FPS, Cyberpunk 2077: 66 FPS

### Troubleshooting

#### Missing Data Files Error
**Symptom**: Constant FPS predictions (e.g., 93 FPS for all games)
**Cause**: PyInstaller spec file missing data files in `datas` section
**Solution**: Verify all 9 files are listed in the `datas` configuration

#### Path Resolution Issues
**Symptom**: FileNotFoundError for data files during execution
**Cause**: Incorrect path mapping in PyInstaller spec
**Solution**: Ensure source paths are relative to spec file location

#### ML Model Loading Failures
**Symptom**: Fallback to PassMark calculations (PassMark/150 formula)
**Cause**: Missing or corrupted ML model files
**Solution**: 
1. Verify all .json and .pkl files are present and valid
2. Check that `ml_model_loader.py` can resolve PyInstaller bundle paths using `sys._MEIPASS`

## Maintenance

### Adding New Data Files
When adding new ML model data files:

1. **Update Spec File**: Add the new file to the `datas` section
2. **Update Documentation**: Add the file to the "Required Data Files" section above
3. **Update Tests**: Include validation for the new file in test suites
4. **Rebuild Executable**: Run PyInstaller with updated spec file

### Version Updates
When updating ML models:

1. **Replace Data Files**: Update the source files in their respective directories
2. **Test Locally**: Validate changes work in development mode
3. **Rebuild Executable**: Use PyInstaller to create new executable with updated data
4. **Validate Changes**: Test the compiled executable for expected behavior

## Best Practices

1. **Always Test Both Modes**: Validate functionality in both development and compiled modes
2. **Use Relative Paths**: Ensure spec file paths are relative for portability
3. **Validate File Presence**: Check that all required files exist before building
4. **Document Changes**: Update this guide when modifying data file requirements
5. **Version Control**: Track spec file changes alongside ML model updates

## Path Resolution Implementation

The CanRun ML model loader (`canrun/src/ml_model_loader.py`) handles PyInstaller path resolution:

```python
import sys
import os

def get_data_file_path(relative_path):
    """Get absolute path to data file, handling PyInstaller bundle."""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        base_path = sys._MEIPASS
    else:
        # Running in development
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, relative_path)
```

This ensures data files are correctly located in both development and compiled environments.

## Success Criteria

A successful PyInstaller build will demonstrate:

1. **Variable FPS Predictions**: Multiple games show different FPS values
2. **ML Model Functionality**: Predictions based on game-specific analysis, not generic PassMark calculations
3. **Complete Feature Set**: All CanRun features work identically to development mode
4. **No File Errors**: No missing file errors during execution
5. **Consistent Performance**: Reliable ML model predictions across different games and scenarios

---

**Last Updated**: January 2025
**Maintainer**: CanRun Development Team
**Related Files**: 
- `g-assist-plugin-canrun.spec`
- `canrun/src/ml_model_loader.py`
- `canrun/test/test_pyinstaller_bundling.py`