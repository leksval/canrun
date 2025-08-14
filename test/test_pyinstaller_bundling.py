#!/usr/bin/env python3
"""
Test PyInstaller data file bundling for ML models
Validates that all required data files can be found in both development and compiled modes
"""

import sys
import os
from pathlib import Path
import json

def test_pyinstaller_data_file_resolution():
    """Test that all required data files can be resolved in PyInstaller bundle"""
    print("Testing PyInstaller data file resolution...")
    
    # Simulate PyInstaller environment
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        bundle_dir = Path(sys._MEIPASS)
        print(f"Running in PyInstaller mode, bundle dir: {bundle_dir}")
        is_frozen = True
    else:
        # Running in development
        bundle_dir = Path(__file__).parent.parent
        print(f"Running in development mode, base dir: {bundle_dir}")
        is_frozen = False
    
    # Define all required data files
    required_files = [
        # Core ML model files
        "data/ml_fps_model_unified_2025.json",
        "canrun/src/ml_fps_model.json",
        
        # Hardware database files
        "data/gpu_hierarchy.json", 
        "data/cpu_benchmarks.json",
        
        # Advanced ML model files
        "data/unified_calibrator_2025.pkl",
        "data/unified_scaler_2025.pkl", 
        "data/unified_stacking_model_2025.pkl",
        
        # Additional data files
        "data/game_requirements.json",
        "data/training_benchmarks.json"
    ]
    
    print(f"\nChecking {len(required_files)} required data files:")
    
    found_files = 0
    missing_files = []
    
    for file_path in required_files:
        if is_frozen:
            # PyInstaller bundle structure
            full_path = bundle_dir / file_path
        else:
            # Development structure
            if file_path.startswith("canrun/src/"):
                full_path = bundle_dir / file_path
            else:
                full_path = bundle_dir / file_path
        
        if full_path.exists():
            print(f"  [OK] {file_path}")
            found_files += 1
            
            # Test JSON files can be loaded
            if file_path.endswith('.json'):
                try:
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                        if 'lookup_table' in data:
                            print(f"    ML model with {len(data['lookup_table'])} predictions")
                        elif 'nvidia' in data:
                            print(f"    GPU hierarchy with {len(data['nvidia'])} GPUs")
                        else:
                            print(f"    JSON data loaded successfully")
                except Exception as e:
                    print(f"    [WARN] JSON load error: {e}")
        else:
            print(f"  [FAIL] {file_path} - NOT FOUND")
            missing_files.append(file_path)
    
    print(f"\nResults: {found_files}/{len(required_files)} files found")
    
    if missing_files:
        print(f"\nMissing files:")
        for missing in missing_files:
            print(f"  - {missing}")
        
        if is_frozen:
            print(f"\nPyInstaller bundle contents:")
            for item in sorted(bundle_dir.rglob("*")):
                if item.is_file():
                    rel_path = item.relative_to(bundle_dir)
                    print(f"  {rel_path}")
    
    return found_files == len(required_files)

def test_ml_model_loading():
    """Test ML model can be loaded and provides predictions"""
    print(f"\nTesting ML model loading...")
    
    try:
        # Import the ML predictor
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from canrun.src.ml_model_loader import LightweightMLPredictor
        
        # Initialize predictor
        predictor = LightweightMLPredictor()
        
        if predictor.is_loaded:
            print("  [OK] ML model loaded successfully")
            
            # Test prediction
            fps = predictor.predict_fps(
                gpu_name="RTX 4090",
                passmark_score=35000,
                resolution="1440p",
                game_name="Counter-Strike 2"
            )
            
            if fps and fps > 200:
                print(f"  [OK] Prediction test: CS2 @ 1440p = {fps} FPS")
                return True
            else:
                print(f"  [FAIL] Invalid prediction: {fps} FPS")
                return False
        else:
            print("  [FAIL] ML model failed to load")
            return False
            
    except Exception as e:
        print(f"  [FAIL] ML model loading error: {e}")
        return False

if __name__ == "__main__":
    print("PyInstaller Data File Bundling Test")
    print("=" * 50)
    
    # Test file resolution
    files_ok = test_pyinstaller_data_file_resolution()
    
    # Test ML model loading 
    ml_ok = test_ml_model_loading()
    
    print(f"\n" + "=" * 50)
    if files_ok and ml_ok:
        print("[OK] All tests passed - PyInstaller bundling should work correctly")
        sys.exit(0)
    else:
        print("[FAIL] Some tests failed - PyInstaller bundling needs fixes")
        sys.exit(1)