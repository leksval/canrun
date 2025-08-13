#!/usr/bin/env python3
"""
DEBUG script to trace exactly which ML model file the production executable loads
This will help confirm if PyInstaller path resolution is the root cause
"""

import subprocess
import json
import sys
from pathlib import Path

def test_executable_ml_model_paths():
    """Test which ML model paths the production executable actually uses"""
    
    # Enhanced test query that triggers ML model loading
    test_query = {
        "game_name": "Cyberpunk 2077",
        "gpu_name": "RTX 4070", 
        "resolution": "1080p",
        "action": "fps_prediction",
        "debug_ml_paths": True  # Request debug information
    }
    
    print("DEBUGGING: Production Executable ML Model Path Resolution")
    print("=" * 70)
    print(f"Test query: {test_query}")
    
    # Path to executable relative to test directory
    exe_path = Path(__file__).parent.parent / "dist" / "g-assist-plugin-canrun.exe"
    
    if not exe_path.exists():
        print(f"ERROR: Executable not found at: {exe_path}")
        return False
    
    print(f"Using executable: {exe_path}")
    
    try:
        # Run the production executable with debug query
        result = subprocess.run(
            [str(exe_path)],
            input=json.dumps(test_query),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"\nReturn Code: {result.returncode}")
        print(f"Stdout: {result.stdout[:500]}...")
        print(f"Stderr: {result.stderr[:500]}...")
        
        if result.returncode == 0 and result.stdout:
            try:
                response = json.loads(result.stdout)
                
                # Extract FPS prediction
                fps = None
                if 'fps_prediction' in response:
                    fps = float(response['fps_prediction'])
                    print(f"\nFPS Prediction: {fps}")
                    
                    # Check if it matches expected fixed value (139) or old value (340)
                    if 130 <= fps <= 150:
                        print(f"SUCCESS: FPS {fps} is in EXPECTED RANGE (130-150) - ML fixes working!")
                        return True
                    elif 300 <= fps <= 400:
                        print(f"FAILURE: FPS {fps} is in OLD RANGE (300-400) - ML fixes NOT working!")
                        print("CONFIRMED: Executable is using old ML model data")
                        return False
                    else:
                        print(f"WARNING: FPS {fps} is in unexpected range")
                        return False
                else:
                    print("ERROR: No fps_prediction found in response")
                    return False
                    
            except json.JSONDecodeError as e:
                print(f"ERROR: Could not parse JSON response: {e}")
                print(f"Raw output: {result.stdout}")
                return False
        else:
            print(f"ERROR: Executable failed or produced no output")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("ERROR: Executable timed out")
        return False
    except Exception as e:
        print(f"ERROR: Error running executable: {e}")
        return False

def test_development_vs_production_comparison():
    """Compare development environment vs production executable predictions"""
    
    print("\n" + "=" * 70)
    print("DEVELOPMENT vs PRODUCTION COMPARISON")
    print("=" * 70)
    
    # Test development environment first
    print("1. Testing DEVELOPMENT environment...")
    try:
        # Import and test ML model loader directly
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from canrun.canrun.src.ml_model_loader import LightweightMLPredictor
        
        predictor = LightweightMLPredictor()
        
        # Test Cyberpunk 2077 prediction
        fps_dev = predictor.predict_fps(
            gpu_name="RTX 4070",
            passmark_score=28000,  # Approximate RTX 4070 score
            resolution="1080p",
            game_name="Cyberpunk 2077",
            memory_gb=12
        )
        
        print(f"Development FPS: {fps_dev}")
        print(f"Model loaded: {predictor.is_loaded}")
        print(f"Lookup table size: {len(predictor.lookup_table)}")
        
        # Print model info
        model_info = predictor.get_model_info()
        print(f"Model info: {model_info}")
        
    except Exception as e:
        print(f"ERROR: Development test failed: {e}")
        fps_dev = None
    
    # Test production executable
    print("\n2. Testing PRODUCTION executable...")
    fps_prod = None
    
    try:
        test_query = {
            "game_name": "Cyberpunk 2077",
            "gpu_name": "RTX 4070", 
            "resolution": "1080p",
            "action": "fps_prediction"
        }
        
        exe_path = Path(__file__).parent.parent / "dist" / "g-assist-plugin-canrun.exe"
        result = subprocess.run(
            [str(exe_path)],
            input=json.dumps(test_query),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            if 'fps_prediction' in response:
                fps_prod = float(response['fps_prediction'])
                print(f"Production FPS: {fps_prod}")
    except Exception as e:
        print(f"ERROR: Production test failed: {e}")
    
    # Compare results
    print(f"\nCOMPARISON RESULTS:")
    print(f"Development FPS: {fps_dev}")
    print(f"Production FPS:  {fps_prod}")
    
    if fps_dev and fps_prod:
        diff = abs(fps_dev - fps_prod)
        print(f"Difference: {diff:.1f} FPS")
        
        if diff > 50:
            print(f"CRITICAL: SIGNIFICANT DIFFERENCE! Confirms ML model mismatch.")
            print(f"CRITICAL: Production executable is NOT using updated ML fixes.")
            return False
        else:
            print(f"SUCCESS: Similar results. ML fixes working correctly.")
            return True
    else:
        print(f"ERROR: Could not compare - one or both tests failed")
        return False

if __name__ == "__main__":
    print("CRITICAL DEPLOYMENT FAILURE DIAGNOSIS")
    print("Tracing ML model path resolution in production executable")
    print("=" * 70)
    
    # Test 1: Basic executable ML model loading
    success1 = test_executable_ml_model_paths()
    
    # Test 2: Development vs Production comparison
    success2 = test_development_vs_production_comparison()
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    if not success1 or not success2:
        print("CONFIRMED: Production executable is NOT using ML model fixes")
        print("ROOT CAUSE: PyInstaller path resolution issue in ml_model_loader.py")
        print("SOLUTION: Implement PyInstaller-aware path resolution")
        sys.exit(1)
    else:
        print("SUCCESS: Production executable is working correctly")
        sys.exit(0)