#!/usr/bin/env python3
"""
Test script to verify the production executable returns realistic FPS predictions
for Cyberpunk 2077 (should be ~60-140 FPS range, not 340+ FPS)
"""

import subprocess
import json
import sys
from pathlib import Path

def test_cyberpunk_fps():
    """Test Cyberpunk 2077 FPS prediction using the production executable"""
    
    # Test query for Cyberpunk 2077 with RTX 4070 (common high-end GPU)
    test_query = {
        "game_name": "Cyberpunk 2077",
        "gpu_name": "RTX 4070", 
        "resolution": "1080p",
        "action": "fps_prediction"
    }
    
    print("Testing production executable with Cyberpunk 2077...")
    print(f"Test query: {test_query}")
    
    # Path to executable relative to test directory
    exe_path = Path(__file__).parent.parent / "dist" / "g-assist-plugin-canrun.exe"
    
    try:
        # Run the production executable
        result = subprocess.run(
            [str(exe_path)],
            input=json.dumps(test_query),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("\n Executable ran successfully!")
            print(f"Output: {result.stdout}")
            
            # Try to parse the response to extract FPS
            try:
                response = json.loads(result.stdout)
                if 'fps_prediction' in response:
                    fps = float(response['fps_prediction'])
                    print(f"\n FPS Prediction: {fps}")
                    
                    # Verify realistic range (60-140 FPS, not 340+ like before the fix)
                    if 50 <= fps <= 150:
                        print(f" PASS: FPS prediction {fps} is in realistic range (50-150)")
                        return True
                    else:
                        print(f" FAIL: FPS prediction {fps} is outside realistic range (50-150)")
                        print("This suggests the bug fix may not be working correctly")
                        return False
                else:
                    print("  No fps_prediction found in response")
                    return False
                    
            except json.JSONDecodeError:
                print("  Could not parse JSON response")
                print(f"Raw output: {result.stdout}")
                return False
                
        else:
            print(f" Executable failed with return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(" Executable timed out")
        return False
    except Exception as e:
        print(f" Error running executable: {e}")
        return False

if __name__ == "__main__":
    success = test_cyberpunk_fps()
    if success:
        print("\n Production deployment test PASSED!")
        sys.exit(0)
    else:
        print("\n Production deployment test FAILED!")
        sys.exit(1)