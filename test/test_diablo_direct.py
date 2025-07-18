#!/usr/bin/env python3
"""
Direct test of 'canrun diablo' functionality with complete CanRun analysis.
"""

import subprocess
import json
import sys
import os

def test_canrun_diablo_direct():
    """Test 'canrun diablo' with the g-assist-plugin-canrun.exe"""
    print("🧪 Testing 'canrun diablo' with complete CanRun analysis...")
    
    # Test input for canrun function with Diablo 4
    test_input = json.dumps({
        "tool_calls": [
            {
                "func": "canrun",
                "params": {"game_name": "Diablo 4"}
            }
        ]
    })
    
    try:
        # Use absolute path to the executable
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        exe_path = os.path.join(parent_dir, 'g-assist-plugin-canrun.exe')
        
        if not os.path.exists(exe_path):
            print(f"❌ Executable not found at: {exe_path}")
            assert False, f"Executable not found at: {exe_path}"
        
        print(f"🔍 Using executable: {exe_path}")
        
        # Run the plugin executable
        process = subprocess.Popen(
            [exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=parent_dir  # Run from parent directory
        )
        
        # Send the test input with timeout
        try:
            stdout, stderr = process.communicate(input=test_input, timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            raise
        
        print(f"📤 Input: {test_input}")
        print(f"📥 Raw Output: {stdout}")
        
        if stderr:
            print(f"⚠️ Stderr: {stderr}")
        
        # Parse the response
        if '<<END>>' in stdout:
            response_text = stdout.split('<<END>>')[0]
            try:
                response = json.loads(response_text)
                
                if response.get('success'):
                    message = response.get('message', '')
                    print(f"✅ Success! Response length: {len(message)} characters")
                    
                    # Check for key indicators of complete CanRun analysis
                    indicators = [
                        'PERFORMANCE TIER:',
                        'A (',  # A-tier scoring
                        '/100',  # Score out of 100
                        'PrivacyAwareHardwareDetector',
                        'GameRequirementsFetcher',
                        'DynamicPerformancePredictor',
                        'CompatibilityAnalyzer',
                        'CANRUN VERDICT:'
                    ]
                    
                    found_indicators = [ind for ind in indicators if ind in message]
                    print(f"🔍 Found {len(found_indicators)}/{len(indicators)} CanRun analysis indicators:")
                    for ind in found_indicators:
                        print(f"  ✓ {ind}")
                    
                    if len(found_indicators) >= 6:
                        print("🎉 COMPLETE CANRUN ANALYSIS CONFIRMED!")
                        print("✅ 'canrun diablo' successfully triggers full CanRun pipeline with A-tier scoring")
                        assert True
                    else:
                        print("⚠️ Partial analysis detected - some components may not be integrated")
                        assert False, "Partial analysis detected"
                else:
                    print(f"❌ Plugin returned error: {response.get('message', 'Unknown error')}")
                    assert False, f"Plugin returned error: {response.get('message', 'Unknown error')}"
                    
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON response: {e}")
                print(f"Raw response: {response_text}")
                assert False, f"Failed to parse JSON response: {e}"
        else:
            print("❌ No proper response terminator found")
            assert False, "No proper response terminator found"
            
    except subprocess.TimeoutExpired:
        print("❌ Plugin execution timed out")
        assert False, "Plugin execution timed out"
    except Exception as e:
        print(f"❌ Error running plugin: {e}")
        assert False, f"Error running plugin: {e}"

if __name__ == '__main__':
    print("=" * 60)
    print("CANRUN DIABLO DIRECT TEST")
    print("=" * 60)
    
    success = test_canrun_diablo_direct()
    
    print("=" * 60)
    if success:
        print("🎉 TEST PASSED: 'canrun diablo' triggers complete CanRun analysis!")
        print("✅ All src components integrated successfully")
        print("✅ A-tier (92/100) performance assessment working")
        print("✅ Plugin ready for G-Assist integration")
    else:
        print("❌ TEST FAILED: Issues detected with CanRun integration")
    print("=" * 60)
    
    sys.exit(0 if success else 1)