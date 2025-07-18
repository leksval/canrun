#!/usr/bin/env python3
"""
Test suite for validating the plugin.exe G-Assist integration
Tests the compiled executable to ensure it works correctly with G-Assist protocol
"""

import subprocess
import json
import os
import sys
from pathlib import Path

def test_plugin_exe_exists():
    """Test that g-assist-plugin-canrun.exe exists in the root directory."""
    plugin_path = Path(__file__).parent.parent / "g-assist-plugin-canrun.exe"
    assert plugin_path.exists(), "g-assist-plugin-canrun.exe not found in root directory"
    print("✅ g-assist-plugin-canrun.exe exists")

def test_plugin_exe_shutdown():
    """Test g-assist-plugin-canrun.exe responds to shutdown command with correct G-Assist protocol."""
    plugin_path = Path(__file__).parent.parent / "g-assist-plugin-canrun.exe"
    
    # Test input following G-Assist protocol
    test_input = '{"tool_calls": [{"func": "shutdown", "params": {}}]}\n'
    
    try:
        result = subprocess.run(
            [str(plugin_path)],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        assert result.returncode == 0, f"Plugin failed with return code {result.returncode}"
        assert "<<END>>" in result.stdout, "Response missing G-Assist termination marker"
        
        # Parse the JSON response (before <<END>>)
        response_json = result.stdout.split("<<END>>")[0]
        response = json.loads(response_json)
        
        assert response.get("success") == True, "Response success should be True"
        assert "shutdown" in response.get("message", "").lower(), "Response should mention shutdown"
        
        print("✅ g-assist-plugin-canrun.exe shutdown test passed")
        print(f"   Response: {response}")
        
    except subprocess.TimeoutExpired:
        assert False, "Plugin timed out during shutdown test"
    except json.JSONDecodeError as e:
        assert False, f"Invalid JSON response: {e}"

def test_plugin_exe_compatibility_check():
    """Test g-assist-plugin-canrun.exe compatibility check functionality."""
    plugin_path = Path(__file__).parent.parent / "g-assist-plugin-canrun.exe"
    
    # Test compatibility check for a popular game
    test_input = '{"tool_calls": [{"func": "check_compatibility", "params": {"game_name": "Diablo 4"}}]}\n'
    
    try:
        result = subprocess.run(
            [str(plugin_path)],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        assert result.returncode == 0, f"Plugin failed with return code {result.returncode}"
        assert "<<END>>" in result.stdout, "Response missing G-Assist termination marker"
        
        # Parse the JSON response
        response_json = result.stdout.split("<<END>>")[0]
        response = json.loads(response_json)
        
        assert response.get("success") == True, "Compatibility check should succeed"
        
        message = response.get("message", "")
        assert "Diablo 4" in message, "Response should mention the game name"
        assert "Performance" in message, "Response should include performance analysis"
        
        print("✅ g-assist-plugin-canrun.exe compatibility check test passed")
        print(f"   Game analyzed: Diablo 4")
        print(f"   Response length: {len(message)} characters")
        
    except subprocess.TimeoutExpired:
        assert False, "Plugin timed out during compatibility check"
    except json.JSONDecodeError as e:
        assert False, f"Invalid JSON response: {e}"

def test_plugin_exe_hardware_detection():
    """Test g-assist-plugin-canrun.exe hardware detection functionality."""
    plugin_path = Path(__file__).parent.parent / "g-assist-plugin-canrun.exe"
    
    # Test hardware detection
    test_input = '{"tool_calls": [{"func": "detect_hardware", "params": {}}]}\n'
    
    try:
        result = subprocess.run(
            [str(plugin_path)],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        assert result.returncode == 0, f"Plugin failed with return code {result.returncode}"
        assert "<<END>>" in result.stdout, "Response missing G-Assist termination marker"
        
        # Parse the JSON response
        response_json = result.stdout.split("<<END>>")[0]
        response = json.loads(response_json)
        
        assert response.get("success") == True, "Hardware detection should succeed"
        
        message = response.get("message", "")
        assert "GPU" in message or "GRAPHICS" in message, "Response should include GPU information"
        assert "CPU" in message or "PROCESSOR" in message, "Response should include CPU information"
        
        print("✅ g-assist-plugin-canrun.exe hardware detection test passed")
        print(f"   Hardware info detected successfully")
        
    except subprocess.TimeoutExpired:
        assert False, "Plugin timed out during hardware detection"
    except json.JSONDecodeError as e:
        assert False, f"Invalid JSON response: {e}"

def test_plugin_exe_invalid_input():
    """Test g-assist-plugin-canrun.exe handles invalid input gracefully."""
    plugin_path = Path(__file__).parent.parent / "g-assist-plugin-canrun.exe"
    
    # Test with invalid JSON
    test_input = 'invalid json input\n'
    
    try:
        result = subprocess.run(
            [str(plugin_path)],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        assert result.returncode == 0, f"Plugin should handle invalid input gracefully"
        assert "<<END>>" in result.stdout, "Response missing G-Assist termination marker"
        
        # Should still provide a valid JSON response
        response_json = result.stdout.split("<<END>>")[0]
        response = json.loads(response_json)
        
        # Should handle as auto-detect with the invalid input
        assert "success" in response, "Response should have success field"
        
        print("✅ g-assist-plugin-canrun.exe invalid input handling test passed")
        
    except subprocess.TimeoutExpired:
        assert False, "Plugin timed out during invalid input test"
    except json.JSONDecodeError as e:
        assert False, f"Invalid JSON response even for error case: {e}"

def run_all_tests():
    """Run all g-assist-plugin-canrun.exe validation tests."""
    print("🧪 Running G-Assist g-assist-plugin-canrun.exe Validation Tests")
    print("=" * 60)
    
    tests = [
        test_plugin_exe_exists,
        test_plugin_exe_shutdown,
        test_plugin_exe_compatibility_check,
        test_plugin_exe_hardware_detection,
        test_plugin_exe_invalid_input
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All G-Assist plugin.exe tests passed!")
        print("✅ Plugin is ready for G-Assist integration")
    else:
        print("⚠️ Some tests failed - plugin needs fixes")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)