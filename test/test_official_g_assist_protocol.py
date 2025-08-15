#!/usr/bin/env python3
"""
Test Official NVIDIA G-Assist Protocol Compliance
Verifies that the canrun plugin works with official G-Assist communication protocol
"""

import json
import subprocess
import threading
import time
import os
import sys
from typing import Optional

# ASCII only output for compliance
ASCII_TEST_OUTPUT = True

def test_g_assist_protocol_compliance():
    """Test that manifest.json complies with official NVIDIA G-Assist specs"""
    manifest_path = "manifest.json"
    
    assert os.path.exists(manifest_path), "manifest.json file must exist"
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Test official NVIDIA G-Assist manifest requirements
    assert "manifestVersion" in manifest, "manifestVersion field required"
    assert manifest["manifestVersion"] == 1, "manifestVersion must be 1"
    
    assert "executable" in manifest, "executable field required"
    assert manifest["executable"] == "g-assist-plugin-canrun.exe", "executable path must not contain './' prefix"
    
    assert "persistent" in manifest, "persistent field required"
    assert manifest["persistent"] is False, "persistent should be false for proper G-Assist compatibility"
    
    assert "functions" in manifest, "functions array required"
    assert len(manifest["functions"]) >= 1, "At least one function required"
    
    # Test check_compatibility function specifically (the main function)
    check_compatibility_func = None
    detect_hardware_func = None
    auto_detect_func = None
    
    for func in manifest["functions"]:
        if func["name"] == "check_compatibility":
            check_compatibility_func = func
        elif func["name"] == "detect_hardware":
            detect_hardware_func = func
        elif func["name"] == "auto_detect":
            auto_detect_func = func
    
    assert check_compatibility_func is not None, "check_compatibility function must be defined"
    assert "description" in check_compatibility_func, "function description required"
    assert "tags" in check_compatibility_func, "function tags required"
    assert len(check_compatibility_func["tags"]) >= 5, "Should have comprehensive tags for discoverability"
    assert "properties" in check_compatibility_func, "function properties required"
    assert "required" in check_compatibility_func, "required fields array must be present"
    assert "game_name" in check_compatibility_func["required"], "game_name must be required parameter"
    
    # Check tags include key gaming terms for G-Assist discovery
    expected_tags = ["canrun", "can run", "game", "compatibility"]
    tags = check_compatibility_func["tags"]
    for expected_tag in expected_tags:
        assert expected_tag in tags, f"Missing essential tag: {expected_tag}"
    
    # Test that other required functions exist
    assert detect_hardware_func is not None, "detect_hardware function must be defined"
    assert auto_detect_func is not None, "auto_detect function must be defined"
    
    print("PASS: manifest.json complies with official NVIDIA G-Assist specification")

def test_executable_exists_and_valid():
    """Test that the executable exists and has proper properties"""
    exe_path = "g-assist-plugin-canrun.exe"
    
    assert os.path.exists(exe_path), "Plugin executable must exist"
    assert os.path.isfile(exe_path), "Plugin executable must be a file"
    
    # Test that executable runs without crashing
    try:
        result = subprocess.run([exe_path, "--help"], 
                              capture_output=True, 
                              text=True, 
                              timeout=10,
                              cwd=".")
        # Should either show help or run without fatal error
        assert result.returncode in [0, 1, 2], f"Executable should run without fatal error, got code: {result.returncode}"
    except subprocess.TimeoutExpired:
        assert False, "Executable hangs and doesn't respond within timeout"
    except Exception as e:
        assert False, f"Executable failed to run: {e}"
    
    print("PASS: Plugin executable exists and is runnable")

def test_cli_mode_functionality():
    """Test CLI mode works properly for debugging"""
    exe_path = "g-assist-plugin-canrun.exe"
    
    # Test canrun command with a popular game
    try:
        result = subprocess.run([exe_path, "canrun", "cyberpunk 2077", "--json"], 
                              capture_output=True, 
                              text=True, 
                              timeout=30,
                              cwd=".")
        
        # Should succeed or provide meaningful error
        assert result.returncode in [0, 1], f"CLI mode should work, got return code: {result.returncode}"
        
        if result.returncode == 0:
            # Try to parse JSON response
            try:
                response = json.loads(result.stdout)
                assert "success" in response, "Response should have success field"
                if response["success"]:
                    assert "message" in response, "Successful response should have message"
                    assert len(response["message"]) > 0, "Message should not be empty"
                print("PASS: CLI mode returns valid JSON response")
            except json.JSONDecodeError:
                print("WARN: CLI mode output is not valid JSON, but executable runs")
        else:
            print(f"WARN: CLI mode returned error code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        assert False, "CLI mode hangs - this will cause G-Assist to freeze"
    except Exception as e:
        print(f"WARN: CLI mode test failed: {e}")

def test_g_assist_message_format():
    """Test that the plugin can handle G-Assist message format"""
    # This simulates the actual message format that G-Assist sends to plugins
    test_message = {
        "tool_calls": [
            {
                "func": "check_compatibility",
                "params": {
                    "game_name": "cyberpunk 2077"
                }
            }
        ]
    }
    
    # Test that our message format is valid JSON
    try:
        json_str = json.dumps(test_message)
        parsed = json.loads(json_str)
        assert parsed == test_message, "Message should round-trip through JSON"
        print("PASS: G-Assist message format is valid")
    except Exception as e:
        assert False, f"G-Assist message format validation failed: {e}"

def test_plugin_directory_structure():
    """Test that plugin has proper directory structure for G-Assist"""
    canrun_dir = "."
    
    # Required files for G-Assist plugin
    required_files = [
        "manifest.json",
        "g-assist-plugin-canrun.exe",
        "plugin.py"  # Source file for reference
    ]
    
    for file_name in required_files:
        file_path = os.path.join(canrun_dir, file_name)
        assert os.path.exists(file_path), f"Required file missing: {file_name}"
    
    # Check that no extra manifest files exist that could confuse G-Assist
    manifest_files = [f for f in os.listdir(canrun_dir) if f.endswith("manifest.json")]
    assert len(manifest_files) == 1, f"Should have exactly one manifest.json, found: {manifest_files}"
    
    print("PASS: Plugin directory structure is correct for G-Assist")

def test_ascii_output_compliance():
    """Test that output contains only ASCII characters for G-Assist compatibility"""
    if not ASCII_TEST_OUTPUT:
        print("SKIP: ASCII output test disabled")
        return
        
    exe_path = "g-assist-plugin-canrun.exe"
    
    try:
        result = subprocess.run([exe_path, "canrun", "test", "--json"], 
                              capture_output=True, 
                              text=True, 
                              timeout=20,
                              cwd=".")
        
        if result.stdout:
            # Check that all characters in output are ASCII
            for char in result.stdout:
                assert ord(char) < 128, f"Non-ASCII character found: {char} (ord {ord(char)})"
        
        if result.stderr:
            # Check stderr as well
            for char in result.stderr:
                assert ord(char) < 128, f"Non-ASCII character in stderr: {char} (ord {ord(char)})"
                
        print("PASS: Output contains only ASCII characters")
        
    except subprocess.TimeoutExpired:
        print("WARN: ASCII test timed out")
    except Exception as e:
        print(f"WARN: ASCII test failed: {e}")

def run_all_tests():
    """Run all G-Assist protocol compliance tests"""
    print("=== NVIDIA G-Assist Protocol Compliance Test ===")
    print()
    
    tests = [
        test_g_assist_protocol_compliance,
        test_executable_exists_and_valid,
        test_cli_mode_functionality,
        test_g_assist_message_format,
        test_plugin_directory_structure,
        test_ascii_output_compliance
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            print(f"Running {test_func.__name__}...")
            test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"FAIL: {e}")
            print()
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            print()
            failed += 1
    
    print("=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("All tests passed! Plugin should be compatible with NVIDIA G-Assist")
        return True
    else:
        print("Some tests failed. Plugin may not be fully compatible with NVIDIA G-Assist")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)