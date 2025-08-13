#!/usr/bin/env python3
"""
Test Official NVIDIA G-Assist Protocol Implementation
Based on official documentation from https://github.com/NVIDIA/G-Assist

This test verifies that our plugin correctly implements the official G-Assist communication protocol.
"""

import json
import subprocess
import sys
import os
import time
import threading
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_official_g_assist_protocol():
    """Test the official G-Assist protocol implementation."""
    print("Testing Official NVIDIA G-Assist Protocol Implementation")
    print("=" * 60)
    
    # Test cases based on actual plugin implementation
    test_cases = [
        {
            "name": "Plugin Initialization",
            "command": {
                "tool_calls": [
                    {
                        "func": "initialize",
                        "params": {}
                    }
                ]
            },
            "expected_success": True
        },
        {
            "name": "Game Compatibility Check - Diablo 4",
            "command": {
                "tool_calls": [
                    {
                        "func": "canrun",
                        "params": {
                            "game_name": "Diablo 4"
                        }
                    }
                ]
            },
            "expected_success": True
        },
        {
            "name": "Game Compatibility Check - Cyberpunk 2077",
            "command": {
                "tool_calls": [
                    {
                        "func": "canrun",
                        "params": {
                            "game_name": "Cyberpunk 2077"
                        }
                    }
                ]
            },
            "expected_success": True
        },
        {
            "name": "Shutdown Command",
            "command": {
                "tool_calls": [
                    {
                        "func": "shutdown",
                        "params": {}
                    }
                ]
            },
            "expected_success": True
        }
    ]
    
    plugin_path = project_root / "plugin.py"
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[TEST] {i}: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Start plugin process
            process = subprocess.Popen(
                [sys.executable, str(plugin_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(project_root)
            )
            
            # Send command in official G-Assist format
            command_json = json.dumps(test_case["command"])
            print(f"[SEND] Sending: {command_json}")
            
            # Send command to plugin
            process.stdin.write(command_json + "\n")
            process.stdin.flush()
            
            # Read response with timeout using communicate
            try:
                # Close stdin to signal end of input
                process.stdin.close()
                
                # Use communicate with timeout
                stdout, stderr = process.communicate(timeout=10.0)
                output = stdout
                
            except subprocess.TimeoutExpired:
                print("[FAIL] TIMEOUT: Plugin did not respond within 10 seconds")
                process.kill()
                continue
            except Exception as e:
                print(f"[FAIL] ERROR: Communication failed: {e}")
                process.terminate()
                continue
            
            print(f"[RECV] Raw output: {repr(output)}")
            
            # Check for official G-Assist termination marker
            if not output.endswith('<<END>>'):
                print("[FAIL] FAILED: Response does not end with official '<<END>>' marker")
                continue
            
            # Remove termination marker and parse JSON
            json_response = output[:-7]  # Remove '<<END>>'
            
            try:
                response = json.loads(json_response)
                print(f"📋 Parsed response: {response}")
                
                # Validate response structure
                if not isinstance(response, dict):
                    print("[FAIL] FAILED: Response is not a dictionary")
                    continue
                
                if "success" not in response:
                    print("[FAIL] FAILED: Response missing 'success' field")
                    continue
                
                if "message" not in response:
                    print("[FAIL] FAILED: Response missing 'message' field")
                    continue
                
                # Check expected success
                if response["success"] == test_case["expected_success"]:
                    print("[PASS] PASSED: Official G-Assist protocol compliance verified")
                else:
                    print(f"[FAIL] FAILED: Expected success={test_case['expected_success']}, got {response['success']}")
                    print(f"   Message: {response.get('message', 'No message')}")
                
            except json.JSONDecodeError as e:
                print(f"[FAIL] FAILED: Invalid JSON response: {e}")
                print(f"   Raw response: {repr(json_response)}")
            
            # Clean up process
            try:
                process.terminate()
                process.wait(timeout=2)
            except:
                process.kill()
                
        except Exception as e:
            print(f"[FAIL] FAILED: Test execution error: {e}")
    
    print("\n" + "=" * 60)
    print("[DONE] Official G-Assist Protocol Test Complete")

def test_stdin_stdout_communication():
    """Test stdin/stdout communication specifically."""
    print("\n[TEST] Testing stdin/stdout Communication")
    print("=" * 40)
    
    plugin_path = project_root / "plugin.py"
    
    try:
        # Test simple initialization
        process = subprocess.Popen(
            [sys.executable, str(plugin_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(project_root)
        )
        
        # Send simple command
        command = {"tool_calls": [{"func": "initialize", "params": {}}]}
        command_json = json.dumps(command)
        
        print(f"[SEND] Sending command: {command_json}")
        
        # Write to stdin
        process.stdin.write(command_json + "\n")
        process.stdin.close()  # Close stdin to signal end of input
        
        # Read from stdout
        stdout, stderr = process.communicate(timeout=15)
        
        print(f"[RECV] stdout: {repr(stdout)}")
        print(f"[RECV] stderr: {repr(stderr)}")
        
        # Verify response format
        if stdout.endswith('<<END>>'):
            json_part = stdout[:-7]
            try:
                response = json.loads(json_part)
                print("[PASS] PASSED: stdin/stdout communication working")
                print(f"   Response: {response}")
            except json.JSONDecodeError:
                print("[FAIL] FAILED: Invalid JSON in stdout")
        else:
            print("[FAIL] FAILED: stdout does not end with <<END>>")
            
    except subprocess.TimeoutExpired:
        print("[FAIL] FAILED: Communication timeout")
        process.kill()
    except Exception as e:
        print(f"[FAIL] FAILED: Communication error: {e}")

if __name__ == "__main__":
    test_official_g_assist_protocol()
    test_stdin_stdout_communication()