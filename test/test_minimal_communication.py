#!/usr/bin/env python3
"""
Test minimal plugin communication to debug the G-Assist protocol issue.
"""

import json
import subprocess
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_minimal_plugin():
    """Test the minimal plugin communication."""
    print("Testing Minimal Plugin Communication")
    print("=" * 40)
    
    plugin_path = project_root / "test" / "test_minimal_plugin.py"
    
    # Test initialization
    print("\n[TEST] Plugin Initialization")
    print("-" * 30)
    
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
        
        # Send initialization command
        command = {"tool_calls": [{"func": "initialize", "params": {}}]}
        command_json = json.dumps(command)
        print(f"[SEND] {command_json}")
        
        # Send command to plugin
        process.stdin.write(command_json + "\n")
        process.stdin.close()
        
        # Read response with timeout
        try:
            stdout, stderr = process.communicate(timeout=10.0)
            print(f"[RECV] stdout: {repr(stdout)}")
            print(f"[RECV] stderr: {repr(stderr)}")
            
            # Check for proper response format
            if stdout.endswith('<<END>>'):
                json_part = stdout[:-7]
                try:
                    response = json.loads(json_part)
                    print(f"[PASS] Response: {response}")
                    return True
                except json.JSONDecodeError as e:
                    print(f"[FAIL] Invalid JSON: {e}")
                    return False
            else:
                print(f"[FAIL] No <<END>> marker found")
                return False
                
        except subprocess.TimeoutExpired:
            print("[FAIL] TIMEOUT: Plugin did not respond within 10 seconds")
            process.kill()
            return False
        except Exception as e:
            print(f"[FAIL] Communication error: {e}")
            process.terminate()
            return False
            
    except Exception as e:
        print(f"[FAIL] Test execution error: {e}")
        return False

if __name__ == "__main__":
    success = test_minimal_plugin()
    if success:
        print("\n[SUCCESS] Minimal plugin communication works!")
    else:
        print("\n[FAILED] Minimal plugin communication failed!")