#!/usr/bin/env python3
"""
Test G-Assist mode functionality by simulating stdin input.
"""

import subprocess
import json
import sys
import time

def test_g_assist_mode():
    """Test the G-Assist mode with simulated input."""
    print("Testing G-Assist mode functionality...")
    
    # Test command for G-Assist
    test_command = {
        "tool_calls": [
            {
                "func": "auto_detect",
                "params": {
                    "user_input": "Can I run Diablo IV?"
                }
            }
        ]
    }
    
    # Convert to JSON
    json_command = json.dumps(test_command)
    print(f"Sending command: {json_command}")
    
    try:
        # Start the plugin process
        process = subprocess.Popen(
            ['plugins/plugin.exe'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )
        
        # Send the command
        stdout, stderr = process.communicate(input=json_command + '\n', timeout=60)
        
        print(f"Exit code: {process.returncode}")
        print(f"STDOUT: {stdout}")
        if stderr:
            print(f"STDERR: {stderr}")
        
        # Check if response contains the expected G-Assist termination marker
        if '<<END>>' in stdout:
            print("✅ G-Assist protocol working correctly - found <<END>> marker")
            
            # Extract JSON response
            response_part = stdout.split('<<END>>')[0].strip()
            print(f"Raw response part: {repr(response_part[:200])}...")
            
            try:
                response = json.loads(response_part)
                print(f"✅ Valid JSON response received")
                print(f"Success: {response.get('success')}")
                print(f"Message length: {len(response.get('message', ''))}")
                return True
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON in response: {e}")
                print(f"Response part: {repr(response_part)}")
                return False
        else:
            print("❌ G-Assist protocol issue - no <<END>> marker found")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        process.kill()
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_g_assist_mode()
    sys.exit(0 if success else 1)