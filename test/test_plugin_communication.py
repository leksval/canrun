#!/usr/bin/env python3
"""
Test script to verify G-Assist plugin communication protocol
"""
import subprocess
import json
import sys

def test_plugin_communication():
    """Test the plugin with G-Assist JSON message format"""
    
    # Test message in G-Assist format
    test_message = {
        "tool_calls": [
            {
                "func": "check_compatibility",
                "params": {
                    "game_name": "Diablo IV"
                }
            }
        ]
    }
    
    print("Testing G-Assist plugin communication...")
    print(f"Sending: {json.dumps(test_message, indent=2)}")
    print("-" * 50)
    
    try:
        # Start the plugin process
        process = subprocess.Popen(
            ["uv", "run", "python", "plugin.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="."
        )
        
        # Send the JSON message
        json_input = json.dumps(test_message) + "\n"
        process.stdin.write(json_input)
        process.stdin.flush()
        
        # Read the response
        stdout = process.stdout.readline()
        
        print("Plugin Response:")
        print(stdout)
        
        # Clean up the process
        process.stdin.close()
        process.wait(timeout=5)
        
        # Check for errors
        stderr = process.stderr.read()
        if stderr:
            print("Plugin Errors:")
            print(stderr)
            
        print(f"Exit code: {process.returncode}")
        
    except subprocess.TimeoutExpired:
        print("Plugin timed out during cleanup")
        process.kill()
    except Exception as e:
        print(f"Error testing plugin: {e}")

if __name__ == "__main__":
    test_plugin_communication()