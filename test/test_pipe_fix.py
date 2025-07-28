#!/usr/bin/env python3
"""
Test the fixed Windows pipe communication for G-Assist
"""

import json
import subprocess
import sys
import os
from pathlib import Path

def test_pipe_communication():
    """Test if the fixed plugin responds to G-Assist protocol"""
    print("🧪 Testing Fixed G-Assist Pipe Communication")
    print("=" * 50)
    
    plugin_path = Path(__file__).parent.parent / "plugin.py"
    
    # Test command that should work with our fixes
    test_command = {
        "tool_calls": [
            {
                "func": "canrun",  # Use the function name from manifest
                "params": {
                    "game_name": "Diablo 4"
                }
            }
        ]
    }
    
    try:
        # Start plugin process
        process = subprocess.Popen(
            [sys.executable, str(plugin_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(plugin_path.parent)
        )
        
        # Send command
        command_json = json.dumps(test_command)
        print(f"📤 Sending: {command_json}")
        
        # Write to stdin and close it
        process.stdin.write(command_json + "\n")
        process.stdin.close()
        
        # Wait for response with timeout
        try:
            stdout, stderr = process.communicate(timeout=15)
            
            print(f"📥 stdout: {repr(stdout)}")
            print(f"📥 stderr: {repr(stderr)}")
            print(f"📥 return code: {process.returncode}")
            
            # Check if response ends with <<END>>
            if stdout and stdout.endswith('<<END>>'):
                json_part = stdout[:-7]
                try:
                    response = json.loads(json_part)
                    print("✅ SUCCESS: G-Assist pipe communication working!")
                    print(f"   Response: {response}")
                    return True
                except json.JSONDecodeError:
                    print("❌ FAILED: Invalid JSON in response")
                    return False
            else:
                print("❌ FAILED: Response doesn't end with <<END>>")
                print(f"   Expected <<END>> marker, got: {repr(stdout[-10:]) if stdout else 'No output'}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ FAILED: Plugin timeout (15 seconds)")
            process.kill()
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Exception: {e}")
        return False

if __name__ == "__main__":
    success = test_pipe_communication()
    if success:
        print("\n🎉 G-Assist communication fix successful!")
    else:
        print("\n⚠️ G-Assist communication still has issues.")
    sys.exit(0 if success else 1)