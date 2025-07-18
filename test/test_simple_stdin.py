#!/usr/bin/env python3
"""
Simple test to debug stdin reading issue.
"""

import subprocess
import sys
import json

def test_simple_stdin():
    """Test simple stdin communication."""
    print("🧪 Testing Simple Stdin Communication")
    
    # Create a simple test script
    test_script = '''
import sys
import json

print("Plugin started", file=sys.stderr)

try:
    line = sys.stdin.readline()
    print(f"Received: {line}", file=sys.stderr)
    
    if line.strip():
        response = {"success": True, "message": f"Echo: {line.strip()}"}
        print(json.dumps(response))
    else:
        response = {"success": False, "message": "No input received"}
        print(json.dumps(response))
        
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    response = {"success": False, "message": f"Error: {str(e)}"}
    print(json.dumps(response))
'''
    
    # Write test script
    with open('test_stdin_simple.py', 'w') as f:
        f.write(test_script)
    
    try:
        # Test the simple script
        process = subprocess.Popen(
            [sys.executable, 'test_stdin_simple.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send test data
        test_input = '{"test": "hello"}'
        stdout, stderr = process.communicate(input=test_input, timeout=5)
        
        print(f"📤 Sent: {test_input}")
        print(f"📥 Stdout: {stdout}")
        print(f"📥 Stderr: {stderr}")
        print(f"📥 Return code: {process.returncode}")
        
        if stdout.strip():
            try:
                response = json.loads(stdout.strip())
                print(f"✅ Valid JSON response: {response}")
                return True
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON: {e}")
                return False
        else:
            print("❌ No stdout received")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout")
        process.kill()
        return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False
    finally:
        # Clean up
        import os
        if os.path.exists('test_stdin_simple.py'):
            os.remove('test_stdin_simple.py')

if __name__ == "__main__":
    success = test_simple_stdin()
    sys.exit(0 if success else 1)