#!/usr/bin/env python3
"""
Test G-Assist plugin protocol directly
"""
import sys
import os
import json
import subprocess
from io import StringIO

def test_g_assist_message_parsing():
    """Test parsing G-Assist JSON messages using subprocess to match real G-Assist usage"""
    
    print("Testing G-Assist message parsing...")
    
    # Test 1: canrun function with Diablo IV
    test_message_1 = {
        "tool_calls": [
            {
                "func": "canrun",
                "params": {
                    "game_name": "Diablo IV"
                }
            }
        ]
    }
    
    print(f"\nTest 1 - CanRun Compatibility Check:")
    print(f"Input: {json.dumps(test_message_1, indent=2)}")
    
    try:
        # Run plugin as subprocess
        process = subprocess.Popen(
            [sys.executable, "canrun/plugin.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="."
        )
        
        stdout, stderr = process.communicate(input=json.dumps(test_message_1), timeout=30)
        
        print(f"Output: {stdout}")
        
        # Verify the response format
        if "<<END>>" in stdout:
            print("✓ Response format is correct (contains <<END>>)")
            
            # Extract JSON part
            json_part = stdout.split("<<END>>")[0]
            
            # Find JSON start
            json_start = json_part.find('{')
            if json_start != -1:
                json_end = json_part.rfind('}') + 1
                json_text = json_part[json_start:json_end]
            else:
                json_text = json_part
                
            try:
                parsed = json.loads(json_text)
                if "success" in parsed and "message" in parsed:
                    print("✓ JSON structure is correct")
                    if parsed.get("success"):
                        print("✓ CanRun function executed successfully")
                    else:
                        print(f"⚠️ CanRun returned error: {parsed.get('message')}")
                else:
                    print("✗ JSON structure is missing required fields")
            except json.JSONDecodeError as e:
                print(f"✗ JSON parsing failed: {e}")
        else:
            print("✗ Response format is incorrect (missing <<END>>)")
            
    except subprocess.TimeoutExpired:
        print("✗ Plugin timed out")
        process.kill()
    except Exception as e:
        print(f"✗ Error running plugin: {e}")
    
    # Test 2: Initialize function
    test_message_2 = {
        "tool_calls": [
            {
                "func": "initialize",
                "params": {}
            }
        ]
    }
    
    print(f"\nTest 2 - Initialize Function:")
    print(f"Input: {json.dumps(test_message_2, indent=2)}")
    
    try:
        process = subprocess.Popen(
            [sys.executable, "canrun/plugin.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="."
        )
        
        stdout, stderr = process.communicate(input=json.dumps(test_message_2), timeout=10)
        
        print(f"Output: {stdout}")
        
        if "<<END>>" in stdout:
            print("✓ Response format is correct (contains <<END>>)")
        else:
            print("✗ Response format is incorrect (missing <<END>>)")
            
    except subprocess.TimeoutExpired:
        print("✗ Plugin timed out")
        process.kill()
    except Exception as e:
        print(f"✗ Error running plugin: {e}")
    
    # Test 3: Invalid function
    test_message_3 = {
        "tool_calls": [
            {
                "func": "invalid_function",
                "params": {}
            }
        ]
    }
    
    print(f"\nTest 3 - Invalid Function:")
    print(f"Input: {json.dumps(test_message_3, indent=2)}")
    
    try:
        process = subprocess.Popen(
            [sys.executable, "canrun/plugin.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="."
        )
        
        stdout, stderr = process.communicate(input=json.dumps(test_message_3), timeout=10)
        
        print(f"Output: {stdout}")
        
        if "<<END>>" in stdout:
            # Extract and parse JSON
            json_part = stdout.split("<<END>>")[0]
            json_start = json_part.find('{')
            if json_start != -1:
                json_end = json_part.rfind('}') + 1
                json_text = json_part[json_start:json_end]
                
                try:
                    parsed = json.loads(json_text)
                    if parsed.get("success") == False:
                        print("✓ Correctly handled invalid function")
                    else:
                        print("✗ Should have returned error for invalid function")
                except json.JSONDecodeError:
                    print("✗ Could not parse error response")
        else:
            print("✗ No proper response format")
            
    except subprocess.TimeoutExpired:
        print("✗ Plugin timed out")
        process.kill()
    except Exception as e:
        print(f"✗ Error running plugin: {e}")

if __name__ == "__main__":
    test_g_assist_message_parsing()