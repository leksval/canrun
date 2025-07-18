#!/usr/bin/env python3
"""
Test G-Assist plugin protocol directly
"""
import sys
import os
import json
from io import StringIO

# Add the parent directory to the path so we can import the plugin modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_g_assist_message_parsing():
    """Test parsing G-Assist JSON messages"""
    
    # Import the plugin functions
    from plugin import process_enhanced_tool_call, read_g_assist_command, write_g_assist_response
    
    print("Testing G-Assist message parsing...")
    
    # Test 1: check_compatibility function
    test_message_1 = {
        "tool_calls": [
            {
                "func": "check_compatibility",
                "params": {
                    "game_name": "Diablo IV"
                }
            }
        ]
    }
    
    print(f"\nTest 1 - Compatibility Check:")
    print(f"Input: {json.dumps(test_message_1, indent=2)}")
    
    try:
        result = handle_g_assist_message(test_message_1)
        print(f"Output: {result}")
        
        # Verify the response format
        if result.endswith("<<END>>"):
            print("✓ Response format is correct (ends with <<END>>)")
        else:
            print("✗ Response format is incorrect (missing <<END>>)")
            
        # Try to parse as JSON
        json_part = result.replace("<<END>>", "").strip()
        try:
            parsed = json.loads(json_part)
            if "success" in parsed and "message" in parsed:
                print("✓ JSON structure is correct")
            else:
                print("✗ JSON structure is missing required fields")
        except json.JSONDecodeError as e:
            print(f"✗ JSON parsing failed: {e}")
            
    except Exception as e:
        print(f"✗ Error handling message: {e}")
    
    # Test 2: detect_hardware function
    test_message_2 = {
        "tool_calls": [
            {
                "func": "detect_hardware",
                "params": {}
            }
        ]
    }
    
    print(f"\nTest 2 - Hardware Detection:")
    print(f"Input: {json.dumps(test_message_2, indent=2)}")
    
    try:
        result = handle_g_assist_message(test_message_2)
        print(f"Output: {result}")
        
        # Verify the response format
        if result.endswith("<<END>>"):
            print("✓ Response format is correct (ends with <<END>>)")
        else:
            print("✗ Response format is incorrect (missing <<END>>)")
            
    except Exception as e:
        print(f"✗ Error handling message: {e}")
    
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
        result = handle_g_assist_message(test_message_3)
        print(f"Output: {result}")
        
        # Should return an error
        json_part = result.replace("<<END>>", "").strip()
        parsed = json.loads(json_part)
        if parsed.get("success") == False:
            print("✓ Correctly handled invalid function")
        else:
            print("✗ Should have returned error for invalid function")
            
    except Exception as e:
        print(f"✗ Error handling message: {e}")

if __name__ == "__main__":
    test_g_assist_message_parsing()