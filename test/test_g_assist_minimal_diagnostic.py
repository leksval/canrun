#!/usr/bin/env python3
"""
Minimal G-Assist Communication Diagnostic Tool
Tests the most basic G-Assist communication to isolate message visibility issues
"""

import json
import subprocess
import sys
import time
import os
from pathlib import Path

def test_minimal_response():
    """Test with the most minimal possible response that should work"""
    print("=== MINIMAL RESPONSE TEST ===")
    
    # Create a minimal test plugin that just returns a simple message
    minimal_plugin_content = '''
import json
import sys
from ctypes import byref, windll, wintypes

def read_command():
    try:
        STD_INPUT_HANDLE = -10
        pipe = windll.kernel32.GetStdHandle(STD_INPUT_HANDLE)
        chunks = []
        
        while True:
            BUFFER_SIZE = 4096
            message_bytes = wintypes.DWORD()
            buffer = bytes(BUFFER_SIZE)
            success = windll.kernel32.ReadFile(
                pipe, buffer, BUFFER_SIZE, byref(message_bytes), None
            )
            
            if not success:
                return None
                
            chunk = buffer.decode('utf-8')[:message_bytes.value]
            chunks.append(chunk)
            
            if message_bytes.value < BUFFER_SIZE:
                break
                
        return json.loads(''.join(chunks))
    except:
        return None

def write_response(response):
    try:
        STD_OUTPUT_HANDLE = -11
        pipe = windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        
        json_message = json.dumps(response) + "<<END>>"
        message_bytes = json_message.encode('utf-8')
        message_len = len(message_bytes)
        
        bytes_written = wintypes.DWORD()
        result = windll.kernel32.WriteFile(
            pipe, message_bytes, message_len, byref(bytes_written), None
        )
        
        # Force flush the pipe
        windll.kernel32.FlushFileBuffers(pipe)
        
        return result != 0
    except:
        return False

def main():
    while True:
        command = read_command()
        if command is None:
            continue
            
        # Test different minimal responses
        if "tool_calls" in command:
            for tool_call in command.get("tool_calls", []):
                func = tool_call.get("func")
                
                if func == "check_compatibility":
                    # Minimal success response - ASCII only, short message
                    response = {"success": True, "message": "TEST: Basic response working"}
                    write_response(response)
                else:
                    # Minimal failure response
                    response = {"success": False, "message": "Function not supported"}
                    write_response(response)

if __name__ == '__main__':
    main()
'''
    
    # Write the minimal test plugin
    test_plugin_path = Path("canrun/test/minimal_test_plugin.py")
    with open(test_plugin_path, 'w', encoding='utf-8') as f:
        f.write(minimal_plugin_content)
    
    print(f"Created minimal test plugin: {test_plugin_path}")
    
    # Test with a simple command
    test_command = {
        "tool_calls": [
            {
                "func": "check_compatibility",
                "params": {"game_name": "test"}
            }
        ]
    }
    
    try:
        # Run the minimal plugin
        process = subprocess.Popen(
            [sys.executable, str(test_plugin_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send command
        command_json = json.dumps(test_command)
        print(f"Sending command: {command_json}")
        
        process.stdin.write(command_json)
        process.stdin.flush()
        process.stdin.close()
        
        # Get response with timeout
        try:
            stdout, stderr = process.communicate(timeout=10)
            
            print(f"Response stdout: {repr(stdout)}")
            print(f"Response stderr: {repr(stderr)}")
            
            # Check if response contains expected format
            if "<<END>>" in stdout:
                print("[PASS] Response contains <<END>> tag")
                
                # Try to parse the JSON part
                json_part = stdout.split("<<END>>")[0]
                try:
                    response_data = json.loads(json_part)
                    print(f"[PASS] Valid JSON response: {response_data}")
                    
                    if response_data.get("success") and response_data.get("message"):
                        print("[PASS] Response has correct structure")
                        return True
                    else:
                        print("[FAIL] Response missing success/message fields")
                        return False
                        
                except json.JSONDecodeError as e:
                    print(f"[FAIL] Invalid JSON in response: {e}")
                    return False
            else:
                print("[FAIL] Response missing <<END>> tag")
                return False
                
        except subprocess.TimeoutExpired:
            print("[FAIL] Timeout waiting for response")
            process.kill()
            return False
            
    except Exception as e:
        print(f"[FAIL] Error running test: {e}")
        return False

def test_ascii_validation():
    """Test ASCII validation function"""
    print("\n=== ASCII VALIDATION TEST ===")
    
    def clean_ascii(text):
        """Remove all non-ASCII characters"""
        if not isinstance(text, str):
            text = str(text)
        # Replace common unicode characters
        replacements = {
            '®': '',
            '™': '',
            '©': '',
            '°': ' degrees',
            '–': '-',
            '—': '-',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'"
        }
        
        for unicode_char, replacement in replacements.items():
            text = text.replace(unicode_char, replacement)
        
        # Remove any remaining non-ASCII characters
        cleaned = ''.join(char for char in text if ord(char) < 128)
        return cleaned
    
    # Test cases
    test_strings = [
        "Normal ASCII text",
        "Text with RTX(R) 4070(TM)",
        "Temperature: 65 degrees C",
        "Unicode quotes: Hello",
        "Mixed: RTX(R) 4070(TM) at 65 degrees C with quotes",
        ""
    ]
    
    for test_str in test_strings:
        cleaned = clean_ascii(test_str)
        print(f"Original: {repr(test_str)}")
        print(f"Cleaned:  {repr(cleaned)}")
        
        # Verify all characters are ASCII
        all_ascii = all(ord(char) < 128 for char in cleaned)
        print(f"All ASCII: {all_ascii}")
        print()
    
    return True

def test_response_format_variations():
    """Test different response format variations to find what works"""
    print("\n=== RESPONSE FORMAT VARIATIONS TEST ===")
    
    # Test different response formats that might work with G-Assist
    format_variations = [
        # Standard format
        {"success": True, "message": "Test message"},
        
        # Alternative field names
        {"status": "success", "text": "Test message"},
        {"result": "success", "content": "Test message"},
        {"error": False, "response": "Test message"},
        
        # Different boolean representations
        {"success": "true", "message": "Test message"},
        {"success": 1, "message": "Test message"},
        
        # Minimal format
        {"message": "Test message"},
        
        # G-Assist specific format (if different)
        {"success": True, "data": {"message": "Test message"}},
    ]
    
    for i, format_test in enumerate(format_variations, 1):
        print(f"Format {i}: {json.dumps(format_test)}")
        
        # Test JSON serialization
        try:
            json_str = json.dumps(format_test)
            parsed_back = json.loads(json_str)
            
            # Check length
            full_response = json_str + "<<END>>"
            print(f"  Length: {len(full_response)} characters")
            print(f"  Valid JSON: ✓")
            
            # Check ASCII compliance
            all_ascii = all(ord(char) < 128 for char in full_response)
            print(f"  All ASCII: {'✓' if all_ascii else '✗'}")
            
        except Exception as e:
            print(f"  Error: {e}")
        
        print()
    
    return True

def main():
    """Run all diagnostic tests"""
    print("G-ASSIST MINIMAL DIAGNOSTIC TOOL")
    print("=" * 50)
    
    tests = [
        ("ASCII Validation", test_ascii_validation),
        ("Response Format Variations", test_response_format_variations),
        ("Minimal Response", test_minimal_response),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("DIAGNOSTIC RESULTS:")
    
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}")
    
    if all(results.values()):
        print("\nAll basic tests passed - issue likely in response content")
    else:
        print("\nBasic communication issues found - fix protocol first")

if __name__ == "__main__":
    main()