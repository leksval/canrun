#!/usr/bin/env python3
"""
G-Assist Response Fixer
Utility functions to ensure G-Assist compatibility
"""

import json
import re
import logging

def clean_ascii_text(text):
    """
    Clean text to ensure ASCII-only characters for G-Assist compatibility
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Common unicode character replacements
    unicode_replacements = {
        # Trademark symbols
        chr(174): '',  # ®
        chr(8482): '',  # ™
        chr(169): '',  # ©
        
        # Temperature and units
        chr(176): ' degrees',  # °
        chr(8451): 'C',  # ℃
        chr(8457): 'F',  # ℉
        
        # Dashes and quotes
        chr(8211): '-',  # –
        chr(8212): '-',  # —
        chr(8220): '"',  # "
        chr(8221): '"',  # "
        chr(8216): "'",  # '
        chr(8217): "'",  # '
        
        # Other common symbols
        chr(8230): '...',  # …
        chr(215): 'x',  # ×
        chr(247): '/',  # ÷
        chr(177): '+/-',  # ±
        
        # Currency
        chr(8364): 'EUR',  # €
        chr(163): 'GBP',  # £
        chr(165): 'YEN',  # ¥
        
        # Fractions
        chr(189): '1/2',  # ½
        chr(188): '1/4',  # ¼
        chr(190): '3/4',  # ¾
        
        # Math symbols
        chr(8805): '>=',  # ≥
        chr(8804): '<=',  # ≤
        chr(8800): '!=',  # ≠
        chr(8776): '~=',  # ≈
    }
    
    # Apply replacements
    for unicode_char, replacement in unicode_replacements.items():
        text = text.replace(unicode_char, replacement)
    
    # Remove any remaining non-ASCII characters
    cleaned = ''.join(char for char in text if ord(char) < 128)
    
    # Clean up multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def validate_g_assist_response(response_dict):
    """
    Validate that a response dictionary is G-Assist compatible
    Returns: (is_valid, error_message)
    """
    if not isinstance(response_dict, dict):
        return False, "Response must be a dictionary"
    
    # Check required fields
    if 'success' not in response_dict:
        return False, "Response must have 'success' field"
    
    # Validate success field
    success = response_dict['success']
    if not isinstance(success, bool):
        return False, f"'success' must be boolean, got {type(success)}"
    
    # Check message field if present
    if 'message' in response_dict:
        message = response_dict['message']
        if not isinstance(message, str):
            return False, f"'message' must be string, got {type(message)}"
        
        # Check ASCII compliance
        try:
            message.encode('ascii')
        except UnicodeEncodeError as e:
            return False, f"Message contains non-ASCII characters: {e}"
    
    # Check JSON serialization
    try:
        json_str = json.dumps(response_dict)
    except Exception as e:
        return False, f"Response not JSON serializable: {e}"
    
    return True, "Valid"

def create_safe_g_assist_response(success, message, max_length=500):
    """
    Create a G-Assist compatible response with safety checks
    """
    # Clean the message
    if message:
        clean_message = clean_ascii_text(message)
        
        # Truncate if too long
        if len(clean_message) > max_length:
            clean_message = clean_message[:max_length-15] + "... [truncated]"
    else:
        clean_message = ""
    
    # Create response
    response = {
        "success": bool(success),
        "message": clean_message
    }
    
    # Validate
    is_valid, error = validate_g_assist_response(response)
    if not is_valid:
        # Fallback to minimal safe response
        response = {
            "success": False,
            "message": f"Response validation failed: {error}"
        }
    
    return response

def format_g_assist_message(response_dict):
    """
    Format response for G-Assist pipe communication
    """
    # Validate first
    is_valid, error = validate_g_assist_response(response_dict)
    if not is_valid:
        logging.error(f"Invalid response format: {error}")
        response_dict = create_safe_g_assist_response(False, f"Format error: {error}")
    
    # Create JSON and append terminator
    try:
        json_message = json.dumps(response_dict) + "<<END>>"
        
        # Final ASCII check
        try:
            json_message.encode('ascii')
        except UnicodeEncodeError:
            # Last resort fallback
            json_message = json.dumps({
                "success": False,
                "message": "Response contained invalid characters"
            }) + "<<END>>"
        
        return json_message
    
    except Exception as e:
        # Ultimate fallback
        logging.error(f"Failed to format response: {e}")
        return json.dumps({
            "success": False,
            "message": "Response formatting failed"
        }) + "<<END>>"

def test_response_fixer():
    """Test the response fixing functions"""
    print("Testing G-Assist Response Fixer")
    print("=" * 40)
    
    # Test ASCII cleaning
    test_texts = [
        "RTX(R) 4070(TM) Graphics Card",
        "Temperature: 65 degrees C", 
        "Performance: Excellent rating",
        "Price: 599 EUR",
        "Ratio: 3/4 performance",
        "Normal ASCII text"
    ]
    
    print("ASCII Cleaning Test:")
    for text in test_texts:
        cleaned = clean_ascii_text(text)
        print(f"  '{text}' -> '{cleaned}'")
    
    print("\nResponse Validation Test:")
    test_responses = [
        {"success": True, "message": "Valid response"},
        {"success": "true", "message": "Invalid boolean"},
        {"message": "Missing success field"},
        {"success": True, "message": "RTX(R) 4070(TM)"},
        {"success": True},  # No message
    ]
    
    for i, resp in enumerate(test_responses, 1):
        is_valid, error = validate_g_assist_response(resp)
        print(f"  Test {i}: {'VALID' if is_valid else 'INVALID'} - {error}")
    
    print("\nSafe Response Creation Test:")
    test_cases = [
        (True, "RTX(R) 4070(TM) works great!"),
        (False, "Error with currency: 599 EUR"),
        (True, "A" * 600),  # Too long
    ]
    
    for success, message in test_cases:
        safe_resp = create_safe_g_assist_response(success, message)
        formatted = format_g_assist_message(safe_resp)
        print(f"  Input: {repr(message[:50])}")
        print(f"  Output: {repr(formatted[:100])}")
        print()

if __name__ == "__main__":
    test_response_fixer()