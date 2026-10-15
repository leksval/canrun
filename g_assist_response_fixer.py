#!/usr/bin/env python3
"""
G-Assist Response Fixer
Utility functions to ensure G-Assist compatibility
"""

import json
import logging
import re


def _codepoint_is_g_assist_safe(char_code: int) -> bool:
    """ASCII, plus the emoji and symbol ranges G-Assist already accepts."""
    return (
        char_code < 128
        or 0x1F300 <= char_code <= 0x1F9FF
        or 0x1F600 <= char_code <= 0x1F64F
        or 0x1F680 <= char_code <= 0x1F6FF
        or 0x1F1E0 <= char_code <= 0x1F1FF
        or 0x2600 <= char_code <= 0x26FF
        or 0x2700 <= char_code <= 0x27BF
    )


def clean_ascii_text(text):
    """
    Clean text to ensure ASCII-only characters for G-Assist compatibility with emoji support
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
        chr(8211): '-',  # en dash
        chr(8212): '-',  # —
        chr(8220): '"',  # "
        chr(8221): '"',  # "
        chr(8216): "'",  # '
        chr(8217): "'",  # '

        # Bullet points and list symbols - CRITICAL FOR G-ASSIST
        chr(8226): '- ',  # • (bullet point)
        chr(8594): '-> ',  # → (arrow)
        chr(8729): '* ',  # ∙ (bullet operator)
        chr(9679): '* ',  # ● (black circle)
        chr(9702): '- ',  # ◦ (white bullet)
        chr(9632): '* ',  # ■ (black square)
        chr(9633): '- ',  # □ (white square)

        # Other common symbols
        chr(8230): '...',  # …
        chr(215): 'x',  # multiplication sign
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

        # Keep emojis as-is since G-Assist supports them
        # Removed emoji-to-text conversions to preserve original emojis
    }

    # Apply replacements
    for unicode_char, replacement in unicode_replacements.items():
        text = text.replace(unicode_char, replacement)

    # For G-Assist, keep emojis but remove other problematic Unicode.
    cleaned_chars = [char for char in text if _codepoint_is_g_assist_safe(ord(char))]

    cleaned = ''.join(cleaned_chars)

    # Clean up multiple spaces and tabs but preserve newlines
    # Replace multiple spaces/tabs with single space, but keep newlines
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Multiple spaces/tabs -> single space
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)  # Clean up extra whitespace between paragraphs
    return cleaned.strip()

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

        # Check for UTF-8 compliance (allow emojis but catch problematic characters)
        try:
            # First try UTF-8 encoding
            message.encode('utf-8')
            # Then check for problematic characters that cause G-Assist issues
            problematic_chars = []
            for char in message:
                char_code = ord(char)
                if _codepoint_is_g_assist_safe(char_code):
                    continue
                problematic_chars.append(f"\\u{char_code:04x}")

            if problematic_chars:
                return False, f"Message contains problematic Unicode characters: {', '.join(problematic_chars[:5])}"

        except UnicodeEncodeError as e:
            return False, f"Message contains encoding issues: {e}"

    # Check JSON serialization
    try:
        json.dumps(response_dict)
    except Exception as e:
        return False, f"Response not JSON serializable: {e}"

    return True, "Valid"

def create_safe_g_assist_response(success, message, max_length=1400):
    """
    Create a G-Assist compatible response with safety checks for structured format
    """
    # Clean the message
    if message:
        clean_message = clean_ascii_text(message)

        # Smart truncation for structured responses
        if len(clean_message) > max_length:
            # Try to preserve structure by truncating at section boundaries
            lines = clean_message.split('\n')
            truncated_lines = []
            current_length = 0

            for line in lines:
                if current_length + len(line) + 1 > max_length - 20:  # Reserve space for truncation notice
                    break
                truncated_lines.append(line)
                current_length += len(line) + 1  # +1 for newline

            if truncated_lines:
                clean_message = '\n'.join(truncated_lines) + "\n... [truncated]"
            else:
                # Fallback to simple truncation
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
    Format a response dictionary as JSON.

    Protocol V2 framing is applied by gassist_sdk. This helper does not append a delimiter.
    """
    # Validate first
    is_valid, error = validate_g_assist_response(response_dict)
    if not is_valid:
        logging.error('Invalid response format: %s', error)
        response_dict = create_safe_g_assist_response(False, f"Format error: {error}")

    try:
        json_message = json.dumps(response_dict)

        # Final UTF-8 check (allow emojis)
        try:
            json_message.encode('utf-8')
        except UnicodeEncodeError:
            # Last resort fallback
            json_message = json.dumps({
                "success": False,
                "message": "Response contained invalid characters"
            })

        return json_message

    except Exception as e:
        # Ultimate fallback
        logging.error('Failed to format response: %s', e)
        return json.dumps({
            "success": False,
            "message": "Response formatting failed"
        })

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
        print(f"  Input: {message[:50]!r}")
        print(f"  Output: {formatted[:100]!r}")
        print()

if __name__ == "__main__":
    test_response_fixer()
