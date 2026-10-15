"""
Test newline preservation in G-Assist responses
Verifies that the response fixer preserves line breaks properly
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from g_assist_response_fixer import clean_ascii_text, create_safe_g_assist_response

def test_newline_preservation():
    """Test that newlines are preserved in text cleaning."""
    print("\n[TEST] Newline Preservation")
    
    # Test cases with various newline scenarios
    test_cases = [
        {
            "name": "Simple newlines",
            "input": "Line 1\nLine 2\nLine 3",
            "expected_lines": 3
        },
        {
            "name": "Multi-line with emojis",
            "input": "🎮 **GAME**\n• FPS: 120\n• Settings: Ultra\n\n✅ VERDICT: CAN RUN",
            "expected_lines": 5
        },
        {
            "name": "Structured format",
            "input": """💻 **YOUR SYSTEM**
• GPU: RTX 4090
• CPU: Ryzen 7 7800X3D
• RAM: 32GB

⚡ **PERFORMANCE**
• FPS: 120-180
• Settings: Ultra""",
            "expected_lines": 8
        },
        {
            "name": "Multiple spaces with newlines",
            "input": "Line 1   \n   Line 2\n\n\nLine 3",
            "expected_lines": 4  # Should clean up extra newlines but preserve structure
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        
        # Test clean_ascii_text function
        cleaned = clean_ascii_text(test_case['input'])
        actual_lines = len(cleaned.split('\n'))
        
        print(f"Input lines: {len(test_case['input'].split('\\n'))}")
        print(f"Expected lines: {test_case['expected_lines']}")
        print(f"Actual lines: {actual_lines}")
        print(f"Cleaned text preview: {repr(cleaned[:100])}")
        
        # Check that we have the expected number of lines (allow some variance for cleanup)
        assert actual_lines >= test_case['expected_lines'] - 1, f"Too few lines: {actual_lines} < {test_case['expected_lines']}"
        assert actual_lines <= test_case['expected_lines'] + 1, f"Too many lines: {actual_lines} > {test_case['expected_lines']}"
        
        # Verify newlines are preserved
        assert '\n' in cleaned, "Newlines should be preserved"
        
        print("✅ PASS")

def test_g_assist_response_newlines():
    """Test that G-Assist response creation preserves newlines."""
    print("\n[TEST] G-Assist Response Newlines")
    
    # Create a multi-line message like our plugin would generate
    message = """🎮 **Game:** **Cyberpunk 2077**

💻 **YOUR SYSTEM**
• GPU: RTX 4090 (24GB)
• CPU: AMD Ryzen 7 7800X3D
• RAM: 32GB
• Display: **1440p**

⚡ **PERFORMANCE**
• FPS: **120-180**
• Settings: **Ultra**
• Score: **S Tier (95/100)**

✅ VERDICT: CAN RUN - EXCELLENT PERFORMANCE!"""
    
    # Create safe response
    response = create_safe_g_assist_response(True, message)
    
    # Check that newlines are preserved
    assert 'message' in response
    assert '\n' in response['message']
    
    lines = response['message'].split('\n')
    print(f"Response has {len(lines)} lines")
    print("First few lines:")
    for i, line in enumerate(lines[:5]):
        print(f"  {i}: {repr(line)}")
    
    # Verify structure is maintained
    assert any("Game:" in line for line in lines), "Game section should be present"
    assert any("YOUR SYSTEM" in line for line in lines), "System section should be present"
    assert any("PERFORMANCE" in line for line in lines), "Performance section should be present"
    assert any("VERDICT" in line for line in lines), "Verdict should be present"
    
    print("✅ PASS - Newlines preserved in G-Assist response")

def test_emoji_and_newline_combination():
    """Test that emojis and newlines work together correctly."""
    print("\n[TEST] Emoji and Newline Combination")
    
    text_with_emojis_and_newlines = """🎮 Game Analysis
💻 System: RTX 3090
⚡ Performance: Excellent
🔧 Optimization: Enable DLSS
✅ Result: Can Run"""
    
    cleaned = clean_ascii_text(text_with_emojis_and_newlines)
    
    # Should preserve both emojis and newlines
    assert '\n' in cleaned, "Newlines should be preserved"
    assert '🎮' in cleaned, "Game emoji should be preserved"
    assert '💻' in cleaned, "Computer emoji should be preserved"
    assert '⚡' in cleaned, "Lightning emoji should be preserved"
    assert '🔧' in cleaned, "Tool emoji should be preserved"
    assert '✅' in cleaned, "Check mark emoji should be preserved"
    
    lines = cleaned.split('\n')
    assert len(lines) == 5, f"Should have 5 lines, got {len(lines)}"
    
    print(f"Cleaned text: {repr(cleaned)}")
    print("✅ PASS - Emojis and newlines work together")

def main():
    """Run all newline preservation tests."""
    print("Testing Newline Preservation in G-Assist Responses")
    print("=" * 55)
    
    try:
        test_newline_preservation()
        test_g_assist_response_newlines()
        test_emoji_and_newline_combination()
        
        print("\n" + "=" * 55)
        print("✅ ALL TESTS PASSED!")
        print("Newlines should now display properly in G-Assist")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)