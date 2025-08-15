#!/usr/bin/env python3

import unittest
import sys
import os

# Add the parent directory to the path to import canrun modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g_assist_response_fixer import clean_ascii_text, validate_g_assist_response, create_safe_g_assist_response

class TestEmojiUnicodeFixing(unittest.TestCase):

    def test_emoji_preservation(self):
        """Test that emojis are preserved while problematic Unicode is fixed"""
        test_message = """🎮 Game Analysis: Cyberpunk 2077
📊 Performance Prediction: 65-85 FPS
• RTX 4070 Ti Super recommended
• DLSS 3.0 supported
🔧 Optimization Tips:
• Enable DLSS Quality mode
• Set Ray Tracing to Medium"""
        
        fixed_message = clean_ascii_text(test_message)
        
        # Should preserve emojis
        assert "🎮" in fixed_message
        assert "📊" in fixed_message
        assert "🔧" in fixed_message
        
        # Should convert bullet points to hyphens
        assert "•" not in fixed_message
        assert "- RTX 4070 Ti Super recommended" in fixed_message
        assert "- DLSS 3.0 supported" in fixed_message
        assert "- Enable DLSS Quality mode" in fixed_message
        
        print("PASS: Emoji preservation test passed")

    def test_validation_allows_emojis(self):
        """Test that validation allows messages with emojis"""
        message_with_emojis = "🎮 Gaming performance: 60-80 FPS 📊"
        response_dict = {"success": True, "message": message_with_emojis}
        
        is_valid, error_msg = validate_g_assist_response(response_dict)
        
        assert is_valid, f"Validation failed for emoji message: {error_msg}"
        print("PASS: Emoji validation test passed")

    def test_validation_blocks_problematic_unicode(self):
        """Test that validation blocks problematic Unicode characters"""
        message_with_bullet = "Performance: • Good"
        response_dict = {"success": True, "message": message_with_bullet}
        
        is_valid, error_msg = validate_g_assist_response(response_dict)
        
        assert not is_valid, "Validation should fail for bullet points"
        assert "problematic Unicode characters" in error_msg
        print("PASS: Problematic Unicode blocking test passed")

    def test_structured_response_fixing(self):
        """Test fixing a full structured response with emojis and Unicode"""
        structured_response = """🎮 Game Analysis: The Witcher 3
📋 System Requirements:
• Minimum: GTX 1060
• Recommended: RTX 3070
📊 Performance Prediction:
• 1080p: 85-110 FPS
• 1440p: 65-85 FPS
🔧 Optimization Tips:
• Enable DLSS if available
• Use High texture quality"""
        
        fixed_response = clean_ascii_text(structured_response)
        
        # Should preserve emojis
        assert "🎮" in fixed_response
        assert "📋" in fixed_response
        assert "📊" in fixed_response
        assert "🔧" in fixed_response
        
        # Should convert all bullet points
        assert "•" not in fixed_response
        bullet_lines = [
            "- Minimum: GTX 1060",
            "- Recommended: RTX 3070", 
            "- 1080p: 85-110 FPS",
            "- 1440p: 65-85 FPS",
            "- Enable DLSS if available",
            "- Use High texture quality"
        ]
        
        for line in bullet_lines:
            assert line in fixed_response, f"Missing converted line: {line}"
        
        print("PASS: Structured response fixing test passed")

    def test_response_length_limits(self):
        """Test that response length limits work with emoji content"""
        long_message_with_emojis = "🎮 " + "Very long message content " * 100 + "📊"
        
        safe_response = create_safe_g_assist_response(True, long_message_with_emojis)
        fixed_message = safe_response["message"]
        
        # Should be truncated but preserve some emojis
        assert len(fixed_message) <= 1500  # Our length limit
        assert "🎮" in fixed_message  # Should preserve emoji at start
        
        print("PASS: Response length limits with emojis test passed")

if __name__ == "__main__":
    print("Testing emoji Unicode fixing functionality...")
    unittest.main(verbosity=2)