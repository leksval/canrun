#!/usr/bin/env python3

import unittest
import sys
import os

# Add the parent directory to the path to import canrun modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g_assist_response_fixer import clean_ascii_text, create_safe_g_assist_response
from plugin import format_canrun_response

class TestEmojiGAssistIntegration(unittest.TestCase):

    def test_clean_ascii_preserves_emojis(self):
        """Test that clean_ascii_text preserves emojis"""
        test_message = "🎮 Game Analysis: Cyberpunk 2077"
        result = clean_ascii_text(test_message)
        
        print("Testing clean_ascii_text with gaming emoji...")
        print(f"Input length: {len(test_message)}")
        print(f"Output length: {len(result)}")
        
        # Should preserve the gaming emoji
        assert "🎮" in result, f"Gaming emoji lost"
        assert "[GAME]" not in result, f"Emoji converted to text"
        
        print("PASS: clean_ascii_text preserves emojis")

    def test_safe_response_preserves_emojis(self):
        """Test that create_safe_g_assist_response preserves emojis"""
        test_message = "🎮 Game: Cyberpunk 2077\n💻 System: RTX 4090\n⚡ Performance: 60-120 FPS"
        result = create_safe_g_assist_response(True, test_message)
        
        print("Testing create_safe_g_assist_response with multiple emojis...")
        print(f"Input length: {len(test_message)}")
        print(f"Output length: {len(result['message'])}")
        
        # Should preserve all emojis
        assert "🎮" in result['message'], f"Gaming emoji lost"
        assert "💻" in result['message'], f"Computer emoji lost"
        assert "⚡" in result['message'], f"Lightning emoji lost"
        
        # Should not convert to text
        assert "[GAME]" not in result['message'], f"Gaming emoji converted"
        assert "[SYSTEM]" not in result['message'], f"System emoji converted"
        assert "[PERFORMANCE]" not in result['message'], f"Performance emoji converted"
        
        print("PASS: create_safe_g_assist_response preserves emojis")

    def test_format_response_emojis(self):
        """Test a mock result object to see how emojis are handled in format_canrun_response"""
        
        # Create a minimal mock result object for testing
        class MockResult:
            def __init__(self):
                self.game_name = "Cyberpunk 2077"
                
            def can_run_game(self):
                return True
                
            def exceeds_recommended_requirements(self):
                return True
        
        # Create mock with minimal required attributes
        mock_result = MockResult()
        
        # Test the format function
        result = format_canrun_response(mock_result)
        
        print(f"Formatted response length: {len(result)}")
        print("Response contains gaming emoji:", "🎮" in result)
        print("Response contains [GAME] text:", "[GAME]" in result)
        
        # Check if emojis are preserved
        if "🎮" in result:
            print("PASS: Gaming emoji preserved in formatted response")
        elif "[GAME]" in result:
            print("FAIL: Gaming emoji converted to [GAME] text")
            assert False, "Emoji converted to text instead of being preserved"
        else:
            print("WARNING: No gaming emoji found in response")
        
        print("PASS: format_canrun_response emoji test completed")

if __name__ == "__main__":
    print("Testing emoji integration with G-Assist...")
    unittest.main(verbosity=2)