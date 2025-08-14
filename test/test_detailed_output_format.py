#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test detailed output formatting for CanRun G-Assist plugin
Validates the comprehensive emoji-rich format matches expected output
"""

import asyncio
import sys
import os

# Configure UTF-8 encoding for Windows console
if sys.platform == "win32":
    import locale
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    # Set console code page to UTF-8 on Windows
    os.system('chcp 65001 >nul 2>&1')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the root level plugin.py
from plugin import CanRunGAssistPlugin

async def test_detailed_format():
    """Test that the detailed output format includes all expected sections"""
    plugin = CanRunGAssistPlugin()
    
    if not plugin.canrun_engine:
        print("❌ CanRun engine not available for testing")
        return False
    
    # Test with a well-known game
    test_params = {"game_name": "Diablo I", "force_refresh": False}
    
    try:
        response = await plugin.check_game_compatibility(test_params)
        
        if not response.get("success", False):
            print(f"❌ Analysis failed: {response.get('message', 'Unknown error')}")
            return False
        
        message = response.get("message", "")
        
        # Check for required sections in detailed format
        required_sections = [
            "✅ CANRUN:",
            "🎮 YOUR SEARCH:",
            "🎮 STEAM MATCHED GAME:",
            "🏆 PERFORMANCE TIER:",
            "💻 SYSTEM SPECIFICATIONS:",
            "🎯 GAME REQUIREMENTS:",
            "⚡ PERFORMANCE PREDICTION:",
            "🔧 OPTIMIZATION SUGGESTIONS:",
            "🎯 CANRUN VERDICT:"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in message:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ Missing sections: {missing_sections}")
            print(f"📝 Actual output preview:\n{message[:500]}...")
            return False
        
        # Check for key details
        required_details = [
            "Expected FPS:",
            "Recommended Settings:",
            "Current Resolution:",
            "RTX Features:",
            "DLSS Support:"
        ]
        
        missing_details = []
        for detail in required_details:
            if detail not in message:
                missing_details.append(detail)
        
        if missing_details:
            print(f"❌ Missing details: {missing_details}")
            return False
        
        print("✅ Detailed format validation successful!")
        print(f"📊 Output contains all {len(required_sections)} required sections")
        print(f"📋 Output contains all {len(required_details)} required details")
        print(f"📏 Total output length: {len(message)} characters")
        
        # Print a sample of the output
        print(f"\n📝 Sample output (first 800 characters):\n{message[:800]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

async def main():
    """Run detailed format validation test"""
    print("🧪 Testing CanRun detailed output format...")
    success = await test_detailed_format()
    
    if success:
        print("\n✅ All tests passed! Detailed format is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed. Please check the output format.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())