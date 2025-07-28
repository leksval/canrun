#!/usr/bin/env python3

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from canrun_engine import CanRunEngine

async def test_diablo4_compatibility():
    """Test complete workflow for Diablo 4 compatibility check."""
    try:
        print("Testing Diablo 4 compatibility workflow...")
        
        # Initialize engine with LLM disabled for testing
        engine = CanRunEngine(enable_llm=False)
        print("✅ CanRunEngine initialized")
        
        # Test game compatibility analysis
        result = await engine.analyze_game_compatibility("Diablo 4")
        
        if result:
            print("✅ Diablo 4 analysis completed successfully")
            print(f"Game: {result.get('game_name', 'Unknown')}")
            print(f"Overall Rating: {result.get('overall_rating', 'Unknown')}")
            print(f"Performance Tier: {result.get('performance_tier', 'Unknown')}")
            print(f"Can Run: {result.get('can_run', 'Unknown')}")
            
            if 'requirements' in result:
                req = result['requirements']
                print(f"Min CPU: {getattr(req, 'minimum_cpu', 'Unknown')}")
                print(f"Min GPU: {getattr(req, 'minimum_gpu', 'Unknown')}")
                print(f"Min RAM: {getattr(req, 'minimum_ram_gb', 'Unknown')}GB")
            
            return True
        else:
            print("❌ Diablo 4 analysis failed - no result returned")
            return False
            
    except Exception as e:
        print(f"❌ Diablo 4 analysis failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing complete Diablo 4 compatibility workflow...")
    success = asyncio.run(test_diablo4_compatibility())
    sys.exit(0 if success else 1)