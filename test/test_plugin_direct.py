#!/usr/bin/env python3
"""
Direct test of CanRun plugin functionality without G-Assist communication layer.
This tests the core functions to ensure they work correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from canrun_engine import CanRunEngine
from privacy_aware_hardware_detector import PrivacyAwareHardwareDetector
from dynamic_performance_predictor import DynamicPerformancePredictor

async def test_canrun_core():
    """Test the core CanRun functionality."""
    print("🧪 Testing CanRun core functionality...")
    
    try:
        # Initialize components
        print("🔧 Initializing components...")
        canrun_engine = CanRunEngine(enable_llm=True)
        privacy_detector = PrivacyAwareHardwareDetector()
        performance_predictor = DynamicPerformancePredictor()
        print("✅ Components initialized successfully")
        
        # Test game compatibility check
        print("\n🎮 Testing game compatibility check for Diablo 4...")
        result = await canrun_engine.check_game_compatibility("Diablo 4")
        
        if result:
            print(f"✅ Game: {result.game_name}")
            print(f"✅ Source: {result.game_requirements.source}")
            print(f"✅ Can run: {result.can_run_game()}")
            print(f"✅ Exceeds recommended: {result.exceeds_recommended_requirements()}")
            print(f"✅ Performance tier: {result.performance_prediction.tier.name}")
            print(f"✅ Score: {result.performance_prediction.score}/100")
            print(f"✅ Expected FPS: {result.performance_prediction.expected_fps}")
            
            # Test hardware detection
            print("\n🔍 Testing hardware detection...")
            hardware_info = await privacy_detector.get_hardware_specs()
            print(f"✅ GPU: {hardware_info.gpu_model}")
            print(f"✅ CPU: {hardware_info.cpu_model}")
            print(f"✅ RAM: {hardware_info.ram_total_gb}GB")
            print(f"✅ RTX Support: {hardware_info.supports_rtx}")
            print(f"✅ DLSS Support: {hardware_info.supports_dlss}")
            
            print("\n🎯 CORE FUNCTIONALITY TEST: ✅ PASSED")
            return True
        else:
            print("❌ Game compatibility check failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plugin_functions():
    """Test the plugin functions in sync context."""
    print("\n🧪 Testing plugin functions...")
    
    # Simulate the plugin function calls
    try:
        # Test check_compatibility function
        print("🎮 Testing check_compatibility function...")
        
        # This simulates what the plugin would do
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Initialize engine
            canrun_engine = CanRunEngine(enable_llm=True)
            
            # Test compatibility check
            result = loop.run_until_complete(canrun_engine.check_game_compatibility("Diablo 4"))
            
            if result:
                message = f"CanRun Analysis: {result.game_name} - Tier {result.performance_prediction.tier.name} - {'EXCELLENT' if result.exceeds_recommended_requirements() else 'GOOD' if result.can_run_game() else 'INSUFFICIENT'}"
                print(f"✅ Plugin response: {message}")
                return True
            else:
                print("❌ Plugin function failed")
                return False
                
        finally:
            loop.close()
            
    except Exception as e:
        print(f"❌ Plugin function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 CanRun Plugin Direct Test")
    print("=" * 50)
    
    # Test async functionality
    async_result = asyncio.run(test_canrun_core())
    
    # Test plugin functions
    sync_result = test_plugin_functions()
    
    print("\n" + "=" * 50)
    if async_result and sync_result:
        print("🎉 ALL TESTS PASSED - Plugin is ready for G-Assist integration!")
    else:
        print("❌ Some tests failed - Plugin needs fixes")