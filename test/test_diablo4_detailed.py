#!/usr/bin/env python3

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from canrun_engine import CanRunEngine

async def test_diablo4_detailed():
    """Test complete workflow for Diablo 4 with detailed debugging."""
    try:
        print("=== Testing Diablo 4 compatibility workflow with detailed debugging ===")
        
        # Initialize engine
        engine = CanRunEngine()
        print("✅ CanRunEngine initialized")
        
        # Test hardware detection first
        print("\n🔧 Testing hardware detection...")
        hardware = await engine.hardware_detector.get_hardware_specs()
        print(f"GPU: {hardware.gpu_model}")
        print(f"CPU: {hardware.cpu_model}")
        print(f"RAM: {hardware.ram_total_gb}GB")
        
        # Test game requirements fetching
        print("\n📋 Testing game requirements fetching...")
        requirements = await engine.requirements_fetcher.fetch_requirements("Diablo 4")
        if requirements:
            print(f"✅ Requirements found for: {requirements.game_name}")
            print(f"Min CPU: {requirements.minimum_cpu}")
            print(f"Min GPU: {requirements.minimum_gpu}")
            print(f"Min RAM: {requirements.minimum_ram_gb}GB")
            print(f"Source: {requirements.source}")
        else:
            print("❌ No requirements found for Diablo 4")
        
        # Test compatibility analysis  
        if requirements:
            print("\n🔍 Testing compatibility analysis...")
            compatibility = engine.compatibility_analyzer.analyze_compatibility("Diablo 4", hardware, requirements)
            print(f"Overall compatibility: {compatibility.overall_compatibility}")
            print(f"Can run minimum: {compatibility.can_run_minimum}")
            print(f"Can run recommended: {compatibility.can_run_recommended}")
            print(f"Overall score: {compatibility.overall_score}")
        
        # Test performance prediction
        if requirements:
            print("\n⚡ Testing performance prediction...")
            performance = engine.performance_predictor.predict_advanced_performance(hardware, requirements)
            print(f"Performance tier: {performance.tier}")
            print(f"Expected FPS: {performance.expected_fps}")
            print(f"Recommended settings: {performance.recommended_settings}")
        
        # Test full game compatibility analysis
        print("\n🎮 Testing full game compatibility analysis...")
        result = await engine.analyze_game_compatibility("Diablo 4")
        
        if result:
            print("✅ Full analysis completed successfully")
            print(f"Game: {result.get('game_name', 'Unknown')}")
            print(f"Overall Rating: {result.get('overall_rating', 'Unknown')}")
            print(f"Performance Tier: {result.get('performance_tier', 'Unknown')}")
            print(f"Can Run: {result.get('can_run', 'Unknown')}")
            
            # Show full result structure
            print("\n📊 Full result structure:")
            for key, value in result.items():
                if isinstance(value, dict):
                    print(f"{key}: {type(value).__name__} with {len(value)} keys")
                else:
                    print(f"{key}: {value}")
            
            return True
        else:
            print("❌ Full analysis returned no result")
            return False
            
    except Exception as e:
        print(f"❌ Analysis failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_diablo4_detailed())
    sys.exit(0 if success else 1)