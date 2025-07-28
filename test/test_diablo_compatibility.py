#!/usr/bin/env python3
"""Test script to verify end-to-end compatibility analysis for Diablo 4."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import asyncio
from canrun_engine import CanRunEngine

async def test_diablo_compatibility():
    """Test the full compatibility analysis for Diablo 4."""
    
    # Initialize the engine with LLM disabled for testing
    engine = CanRunEngine(enable_llm=False)
    
    # Test the full compatibility check
    print("Testing full compatibility analysis for 'Diablo 4'...")
    
    try:
        # This should:
        # 1. Find "Diablo 4" -> "Diablo IV" in cache
        # 2. Get current system specs
        # 3. Analyze compatibility between system and game requirements
        # 4. Provide clear assessment
        
        result = await engine.check_game_compatibility("Diablo 4", use_cache=False)
        
        print(f"Game Name: {result.game_name}")
        print(f"System Specs: GPU={result.hardware_specs.gpu_model}, CPU={result.hardware_specs.cpu_model}, RAM={result.hardware_specs.ram_total_gb}GB")
        print(f"Game Requirements Found: {result.game_requirements.game_name}")
        print(f"  Minimum GPU: {result.game_requirements.minimum.get('graphics', 'N/A')}")
        print(f"  Recommended GPU: {result.game_requirements.recommended.get('graphics', 'N/A')}")
        print(f"Compatibility: {result.compatibility_analysis.overall_compatibility}")
        print(f"Overall Score: {result.compatibility_analysis.overall_score}")
        
        # Check performance predictions
        if result.performance_prediction.predictions:
            pred = result.performance_prediction.predictions[0]
            print(f"Expected FPS: {pred.get('fps', 'N/A')}")
            print(f"Performance Level: {pred.get('tier', 'N/A')}")
            print(f"Recommended Settings: {pred.get('settings', 'N/A')}")
            print(f"Recommended Resolution: {pred.get('resolution', 'N/A')}")
        
        # Check if we get proper LLM analysis
        if result.llm_analysis:
            print("LLM Analysis available:")
            for analysis_type, analysis_result in result.llm_analysis.items():
                print(f"  {analysis_type}: {analysis_result.confidence_score}")
        else:
            print("No LLM analysis performed")
        
        # Final assessment
        print(f"\n=== FINAL ASSESSMENT ===")
        print(f"System: RTX 4090 + Ryzen 7 7800X3D + 31GB RAM")
        print(f"Game: Diablo IV")
        print(f"Result: {result.compatibility_analysis.overall_compatibility}")
        
        if result.performance_prediction.predictions:
            pred = result.performance_prediction.predictions[0]
            print(f"Expected Performance: {pred.get('fps', 'N/A')} FPS at {pred.get('settings', 'N/A')} settings, {pred.get('resolution', 'N/A')}")
            print(f"Performance Tier: {pred.get('tier', 'N/A')}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_diablo_compatibility())