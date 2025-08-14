#!/usr/bin/env python3
"""
Debug test for FPS prediction to identify why FPS values are stuck at 77.
This test checks both ML model loading and fallback calculations.
"""

import sys
import os
import asyncio
import logging

# Add the parent directory to sys.path to import canrun modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from canrun.src.dynamic_performance_predictor import DynamicPerformancePredictor
from canrun.src.canrun_engine import CanRunEngine

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

def test_ml_predictor_availability():
    """Test if ML predictor is available and loaded."""
    print("=== ML PREDICTOR AVAILABILITY TEST ===")
    
    predictor = DynamicPerformancePredictor()
    
    print(f"ML predictor object: {predictor.ml_predictor}")
    if predictor.ml_predictor:
        print(f"ML predictor loaded: {predictor.ml_predictor.is_loaded}")
        print(f"ML predictor model path exists: {hasattr(predictor.ml_predictor, 'model_path')}")
    else:
        print("ML predictor is None - fallback mode active")
    
    return predictor

def test_fps_calculation_direct():
    """Test FPS calculation directly with different games."""
    print("\n=== DIRECT FPS CALCULATION TEST ===")
    
    predictor = DynamicPerformancePredictor()
    
    # Test hardware specs (RTX 4090 system)
    hardware_specs = {
        'gpu_model': 'NVIDIA GeForce RTX 4090',
        'gpu_vram_gb': 24,
        'cpu_model': 'AMD Ryzen 7 7800X3D 8-Core Processor',
        'cpu_cores': 8,
        'ram_total_gb': 32,
        'display_width': 2560,
        'display_height': 1440
    }
    
    # Test different games
    test_games = [
        {'game_name': 'Cyberpunk 2077'},
        {'game_name': 'Counter-Strike 2'},
        {'game_name': 'Valorant'},
        {'game_name': 'Fortnite'},
        {'game_name': 'Call of Duty: Modern Warfare'},
        {'game_name': 'The Witcher 3'}
    ]
    
    for game_req in test_games:
        print(f"\nTesting {game_req['game_name']}:")
        
        result = predictor.assess_performance(hardware_specs, game_req)
        print(f"  - Score: {result.score}")
        print(f"  - Tier: {result.tier.name}")
        print(f"  - Expected FPS: {result.expected_fps}")
        print(f"  - Settings: {result.recommended_settings}")
        print(f"  - Resolution: {result.recommended_resolution}")

def test_fps_calculation_components():
    """Test individual components of FPS calculation."""
    print("\n=== FPS CALCULATION COMPONENTS TEST ===")
    
    predictor = DynamicPerformancePredictor()
    
    # Test GPU score calculation
    gpu_info = {'model': 'NVIDIA GeForce RTX 4090', 'vram': 24}
    gpu_score, gpu_passmark = predictor._get_gpu_score(gpu_info)
    print(f"GPU Score: {gpu_score}, PassMark: {gpu_passmark}")
    
    # Test CPU score calculation  
    cpu_info = {'model': 'AMD Ryzen 7 7800X3D 8-Core Processor', 'cores': 8, 'frequency': 4200}
    cpu_score, cpu_passmark = predictor._get_cpu_score(cpu_info)
    print(f"CPU Score: {cpu_score}, PassMark: {cpu_passmark}")
    
    # Test different FPS calculations
    test_cases = [
        ('Cyberpunk 2077', '1440p'),
        ('Counter-Strike 2', '1440p'),
        ('Valorant', '1440p'),
        ('General Gaming', '1440p'),
        ('Cyberpunk 2077', '1080p'),
        ('Counter-Strike 2', '1080p')
    ]
    
    for game_name, resolution in test_cases:
        game_req = {'game_name': game_name} if game_name != 'General Gaming' else None
        
        # Test ML prediction
        ml_fps = predictor._calculate_fps_with_ml(
            gpu_passmark=gpu_passmark,
            cpu_passmark=cpu_passmark,
            resolution=resolution,
            gpu_model='NVIDIA GeForce RTX 4090',
            game_requirements=game_req
        )
        
        # Test formula prediction
        formula_fps = predictor._calculate_fps(
            gpu_passmark=gpu_passmark,
            cpu_passmark=cpu_passmark,
            resolution=resolution,
            gpu_model='NVIDIA GeForce RTX 4090',
            game_requirements=game_req
        )
        
        print(f"{game_name} @ {resolution}: ML={ml_fps}, Formula={formula_fps}")

async def test_full_canrun_engine():
    """Test FPS prediction through the full CanRun engine."""
    print("\n=== FULL CANRUN ENGINE TEST ===")
    
    engine = CanRunEngine(enable_llm=False)  # Disable LLM for faster testing
    
    test_games = ['Cyberpunk 2077', 'Counter-Strike 2', 'Valorant', 'Diablo IV']
    
    for game_name in test_games:
        print(f"\nTesting {game_name} through CanRun engine:")
        try:
            result = await engine.check_game_compatibility(game_name, use_cache=False)
            
            print(f"  - Game: {result.game_name}")
            print(f"  - Score: {result.performance_prediction.score}")
            print(f"  - Tier: {result.performance_prediction.tier.name}")
            print(f"  - Expected FPS: {result.performance_prediction.expected_fps}")
            print(f"  - Settings: {result.performance_prediction.recommended_settings}")
            print(f"  - Cache used: {result.cache_used}")
            
        except Exception as e:
            print(f"  - ERROR: {e}")

def main():
    """Run all FPS debugging tests."""
    print("CANRUN FPS PREDICTION DEBUG TESTS")
    print("=" * 50)
    
    # Test 1: Check ML predictor availability
    predictor = test_ml_predictor_availability()
    
    # Test 2: Direct FPS calculation
    test_fps_calculation_direct()
    
    # Test 3: Component-level testing
    test_fps_calculation_components()
    
    # Test 4: Full engine testing
    print("\nRunning full engine tests...")
    asyncio.run(test_full_canrun_engine())
    
    print("\n" + "=" * 50)
    print("DEBUG TESTS COMPLETED")

if __name__ == "__main__":
    main()