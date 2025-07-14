#!/usr/bin/env python3
"""Debug performance prediction issue."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dynamic_performance_predictor import DynamicPerformancePredictor

def test_performance_prediction():
    predictor = DynamicPerformancePredictor()
    
    # Test with RTX 3070 + i5-12400 system (mid-range)
    hardware_specs = {
        'gpu_model': 'RTX 3070',
        'cpu_model': 'Intel i5-12400',
        'ram_total_gb': 16,
        'cpu_cores': 6,
        'cpu_threads': 12,
        'cpu_frequency': 3600,
        'gpu_vram_gb': 8
    }
    
    game_requirements = {
        'recommended': {
            'processor': 'Intel Core i5-8400',
            'graphics': 'NVIDIA GeForce GTX 1060',
            'memory': 8
        }
    }
    
    # Test the assess_performance method
    try:
        result = predictor.assess_performance(
            hardware_specs=hardware_specs,
            game_requirements=game_requirements
        )
        print(f"Result type: {type(result)}")
        print(f"Performance Score: {result.score}")
        print(f"Performance Tier: {result.tier.name}")
        print(f"Expected FPS: {result.expected_fps}")
        print(f"Recommended Settings: {result.recommended_settings}")
        print(f"Recommended Resolution: {result.recommended_resolution}")
        
        if result.bottlenecks:
            print(f"Bottlenecks: {result.bottlenecks}")
        if result.upgrade_suggestions:
            print(f"Upgrade Suggestions: {result.upgrade_suggestions}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_performance_prediction()