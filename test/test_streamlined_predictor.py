#!/usr/bin/env python3
"""
Quick test script to validate the streamlined performance predictor
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dynamic_performance_predictor import StreamlinedPerformancePredictor, PerformanceTier

def test_streamlined_predictor():
    """Test the streamlined performance predictor with various GPU classes"""
    
    print("Testing Streamlined Performance Predictor")
    print("=" * 60)
    
    predictor = StreamlinedPerformancePredictor()
    
    # Test cases for different GPU classes
    test_cases = [
        {
            "name": "RTX 5090 Flagship",
            "hardware": {
                "gpu_model": "NVIDIA GeForce RTX 5090",
                "gpu_vram_gb": 32,
                "cpu_model": "Intel Core i9-14900K",
                "cpu_cores": 24,
                "cpu_frequency": 5200,
                "ram_total_gb": 32,
                "ram_available_gb": 24
            },
            "game": {
                "game_name": "Cyberpunk 2077",
                "minimum_gpu": "GTX 1060",
                "recommended_gpu": "RTX 3070"
            },
            "expected_tier": "S",
            "expected_fps_min": 80
        },
        {
            "name": "RTX 4070 High-End",
            "hardware": {
                "gpu_model": "NVIDIA GeForce RTX 4070",
                "gpu_vram_gb": 12,
                "cpu_model": "AMD Ryzen 7 7700X",
                "cpu_cores": 8,
                "cpu_frequency": 4500,
                "ram_total_gb": 32,
                "ram_available_gb": 24
            },
            "game": {
                "game_name": "Counter-Strike 2",
                "minimum_gpu": "GTX 1060",
                "recommended_gpu": "RTX 3060"
            },
            "expected_tier": "A",
            "expected_fps_min": 120
        },
        {
            "name": "GTX 1650 Budget",
            "hardware": {
                "gpu_model": "NVIDIA GeForce GTX 1650",
                "gpu_vram_gb": 4,
                "cpu_model": "Intel Core i5-10400F",
                "cpu_cores": 6,
                "cpu_frequency": 4300,
                "ram_total_gb": 16,
                "ram_available_gb": 12
            },
            "game": {
                "game_name": "Counter-Strike 2",
                "minimum_gpu": "GTX 1050",
                "recommended_gpu": "GTX 1060"
            },
            "expected_tier": "C",
            "expected_fps_min": 45
        }
    ]
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Run assessment
            assessment = predictor.assess_performance(
                hardware_specs=test_case['hardware'],
                game_requirements=test_case['game']
            )
            
            # Validate results
            tier_correct = assessment.tier.name == test_case['expected_tier']
            fps_adequate = assessment.expected_fps >= test_case['expected_fps_min']
            
            print(f"GPU: {test_case['hardware']['gpu_model']}")
            print(f"Game: {test_case['game']['game_name']}")
            print(f"Tier: {assessment.tier.name} (Expected: {test_case['expected_tier']}) {'PASS' if tier_correct else 'FAIL'}")
            print(f"FPS: {assessment.expected_fps} (Min Expected: {test_case['expected_fps_min']}) {'PASS' if fps_adequate else 'FAIL'}")
            print(f"Settings: {assessment.recommended_settings}")
            print(f"Resolution: {assessment.recommended_resolution}")
            
            if assessment.bottlenecks:
                print(f"Bottlenecks: {', '.join(assessment.bottlenecks)}")
            
            results.append({
                'name': test_case['name'],
                'tier_correct': tier_correct,
                'fps_adequate': fps_adequate,
                'tier': assessment.tier.name,
                'fps': assessment.expected_fps
            })
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'name': test_case['name'],
                'tier_correct': False,
                'fps_adequate': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r.get('tier_correct', False) and r.get('fps_adequate', False))
    total = len(results)
    
    for result in results:
        status = "PASS" if result.get('tier_correct', False) and result.get('fps_adequate', False) else "FAIL"
        print(f"{result['name']}: {status}")
        if 'error' not in result:
            print(f"  -> Tier: {result.get('tier', 'N/A')}, FPS: {result.get('fps', 'N/A')}")
    
    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("*** ALL TESTS PASSED - Streamlined predictor working correctly! ***")
    else:
        print("*** Some tests failed - Check implementation ***")
    
    return passed == total


if __name__ == "__main__":
    success = test_streamlined_predictor()
    sys.exit(0 if success else 1)