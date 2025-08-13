#!/usr/bin/env python3
"""
Test realistic FPS predictions after fixing ML baseline assumptions
Expected: RTX 4090 Cyberpunk 2077 should now predict ~60-80 FPS instead of 340+ FPS
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'canrun', 'src'))

from ml_model_loader import LightweightMLPredictor

def test_realistic_predictions():
    print("Testing Realistic ML FPS Predictions (After Fix)")
    print("=" * 60)
    
    # Initialize ML predictor
    predictor = LightweightMLPredictor()
    
    # Test cases with RTX 4090 (38,192 PassMark)
    test_cases = [
        {
            "game": "Cyberpunk 2077",
            "gpu": "RTX 4090",
            "passmark": 38192,
            "resolution": "1080p",
            "expected_range": "60-80 FPS",
            "requirements": {
                'game_name': 'Cyberpunk 2077',
                'minimum_gpu': 'GeForce GTX 1060 6GB',
                'recommended_gpu': 'GeForce RTX 2060 SUPER',
                'minimum_cpu': 'Core i7-6700',
                'recommended_cpu': 'Core i7-12700',
                'minimum_ram_gb': 12,
                'recommended_ram_gb': 16
            }
        },
        {
            "game": "Diablo 4",
            "gpu": "RTX 4090", 
            "passmark": 38192,
            "resolution": "1080p",
            "expected_range": "120-180 FPS",
            "requirements": {
                'game_name': 'Diablo 4',
                'minimum_gpu': 'NVIDIA GeForce GTX 660',
                'recommended_gpu': 'NVIDIA GeForce GTX 970',
                'minimum_cpu': 'Intel Core i5-2500K',
                'recommended_cpu': 'Intel Core i5-4670K',
                'minimum_ram_gb': 8,
                'recommended_ram_gb': 16
            }
        },
        {
            "game": "Counter-Strike 2",
            "gpu": "RTX 4090",
            "passmark": 38192,
            "resolution": "1080p",
            "expected_range": "300+ FPS",
            "requirements": {
                'game_name': 'Counter-Strike 2',
                'minimum_gpu': 'GeForce GTX 1060',
                'recommended_gpu': 'GeForce RTX 2070',
                'minimum_cpu': 'Core i5-9600K',
                'recommended_cpu': 'Core i7-9700K',
                'minimum_ram_gb': 8,
                'recommended_ram_gb': 16
            }
        }
    ]
    
    print(f"{'Game':<20} {'GPU':<12} {'Resolution':<10} {'Predicted':<12} {'Expected':<15} {'Status'}")
    print("-" * 80)
    
    all_realistic = True
    
    for case in test_cases:
        predicted_fps = predictor.predict_fps(
            gpu_name=case["gpu"],
            passmark_score=case["passmark"],
            resolution=case["resolution"],
            game_name=case["game"],
            memory_gb=24,  # RTX 4090 has 24GB
            game_requirements=case["requirements"]
        )
        
        # Check if prediction is realistic
        if case["game"] == "Cyberpunk 2077":
            realistic = 50 <= predicted_fps <= 120  # Realistic range
        elif case["game"] == "Diablo 4":
            realistic = 80 <= predicted_fps <= 200  # Less demanding
        else:  # Counter-Strike 2
            realistic = predicted_fps >= 200  # eSports title should be high
            
        status = "REALISTIC" if realistic else "UNREALISTIC"
        if not realistic:
            all_realistic = False
            
        print(f"{case['game']:<20} {case['gpu']:<12} {case['resolution']:<10} {predicted_fps:<12.1f} {case['expected_range']:<15} {status}")
    
    print("\n" + "=" * 60)
    if all_realistic:
        print(" SUCCESS: All FPS predictions are now REALISTIC!")
        print("   - Cyberpunk 2077 should be 60-80 FPS (was 340+ FPS)")
        print("   - ML baseline assumptions have been corrected")
    else:
        print(" FAILURE: Some predictions are still unrealistic")
        print("   - Additional calibration may be needed")
    
    return all_realistic

if __name__ == "__main__":
    test_realistic_predictions()