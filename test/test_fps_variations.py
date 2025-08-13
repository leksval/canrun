#!/usr/bin/env python3
"""
Test script to verify FPS predictions vary by game
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from canrun.src.ml_model_loader import LightweightMLPredictor

def test_fps_variations():
    """Test that different games produce different FPS predictions"""
    print("=== Testing FPS Variations by Game ===\n")
    
    # Initialize ML predictor
    predictor = LightweightMLPredictor()
    
    # Test configuration
    test_gpu = "RTX 4070"
    test_passmark = 26925
    test_resolution = "1080p"
    test_memory = 12
    
    # Different games to test
    test_games = [
        "Cyberpunk 2077",
        "Fortnite", 
        "Counter-Strike 2",
        "Valorant",
        "Call of Duty Warzone",
        "Apex Legends"
    ]
    
    print(f"GPU: {test_gpu} (PassMark: {test_passmark})")
    print(f"Resolution: {test_resolution}")
    print(f"VRAM: {test_memory}GB\n")
    
    results = {}
    
    # Test each game
    for game in test_games:
        fps = predictor.predict_fps(
            gpu_name=test_gpu,
            passmark_score=test_passmark,
            resolution=test_resolution,
            game_name=game,
            memory_gb=test_memory
        )
        
        results[game] = fps
        print(f"{game:20} -> {fps} FPS")
    
    # Check for variations
    unique_fps = set(results.values())
    print(f"\n=== Analysis ===")
    print(f"Total games tested: {len(test_games)}")
    print(f"Unique FPS values: {len(unique_fps)}")
    print(f"FPS range: {min(results.values())} - {max(results.values())}")
    
    if len(unique_fps) == 1:
        print("❌ ISSUE: All games return the same FPS!")
        return False
    else:
        print("[SUCCESS] Games return different FPS values!")
        return True

def test_with_steam_requirements():
    """Test with mock Steam requirements"""
    print("\n=== Testing with Steam Requirements ===\n")
    
    predictor = LightweightMLPredictor()
    
    # Mock Steam requirements for different games
    cyberpunk_reqs = {
        'minimum_gpu': 'GTX 1060',
        'recommended_gpu': 'RTX 3070',
        'minimum_vram_gb': 6,
        'recommended_vram_gb': 8
    }
    
    fortnite_reqs = {
        'minimum_gpu': 'GTX 660',
        'recommended_gpu': 'GTX 1660',
        'minimum_vram_gb': 2,
        'recommended_vram_gb': 4
    }
    
    test_gpu = "RTX 4070"
    test_passmark = 26925
    
    # Test Cyberpunk 2077
    cyberpunk_fps = predictor.predict_fps(
        gpu_name=test_gpu,
        passmark_score=test_passmark,
        resolution="1080p",
        game_name="Cyberpunk 2077",
        memory_gb=12,
        game_requirements=cyberpunk_reqs
    )
    
    # Test Fortnite
    fortnite_fps = predictor.predict_fps(
        gpu_name=test_gpu,
        passmark_score=test_passmark,
        resolution="1080p", 
        game_name="Fortnite",
        memory_gb=12,
        game_requirements=fortnite_reqs
    )
    
    print(f"Cyberpunk 2077 (demanding): {cyberpunk_fps} FPS")
    print(f"Fortnite (optimized):       {fortnite_fps} FPS")
    
    if cyberpunk_fps != fortnite_fps:
        print("✅ SUCCESS: Steam requirements produce different FPS!")
        return True
    else:
        print("❌ ISSUE: Steam requirements return same FPS!")
        return False

if __name__ == "__main__":
    success1 = test_fps_variations()
    success2 = test_with_steam_requirements()
    
    if success1 and success2:
        print("\n🎉 All tests passed! FPS predictions now vary by game.")
    else:
        print("\n❌ Some tests failed. FPS prediction issues remain.")