"""
Test to demonstrate and fix the ML FPS prediction bug
that's causing identical FPS predictions for all games
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "canrun" / "src"))

from ml_model_loader import LightweightMLPredictor

def test_fps_prediction_variance():
    """Test that different games produce different FPS predictions"""
    
    predictor = LightweightMLPredictor()
    
    # Debug: Check if GPU hierarchy loaded
    print(f"GPU hierarchy loaded: {len(predictor.gpu_hierarchy)} GPUs")
    print(f"ML model loaded: {predictor.is_loaded}")
    
    # Test with RTX 4090 specs
    gpu_name = "RTX 4090"
    passmark_score = 35000
    resolution = "1440p"
    memory_gb = 24
    
    # Different games with VERY different requirements
    games_and_requirements = {
        "Diablo 4": {
            "game_name": "Diablo 4",
            "minimum_gpu": "GTX 1060 3GB",
            "recommended_gpu": "RTX 2080 Ti",
            "minimum_vram_gb": 4,
            "recommended_vram_gb": 8
        },
        "Cyberpunk 2077": {
            "game_name": "Cyberpunk 2077",
            "minimum_gpu": "GTX 1060 6GB",
            "recommended_gpu": "RTX 3070",
            "minimum_vram_gb": 6,
            "recommended_vram_gb": 12
        },
        "Elden Ring": {
            "game_name": "Elden Ring",
            "minimum_gpu": "GTX 1060 3GB",
            "recommended_gpu": "GTX 1070",
            "minimum_vram_gb": 3,
            "recommended_vram_gb": 8
        }
    }
    
    fps_results = {}
    
    for game_name, requirements in games_and_requirements.items():
        print(f"\n--- Testing {game_name} ---")
        
        # Test requirement GPU scores
        min_gpu = requirements.get('minimum_gpu', '')
        rec_gpu = requirements.get('recommended_gpu', '')
        min_score = predictor._get_gpu_passmark_score(min_gpu)
        rec_score = predictor._get_gpu_passmark_score(rec_gpu)
        
        print(f"Min GPU: {min_gpu} -> {min_score} PassMark")
        print(f"Rec GPU: {rec_gpu} -> {rec_score} PassMark")
        print(f"User GPU: {gpu_name} -> {passmark_score} PassMark")
        
        fps = predictor.predict_fps(
            gpu_name=gpu_name,
            passmark_score=passmark_score,
            resolution=resolution,
            game_name=game_name,
            memory_gb=memory_gb,
            game_requirements=requirements
        )
        fps_results[game_name] = fps
        print(f"Final FPS: {fps}")
    
    # Check that we get different FPS values
    unique_fps_values = set(fps_results.values())
    
    print(f"\nUnique FPS values: {unique_fps_values}")
    print(f"All FPS results: {fps_results}")
    
    # This should pass if the bug is fixed
    assert len(unique_fps_values) > 1, f"All games returned same FPS: {unique_fps_values}"
    
    # More specific checks
    cyberpunk_fps = fps_results["Cyberpunk 2077"]
    elden_ring_fps = fps_results["Elden Ring"]
    
    # Cyberpunk should be more demanding (lower FPS) than Elden Ring
    assert cyberpunk_fps < elden_ring_fps, f"Cyberpunk ({cyberpunk_fps}) should be more demanding than Elden Ring ({elden_ring_fps})"

if __name__ == "__main__":
    test_fps_prediction_variance()