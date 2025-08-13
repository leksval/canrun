#!/usr/bin/env python3
"""Test GPU detection specifically to debug ML prediction failures"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'canrun', 'src'))

from dynamic_performance_predictor import DynamicPerformancePredictor

def test_gpu_detection():
    """Test GPU detection to see why ML prediction fails"""
    
    predictor = DynamicPerformancePredictor()
    
    # Test auto-detected hardware
    print("=== Auto-Detected Hardware ===")
    hardware = predictor._detect_hardware()
    print(f"Full hardware: {hardware}")
    
    gpu_info = hardware.get('gpu', {})
    print(f"GPU model: '{gpu_info.get('model', '')}'")
    print(f"GPU VRAM: {gpu_info.get('vram', 0)}")
    
    # Test GPU score calculation
    print(f"\n=== GPU Score Calculation ===")
    gpu_score, gpu_passmark = predictor._get_gpu_score(gpu_info)
    print(f"GPU score: {gpu_score}")
    print(f"GPU PassMark: {gpu_passmark}")
    
    # Test ML prediction with current GPU
    print(f"\n=== ML Prediction Test ===")
    if predictor.ml_predictor and predictor.ml_predictor.is_loaded:
        print("ML predictor is loaded")
        
        # Try prediction with detected GPU
        try:
            ml_fps = predictor.ml_predictor.predict_fps(
                gpu_name=gpu_info.get('model', ''),
                passmark_score=gpu_passmark,
                resolution="4K",
                game_name="Cyberpunk 2077"
            )
            print(f"ML prediction result: {ml_fps}")
        except Exception as e:
            print(f"ML prediction failed: {e}")
            
        # Try prediction with explicit RTX 4090
        try:
            ml_fps_explicit = predictor.ml_predictor.predict_fps(
                gpu_name="RTX 4090",
                passmark_score=35000,
                resolution="4K", 
                game_name="Cyberpunk 2077"
            )
            print(f"ML prediction with RTX 4090: {ml_fps_explicit}")
        except Exception as e:
            print(f"ML prediction with RTX 4090 failed: {e}")
    else:
        print("ML predictor not loaded or not available")

if __name__ == "__main__":
    test_gpu_detection()