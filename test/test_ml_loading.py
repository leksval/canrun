"""Test ML model loading and FPS prediction"""
import json
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from canrun_model_loader import LightweightMLPredictor
from canrun_ml_predictor import CanRunMLPredictor

def test_ml_loading():
    """Test if ML model loads correctly"""
    print("Testing ML Model Loading...")
    print("=" * 50)
    
    # Test 1: Check if ML model file exists
    model_paths = [
        Path("data/ml_fps_model_unified.json"),
        Path("canrun/data/ml_fps_model_unified.json"),
        Path(__file__).parent.parent / "data" / "ml_fps_model_unified.json"
    ]
    
    for path in model_paths:
        if path.exists():
            print(f"[OK] Found ML model at: {path}")
            with open(path, 'r') as f:
                data = json.load(f)
                print(f"    - Lookup table entries: {len(data.get('lookup_table', {}))}")
                print(f"    - Model version: {data.get('version', 'unknown')}")
            break
    else:
        print("[ERROR] ML model file not found!")
    
    # Test 2: Check LightweightMLPredictor
    print("\nTesting LightweightMLPredictor...")
    predictor = LightweightMLPredictor()
    print(f"  - Is loaded: {predictor.is_loaded}")
    if predictor.is_loaded:
        info = predictor.get_model_info()
        print(f"  - Status: {info['status']}")
        print(f"  - Lookup table size: {info['lookup_table_size']}")
    
    # Test 3: Check CanRunMLPredictor
    print("\nTesting CanRunMLPredictor...")
    ml_predictor = CanRunMLPredictor()
    print(f"  - ML predictor initialized: {ml_predictor.ml_predictor is not None}")
    if ml_predictor.ml_predictor:
        print(f"  - ML predictor loaded: {ml_predictor.ml_predictor.is_loaded}")
    
    # Test 4: Try actual FPS prediction
    print("\nTesting FPS Prediction for RTX 4090...")
    hardware_specs = {
        'gpu_model': 'RTX 4090',
        'gpu_vram_gb': 24,
        'cpu_model': 'AMD Ryzen 7 7800X3D',
        'cpu_cores': 8,
        'ram_total_gb': 32,
        'display_width': 1920,
        'display_height': 1080
    }
    
    game_requirements = {
        'game_name': 'Diablo 4',
        'minimum_gpu': 'GTX 660',
        'recommended_gpu': 'GTX 970',
        'minimum_ram_gb': 8,
        'recommended_ram_gb': 16,
        'minimum_vram_gb': 4,
        'recommended_vram_gb': 4
    }
    
    assessment = ml_predictor.assess_performance(hardware_specs, game_requirements)
    print(f"  - Score: {assessment.score}")
    print(f"  - Tier: {assessment.tier.name}")
    print(f"  - Expected FPS: {assessment.expected_fps}")
    print(f"  - FPS source: {assessment.fps_source}")

    from canrun_steam_fps import resolve_fps
    steam_estimate = resolve_fps(
        "RTX 4090",
        "AMD Ryzen 7 7800X3D",
        {"response": {"frame_rates": [{
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "cpu_name": "AMD Ryzen 7 7800X3D",
            "report_count": 120,
            "mean_frame_rate": 144,
            "mean_frame_rate_stddev": 8,
        }]}},
        cluster_map={"clusters": {}, "reference_gpu": "RTX 5070"},
    )
    print(f"  - Steam FPS: {steam_estimate.expected_fps}")

    assert predictor.is_loaded, "ML model should be loaded"
    assert ml_predictor.ml_predictor is not None, "ML predictor should be initialized"
    assert assessment.expected_fps > 0, "A known recommended GPU still produces an estimate"
    assert assessment.fps_source == "requirements"
    assert steam_estimate.expected_fps == 144
    assert steam_estimate.fps_source == "steam"

if __name__ == "__main__":
    test_ml_loading()
    print("\nAll tests passed!")