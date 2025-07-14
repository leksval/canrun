#!/usr/bin/env python3
"""Debug performance prediction issue."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from performance_predictor import PerformancePredictor
from compatibility_analyzer import CompatibilityAnalysis

def test_performance_prediction():
    predictor = PerformancePredictor()
    
    # Create a dummy compatibility analysis
    class DummyAnalysis:
        def __init__(self):
            self.game_name = "Diablo IV"
            self.overall_score = 0.85
    
    analysis = DummyAnalysis()
    
    # Test the predict_performance method
    try:
        result = predictor.predict_performance(analysis, "RTX 4090", 24, True, True)
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        if hasattr(result, 'predictions'):
            print(f"Predictions: {result.predictions}")
            if result.predictions:
                print(f"First prediction: {result.predictions[0]}")
                if 'fps' in result.predictions[0]:
                    print(f"Expected FPS: {result.predictions[0]['fps']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_performance_prediction()