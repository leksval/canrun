#!/usr/bin/env python3
"""Test actual display detection vs expected results"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'canrun', 'src'))

from display_detector import DisplayDetector
from dynamic_performance_predictor import DynamicPerformancePredictor

def test_actual_display_detection():
    """Test what the display detector actually returns vs expectations"""
    
    # Test DisplayDetector directly
    print("=== DisplayDetector Test ===")
    try:
        detector = DisplayDetector()
        display = detector.get_display_resolution()
        print(f"DisplayDetector result: {display}")
        
        if 'width' in display and 'height' in display:
            width, height = display['width'], display['height']
            resolution_tier = detector.get_resolution_tier(width, height)
            print(f"Resolution: {width}x{height} -> {resolution_tier}")
        else:
            print("No width/height in display result")
            
    except Exception as e:
        print(f"DisplayDetector failed: {e}")
    
    # Test DynamicPerformancePredictor auto-detection
    print(f"\n=== DynamicPerformancePredictor Test ===")
    try:
        predictor = DynamicPerformancePredictor()
        hardware = predictor._detect_hardware()
        print(f"Auto-detected hardware: {hardware}")
        
        display_info = hardware.get('display', {})
        resolution_class = display_info.get('resolution_class', 'Unknown')
        print(f"Resolution class from predictor: {resolution_class}")
        
        # Test manual classification
        if 'width' in display and 'height' in display:
            manual_class = predictor._classify_resolution(display['width'], display['height'])
            print(f"Manual classification of {display['width']}x{display['height']}: {manual_class}")
            
    except Exception as e:
        print(f"DynamicPerformancePredictor failed: {e}")

if __name__ == "__main__":
    test_actual_display_detection()