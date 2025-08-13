#!/usr/bin/env python3
"""Test 4K ultrawide resolution classification fix"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'canrun', 'src'))

from dynamic_performance_predictor import DynamicPerformancePredictor

def test_4k_ultrawide_classification():
    """Test that 3072x1728 is correctly classified as 4K performance tier"""
    predictor = DynamicPerformancePredictor()
    
    # Test user's actual resolution
    user_resolution = predictor._classify_resolution(3072, 1728)
    total_pixels = 3072 * 1728
    
    print(f"Resolution: 3072x1728")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Classification: {user_resolution}")
    
    # Compare with standard resolutions
    resolutions = [
        (1920, 1080, "1080p"),
        (2560, 1440, "1440p"), 
        (3072, 1728, "4K (ultrawide)"),
        (3840, 2160, "4K")
    ]
    
    print(f"\nResolution comparison:")
    for width, height, name in resolutions:
        pixels = width * height
        classification = predictor._classify_resolution(width, height)
        print(f"{name}: {width}x{height} = {pixels:,} pixels -> {classification}")
    
    # Verify user's resolution is classified as 4K
    assert user_resolution == "4K", f"Expected 4K, got {user_resolution}"
    print(f"\n✓ 3072x1728 correctly classified as 4K performance tier")

if __name__ == "__main__":
    test_4k_ultrawide_classification()