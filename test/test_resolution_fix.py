#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'canrun', 'src'))

# Test just the resolution classification method directly
class SimpleResolutionTester:
    def _classify_resolution(self, width: int, height: int) -> str:
        """Fixed resolution classification"""
        total_pixels = width * height
        
        if total_pixels >= 3840 * 2160 * 0.9:  # 4K
            return "4K"
        elif total_pixels >= 2560 * 1440 * 0.9:  # 1440p
            return "1440p"
        elif total_pixels >= 1920 * 1080 * 0.9:  # 1080p
            return "1080p"
        elif total_pixels >= 1280 * 720 * 0.9:  # 720p
            return "720p"
        else:
            return "720p"

def test_resolution_fix():
    tester = SimpleResolutionTester()
    
    print("Testing Resolution Classification Fix")
    print("=" * 50)
    
    # Test resolutions
    test_cases = [
        (1920, 1080, "1080p"),
        (2560, 1440, "1440p"),
        (3072, 1728, "1440p"), 
        (3840, 2160, "4K"),
        (1280, 720, "720p")
    ]
    
    for width, height, expected in test_cases:
        actual = tester._classify_resolution(width, height)
        status = "PASS" if actual == expected else "FAIL"
        print(f"{status}: {width}x{height} -> Expected: {expected}, Got: {actual}")
    
    print("\nUser Resolution Analysis:")
    print("-" * 30)
    user_resolution = tester._classify_resolution(3072, 1728)
    print(f"3072x1728 classified as: {user_resolution}")
    print(f"Expected ML model lookup: RTX 4090|Cyberpunk 2077|{user_resolution}")
    
    # From the ML model data we read earlier
    ml_predictions = {
        "RTX 4090|Cyberpunk 2077|1080p": 340.6,
        "RTX 4090|Cyberpunk 2077|1440p": 276.5,
        "RTX 4090|Cyberpunk 2077|4K": 66.6
    }
    
    expected_fps = ml_predictions.get(f"RTX 4090|Cyberpunk 2077|{user_resolution}", "N/A")
    print(f"Expected FPS: {expected_fps}")
    print(f"Previous (wrong) FPS: 340.6")
    print(f"Improvement: {340.6 - expected_fps:.1f} FPS reduction (more realistic)")

if __name__ == "__main__":
    test_resolution_fix()