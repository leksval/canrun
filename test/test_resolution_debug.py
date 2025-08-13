#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'canrun', 'src'))

from display_detector import DisplayDetector

def test_resolution_detection():
    detector = DisplayDetector()
    
    # Test the user's actual resolution
    user_width, user_height = 3072, 1728
    tier = detector.get_resolution_tier(user_width, user_height)
    
    print(f"User Resolution: {user_width}x{user_height}")
    print(f"Total Pixels: {user_width * user_height:,}")
    print(f"Detected Tier: {tier}")
    print()
    
    # Test standard resolutions for comparison
    test_resolutions = [
        (1920, 1080, "1080p"),
        (2560, 1440, "1440p"), 
        (3072, 1728, "User's"),
        (3840, 2160, "4K")
    ]
    
    print("Resolution Analysis:")
    print("=" * 50)
    for width, height, name in test_resolutions:
        pixels = width * height
        tier = detector.get_resolution_tier(width, height)
        print(f"{name:8}: {width:4}x{height:4} = {pixels:8,} pixels → {tier}")
    
    print("\nThreshold Analysis:")
    print("=" * 50)
    print(f"1080p threshold: {int(1920 * 1080 * 0.9):,} pixels")
    print(f"1440p threshold: {int(2560 * 1440 * 0.9):,} pixels") 
    print(f"4K threshold:    {int(3840 * 2160 * 0.9):,} pixels")
    print(f"User pixels:     {user_width * user_height:,} pixels")

if __name__ == "__main__":
    test_resolution_detection()