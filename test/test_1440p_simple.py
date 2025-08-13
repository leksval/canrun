#!/usr/bin/env python3

"""
Simple 1440p Resolution Classification Test
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from canrun.src.display_detector import DisplayDetector
from canrun.src.dynamic_performance_predictor import StreamlinedPerformancePredictor


def test_1440p_classification():
    """Test 1440p resolution classification"""
    
    print("=" * 60)
    print("1440P RESOLUTION CLASSIFICATION TEST")
    print("=" * 60)
    
    detector = DisplayDetector()
    predictor = StreamlinedPerformancePredictor()
    
    # Key test cases
    test_cases = [
        (2560, 1440, "Standard 1440p"),
        (3440, 1440, "Ultrawide 1440p (21:9)"),
        (3072, 1728, "User's problematic resolution"),
        (3840, 1600, "Ultrawide 1600p"),
        (2880, 1620, "Scaled 1440p"),
    ]
    
    print("\nCLASSIFICATION RESULTS:")
    print(f"{'Resolution':<15} {'Detector':<10} {'Predictor':<10} {'Match':<8} {'Pixels'}")
    print("-" * 60)
    
    issues = []
    
    for width, height, description in test_cases:
        detector_result = detector.get_resolution_tier(width, height)
        predictor_result = predictor._classify_resolution(width, height)
        
        match = "YES" if detector_result == predictor_result else "NO"
        pixels = width * height
        
        print(f"{width}x{height:<8} {detector_result:<10} {predictor_result:<10} {match:<8} {pixels:,}")
        
        # Check for 1440p classification issues
        if "1440p" in description and detector_result != "1440p":
            issues.append(f"{description}: classified as {detector_result} instead of 1440p")
        
        if match == "NO":
            issues.append(f"{description}: methods disagree - Detector:{detector_result} vs Predictor:{predictor_result}")
    
    # Calculate thresholds
    print("\nTHRESHOLD ANALYSIS:")
    standard_1440p = 2560 * 1440
    threshold_90 = standard_1440p * 0.9
    ultrawide_threshold = standard_1440p * 1.4
    
    print(f"Standard 1440p: {standard_1440p:,} pixels")
    print(f"90% threshold: {threshold_90:,} pixels")
    print(f"Ultrawide threshold (1.4x): {ultrawide_threshold:,} pixels")
    
    # Check specific problematic case
    print(f"\nUSER'S CASE ANALYSIS:")
    user_pixels = 3072 * 1728
    print(f"3072x1728 = {user_pixels:,} pixels")
    print(f"vs 1440p threshold: {user_pixels / standard_1440p:.2f}x")
    print(f"vs ultrawide threshold: {user_pixels / ultrawide_threshold:.2f}x")
    
    if user_pixels >= ultrawide_threshold:
        print("ISSUE: User's resolution exceeds ultrawide threshold - classified as 4K")
    else:
        print("OK: User's resolution below ultrawide threshold - should be 1440p")
    
    # Summary
    print("\nSUMMARY:")
    if issues:
        print("ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("No classification issues found")
    
    return issues


if __name__ == "__main__":
    test_1440p_classification()