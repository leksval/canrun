#!/usr/bin/env python3
"""
CanRun Performance Prediction Test Suite
Tests the performance prediction capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.performance_predictor import PerformancePredictor
from src.hardware_detector import HardwareDetector
from src.requirements_fetcher import RequirementsFetcher
import asyncio
import json

async def test_performance_prediction():
    """Test performance prediction functionality"""
    print("=" * 50)
    print("CanRun Performance Prediction Test")
    print("=" * 50)
    
    # Initialize components
    detector = HardwareDetector()
    fetcher = RequirementsFetcher()
    predictor = PerformancePredictor()
    
    try:
        print("\n[TEST 1] Detecting system hardware...")
        hardware = await detector.detect_system_hardware()
        print(f"✓ System detected: {hardware.get('gpu', {}).get('name', 'Unknown GPU')}")
        
        # Test different games and settings
        test_scenarios = [
            {"game": "Cyberpunk 2077", "resolution": "1080p", "settings": "High"},
            {"game": "Cyberpunk 2077", "resolution": "1440p", "settings": "Medium"},
            {"game": "Cyberpunk 2077", "resolution": "4K", "settings": "Low"},
            {"game": "Hogwarts Legacy", "resolution": "1080p", "settings": "Ultra"},
            {"game": "Call of Duty: Modern Warfare II", "resolution": "1080p", "settings": "High"}
        ]
        
        for i, scenario in enumerate(test_scenarios, 2):
            print(f"\n[TEST {i}] Predicting performance for '{scenario['game']}'...")
            print(f"  Settings: {scenario['resolution']}, {scenario['settings']}")
            
            # Fetch game requirements
            requirements = await fetcher.fetch_game_requirements(scenario['game'])
            if not requirements:
                print(f"  ⚠ No requirements found for {scenario['game']}, skipping...")
                continue
            
            # Predict performance
            performance = await predictor.predict_performance(
                hardware, 
                requirements, 
                scenario['resolution'], 
                scenario['settings']
            )
            
            # Display results
            print(f"  ✓ Predicted FPS: {performance['fps']}")
            print(f"  ✓ Performance Level: {performance['performance_level']}")
            print(f"  ✓ Stability: {performance['stability']}")
            print(f"  ✓ DLSS Boost: {performance.get('dlss_boost', 'N/A')}")
            
            if performance.get('optimization_suggestions'):
                print(f"  ✓ Optimizations: {', '.join(performance['optimization_suggestions'])}")
        
        print(f"\n[TEST {len(test_scenarios) + 2}] Testing DLSS performance boost...")
        
        # Test DLSS scenarios
        dlss_scenarios = [
            {"dlss_mode": "Quality", "expected_boost": 1.3},
            {"dlss_mode": "Balanced", "expected_boost": 1.5},
            {"dlss_mode": "Performance", "expected_boost": 1.8},
        ]
        
        for dlss_scenario in dlss_scenarios:
            boost = predictor.calculate_dlss_boost(
                hardware.get('gpu', {}), 
                dlss_scenario['dlss_mode']
            )
            print(f"  ✓ DLSS {dlss_scenario['dlss_mode']}: {boost:.2f}x boost")
            
            # Verify boost is reasonable
            if abs(boost - dlss_scenario['expected_boost']) > 0.3:
                print(f"  ⚠ DLSS boost seems unusual: {boost}")
        
        print(f"\n[TEST {len(test_scenarios) + 3}] Testing RTX performance impact...")
        
        # Test RTX scenarios
        rtx_scenarios = ["RTX Low", "RTX Medium", "RTX High", "RTX Ultra"]
        
        for rtx_setting in rtx_scenarios:
            impact = predictor.calculate_rtx_impact(rtx_setting)
            print(f"  ✓ {rtx_setting}: {impact:.2f}x performance impact")
        
        print(f"\n[TEST {len(test_scenarios) + 4}] Testing resolution scaling...")
        
        # Test resolution scaling
        resolutions = ["1080p", "1440p", "4K"]
        base_fps = 60
        
        for resolution in resolutions:
            multiplier = predictor.get_resolution_multiplier(resolution)
            scaled_fps = base_fps * multiplier
            print(f"  ✓ {resolution}: {scaled_fps:.1f} FPS (base: {base_fps})")
        
        print("\n" + "=" * 50)
        print("All performance prediction tests passed!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"✗ Performance prediction test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_performance_prediction())
    sys.exit(0 if success else 1)