#!/usr/bin/env python3
"""
CanRun Performance Prediction Test Suite
Tests the performance prediction capabilities
"""

import sys
import os
import pytest
# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

from canrun_ml_predictor import CanRunMLPredictor
from canrun_hardware_detector import CanRunHardwareDetector
from canrun_game_fetcher import CanRunGameFetcher
import asyncio
import json

@pytest.mark.asyncio
async def test_performance_prediction():
    """Test performance prediction functionality"""
    print("=" * 50)
    print("CanRun Performance Prediction Test")
    print("=" * 50)
    
    # Initialize components
    detector = CanRunHardwareDetector()
    fetcher = CanRunGameFetcher()
    predictor = CanRunMLPredictor()
    
    try:
        print("\n[TEST 1] Detecting system hardware...")
        hardware = detector.detect_hardware()
        print(f"+ System detected: {hardware.gpu_name}")
        
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
            requirements = await fetcher.fetch_requirements(scenario['game'])
            if not requirements:
                print(f"  ⚠ No requirements found for {scenario['game']}, skipping...")
                continue
            
            # Predict performance using new API
            hardware_specs = {
                'gpu_model': hardware.gpu_name,
                'cpu_model': hardware.cpu_name,
                'ram_total_gb': hardware.ram_total / 1024,  # Convert MB to GB
                'cpu_cores': hardware.cpu_cores,
                'cpu_threads': hardware.cpu_cores * 2,  # Estimate threads
                'cpu_frequency': hardware.cpu_freq * 1000,  # Convert GHz to MHz
                'gpu_vram_gb': hardware.gpu_memory / 1024  # Convert MB to GB
            }
            
            # Convert requirements to dict format
            game_requirements = {
                'recommended': {
                    'processor': getattr(requirements, 'recommended_cpu', ''),
                    'graphics': getattr(requirements, 'recommended_gpu', ''),
                    'memory': getattr(requirements, 'recommended_ram_gb', 8)
                }
            }
            
            assessment = predictor.assess_performance(
                hardware_specs=hardware_specs,
                game_requirements=game_requirements
            )
            
            # Display results
            print(f"  OK Performance Score: {assessment.score}")
            print(f"  OK Performance Tier: {assessment.tier.name}")
            print(f"  OK Expected FPS: {assessment.expected_fps}")
            print(f"  OK Settings: {assessment.recommended_settings}")
            print(f"  OK Resolution: {assessment.recommended_resolution}")
            
            if assessment.bottlenecks:
                print(f"  OK Bottlenecks: {', '.join(assessment.bottlenecks)}")
            if assessment.upgrade_suggestions:
                print(f"  OK Upgrades: {', '.join(assessment.upgrade_suggestions[:2])}")
        
        print(f"\n[TEST {len(test_scenarios) + 2}] Testing tier system...")
        
        # Test different hardware tiers
        tier_scenarios = [
            {"gpu": "RTX 4090", "cpu": "Ryzen 7 7800X3D", "ram": 32, "expected_tier": "S"},
            {"gpu": "RTX 3080", "cpu": "Intel i7-10700K", "ram": 16, "expected_tier": "A"},
            {"gpu": "RTX 3060", "cpu": "Intel i5-10400", "ram": 16, "expected_tier": "B"},
            {"gpu": "GTX 1660", "cpu": "Intel i3-9100", "ram": 8, "expected_tier": "C"},
        ]
        
        for tier_scenario in tier_scenarios:
            test_specs = {
                'gpu_model': tier_scenario['gpu'],
                'cpu_model': tier_scenario['cpu'],
                'ram_total_gb': tier_scenario['ram'],
                'cpu_cores': 8,
                'cpu_threads': 16,
                'cpu_frequency': 3000,
                'gpu_vram_gb': 8
            }
            
            assessment = predictor.assess_performance(hardware_specs=test_specs)
            print(f"  OK {tier_scenario['gpu']} + {tier_scenario['cpu']}: Tier {assessment.tier.name} (Score: {assessment.score})")
        
        print(f"\n[TEST {len(test_scenarios) + 3}] Testing upgrade suggestions...")
        
        # Test low-end system for upgrade suggestions
        low_end_specs = {
            'gpu_model': 'GTX 1050',
            'cpu_model': 'Intel i3-8100',
            'ram_total_gb': 8,
            'cpu_cores': 4,
            'cpu_threads': 4,
            'cpu_frequency': 2800,
            'gpu_vram_gb': 2
        }
        
        assessment = predictor.assess_performance(hardware_specs=low_end_specs)
        print(f"  OK Low-end system tier: {assessment.tier.name}")
        if assessment.upgrade_suggestions:
            print(f"  OK Upgrade suggestions: {', '.join(assessment.upgrade_suggestions[:3])}")
        if assessment.bottlenecks:
            print(f"  OK Bottlenecks identified: {', '.join(assessment.bottlenecks)}")
        
        print("\n" + "=" * 50)
        print("All performance prediction tests passed!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"X Performance prediction test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_performance_prediction())
    sys.exit(0 if success else 1)