#!/usr/bin/env python3
"""
CanRun Compatibility Analysis Test Suite
Tests the game compatibility analysis capabilities
"""

import sys
import os
import pytest
# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

from compatibility_analyzer import CompatibilityAnalyzer
from hardware_detector import HardwareDetector
from game_requirements_fetcher import GameRequirementsFetcher
import asyncio
import json

@pytest.mark.asyncio
async def test_compatibility_analysis():
    """Test compatibility analysis functionality"""
    print("=" * 50)
    print("CanRun Compatibility Analysis Test")
    print("=" * 50)
    
    # Initialize components
    detector = HardwareDetector()
    fetcher = GameRequirementsFetcher()
    analyzer = CompatibilityAnalyzer()
    
    try:
        # Test games to analyze
        test_games = [
            "Cyberpunk 2077",
            "Hogwarts Legacy",
            "Call of Duty: Modern Warfare II",
            "Red Dead Redemption 2"
        ]
        
        print("\n[TEST 1] Detecting system hardware...")
        hardware = detector.detect_hardware()
        print(f"+ System detected: {hardware.gpu_name}")
        
        for i, game in enumerate(test_games, 1):
            print(f"\n[TEST {i+1}] Analyzing compatibility for '{game}'...")
            
            # Fetch game requirements
            requirements = await fetcher.fetch_game_requirements(game)
            if not requirements:
                print(f"  ⚠ No requirements found for {game}, skipping...")
                continue
            
            # Analyze compatibility
            compatibility = await analyzer.analyze_compatibility(hardware, requirements)
            
            # Display results
            print(f"  OK Compatibility Level: {compatibility['compatibility_level']}")
            print(f"  OK Overall Score: {compatibility['overall_score']}/100")
            print(f"  OK Bottlenecks: {', '.join(compatibility['bottlenecks']) if compatibility['bottlenecks'] else 'None'}")
            
            # Show component analysis
            components = compatibility['component_analysis']
            print(f"  OK CPU: {components['cpu']['status']} (Score: {components['cpu']['score']})")
            print(f"  OK GPU: {components['gpu']['status']} (Score: {components['gpu']['score']})")
            print(f"  OK Memory: {components['memory']['status']} (Score: {components['memory']['score']})")
            print(f"  OK Storage: {components['storage']['status']} (Score: {components['storage']['score']})")
        
        print("\n[TEST 6] Testing edge cases...")
        
        # Test with minimal hardware
        minimal_hardware = {
            'cpu': {'name': 'Intel Core i3-8100', 'cores': 4, 'threads': 4, 'base_clock': 3.6},
            'gpu': {'name': 'NVIDIA GTX 1050', 'memory': 2, 'tier': 'Entry'},
            'memory': {'total': 8, 'available': 6},
            'storage': {'total': 500, 'available': 400, 'type': 'HDD'}
        }
        
        test_requirements = {
            'minimum': {
                'cpu': {'name': 'Intel Core i5-8400', 'cores': 6},
                'gpu': {'name': 'NVIDIA GTX 1060', 'memory': 6},
                'memory': {'total': 12},
                'storage': {'space': 70}
            },
            'recommended': {
                'cpu': {'name': 'Intel Core i7-9700K', 'cores': 8},
                'gpu': {'name': 'NVIDIA RTX 2070', 'memory': 8},
                'memory': {'total': 16},
                'storage': {'space': 70}
            }
        }
        
        edge_compatibility = await analyzer.analyze_compatibility(minimal_hardware, test_requirements)
        print(f"  OK Edge case compatibility: {edge_compatibility['compatibility_level']}")
        print(f"  OK Bottlenecks identified: {len(edge_compatibility['bottlenecks'])}")
        
        print("\n" + "=" * 50)
        print("All compatibility analysis tests passed!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"X Compatibility analysis test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_compatibility_analysis())
    sys.exit(0 if success else 1)