#!/usr/bin/env python3
"""
CanRun Sample Queries and Usage Examples
Demonstrates how to use the CanRun system with various queries
"""

import sys
import os
sys.path.append('..')

from canrun_engine import CanRunEngine
import asyncio
import json

class CanRunSampleQueries:
    def __init__(self):
        self.engine = CanRunEngine(enable_llm=False)
    
    async def initialize(self):
        """Initialize the CanRun engine"""
        await self.engine.initialize()
        print("CanRun engine initialized successfully!")
    
    async def sample_basic_compatibility_check(self):
        """Sample: Basic game compatibility check"""
        print("\n" + "="*60)
        print("SAMPLE: Basic Game Compatibility Check")
        print("="*60)
        
        game = "Cyberpunk 2077"
        print(f"Checking compatibility for '{game}'...")
        
        result = await self.engine.analyze_game_compatibility(game, "High", "1080p")
        
        if result:
            compatibility = result['compatibility']
            print(f"OK Compatibility Level: {compatibility['compatibility_level']}")
            print(f"OK Overall Score: {compatibility['overall_score']}/100")
            
            if compatibility['bottlenecks']:
                print(f"⚠ Bottlenecks: {', '.join(compatibility['bottlenecks'])}")
            else:
                print("OK No bottlenecks detected")
        else:
            print("ERROR Failed to analyze compatibility")
    
    async def sample_performance_prediction(self):
        """Sample: Performance prediction with different settings"""
        print("\n" + "="*60)
        print("SAMPLE: Performance Prediction")
        print("="*60)
        
        game = "Hogwarts Legacy"
        settings_tests = [
            ("1080p", "Ultra"),
            ("1440p", "High"),
            ("4K", "Medium")
        ]
        
        print(f"Testing performance for '{game}' with different settings:")
        
        for resolution, settings in settings_tests:
            result = await self.engine.analyze_game_compatibility(game, settings, resolution)
            
            if result:
                performance = result['performance']
                print(f"  {resolution} {settings}: {performance['fps']} FPS ({performance['performance_level']})")
            else:
                print(f"  {resolution} {settings}: Analysis failed")
    
    async def sample_optimization_suggestions(self):
        """Sample: Get optimization suggestions"""
        print("\n" + "="*60)
        print("SAMPLE: Optimization Suggestions")
        print("="*60)
        
        game = "Call of Duty: Modern Warfare II"
        print(f"Getting optimization suggestions for '{game}'...")
        
        optimizations = await self.engine.get_optimization_suggestions(game, "Ultra", "4K")
        
        if optimizations:
            print(f"Found {len(optimizations)} optimization suggestions:")
            for i, opt in enumerate(optimizations[:5], 1):  # Show first 5
                print(f"  {i}. {opt['type']}: {opt['description']}")
                if 'expected_improvement' in opt:
                    print(f"     Expected improvement: {opt['expected_improvement']}")
        else:
            print("No optimization suggestions available")
    
    async def sample_batch_analysis(self):
        """Sample: Batch analysis of multiple games"""
        print("\n" + "="*60)
        print("SAMPLE: Batch Game Analysis")
        print("="*60)
        
        games = ["Cyberpunk 2077", "Hogwarts Legacy", "Valorant", "Apex Legends"]
        print(f"Analyzing {len(games)} games in batch...")
        
        results = await self.engine.analyze_multiple_games(games)
        
        print("Results:")
        for game, result in results.items():
            if result:
                compatibility = result['compatibility']['compatibility_level']
                fps = result['performance']['fps']
                print(f"  OK {game}: {compatibility} ({fps} FPS)")
            else:
                print(f"  ERROR {game}: Analysis failed")
    
    async def sample_system_information(self):
        """Sample: Get system information"""
        print("\n" + "="*60)
        print("SAMPLE: System Information")
        print("="*60)
        
        system_info = await self.engine.get_system_info()
        
        print("System Hardware:")
        print(f"  CPU: {system_info['cpu']['name']}")
        print(f"  GPU: {system_info['gpu']['name']}")
        print(f"  Memory: {system_info['memory']['total']} GB")
        print(f"  Storage: {system_info['storage']['total']} GB")
        
        # Show performance scores
        if 'performance_scores' in system_info:
            scores = system_info['performance_scores']
            print(f"\nPerformance Scores:")
            print(f"  CPU Score: {scores.get('cpu', 'N/A')}")
            print(f"  GPU Score: {scores.get('gpu', 'N/A')}")
    
    async def sample_advanced_queries(self):
        """Sample: Advanced queries with specific parameters"""
        print("\n" + "="*60)
        print("SAMPLE: Advanced Queries")
        print("="*60)
        
        # Test RTX features
        print("Testing RTX-enabled games:")
        rtx_games = ["Cyberpunk 2077", "Control", "Metro Exodus"]
        
        for game in rtx_games:
            result = await self.engine.analyze_game_compatibility(game, "RTX High", "1440p")
            if result:
                performance = result['performance']
                print(f"  {game}: {performance['fps']} FPS with RTX")
                if 'rtx_features' in performance:
                    print(f"    RTX Features: {', '.join(performance['rtx_features'])}")
        
        # Test DLSS performance
        print("\nTesting DLSS performance:")
        dlss_settings = ["DLSS Quality", "DLSS Balanced", "DLSS Performance"]
        
        for setting in dlss_settings:
            result = await self.engine.analyze_game_compatibility("Cyberpunk 2077", setting, "4K")
            if result:
                performance = result['performance']
                print(f"  {setting}: {performance['fps']} FPS")
    
    async def sample_json_output(self):
        """Sample: JSON output for API usage"""
        print("\n" + "="*60)
        print("SAMPLE: JSON Output (API Usage)")
        print("="*60)
        
        result = await self.engine.analyze_game_compatibility("Cyberpunk 2077", "High", "1080p")
        
        if result:
            print("JSON Response:")
            print(json.dumps(result, indent=2))
        else:
            print("No result to display")

async def main():
    """Run all sample queries"""
    print("CanRun Sample Queries and Usage Examples")
    print("="*60)
    
    samples = CanRunSampleQueries()
    await samples.initialize()
    
    # Run sample queries
    await samples.sample_basic_compatibility_check()
    await samples.sample_performance_prediction()
    await samples.sample_optimization_suggestions()
    await samples.sample_batch_analysis()
    await samples.sample_system_information()
    await samples.sample_advanced_queries()
    await samples.sample_json_output()
    
    print("\n" + "="*60)
    print("All sample queries completed!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())