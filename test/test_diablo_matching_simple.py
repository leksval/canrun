#!/usr/bin/env python3
"""Simple test script to debug the Diablo 4 vs Diablo IV matching issue."""

import json
import asyncio
import sys
import os
from pathlib import Path

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

from game_requirements_fetcher import GameRequirementsFetcher, LocalCacheSource

async def test_diablo_matching():
    """Test the Diablo 4 matching logic."""
    
    # Test 1: Check if cache loads correctly
    cache_source = LocalCacheSource()
    print(f"Cache loaded: {len(cache_source._cache.get('games', {}))} games")
    
    available_games = cache_source._cache.get('games', {})
    print(f"Available games: {list(available_games.keys())}")
    
    # Test 2: Check if Diablo IV is in cache
    diablo_iv_in_cache = "Diablo IV" in available_games
    print(f"Diablo IV in cache: {diablo_iv_in_cache}")
    
    if diablo_iv_in_cache:
        print(f"Diablo IV requirements: {available_games['Diablo IV']}")
    
    # Test 3: Test LocalCacheSource fuzzy matching
    requirements = await cache_source.fetch("Diablo 4")
    print(f"LocalCacheSource fetch result: {requirements}")
    
    # Test 4: Test the full fetcher logic WITHOUT LLM analyzer
    fetcher = GameRequirementsFetcher(llm_analyzer=None)
    requirements = await fetcher.fetch_requirements("Diablo 4")
    
    if requirements:
        print(f"Final result - Game: {requirements.game_name}")
        print(f"Source: {requirements.source}")
        print(f"Minimum CPU: {requirements.minimum_cpu}")
        print(f"Minimum GPU: {requirements.minimum_gpu}")
        print(f"Recommended CPU: {requirements.recommended_cpu}")
        print(f"Recommended GPU: {requirements.recommended_gpu}")
    else:
        print("No requirements found!")

if __name__ == "__main__":
    asyncio.run(test_diablo_matching())