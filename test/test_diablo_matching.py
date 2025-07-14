#!/usr/bin/env python3
"""Test script to debug the Diablo 4 vs Diablo IV matching issue."""

import json
import asyncio
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from game_requirements_fetcher import GameRequirementsFetcher, LocalCacheSource
from rtx_llm_analyzer import GAssistLLMAnalyzer

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
    
    # Test 3: Test LLM analyzer fallback logic
    analyzer = GAssistLLMAnalyzer()
    print(f"LLM Model available: {analyzer.model_available}")
    
    # Test 4: Test the fallback game matching directly
    result = analyzer._fallback_game_matching("Diablo 4", available_games)
    print(f"Fallback matching result: {result}")
    
    # Test 5: Test the full LLM interpretation
    llm_result = await analyzer.interpret_game_requirements("Diablo 4", available_games)
    print(f"LLM interpretation result: {llm_result}")
    
    # Test 6: Test the full fetcher logic
    fetcher = GameRequirementsFetcher(llm_analyzer=analyzer)
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