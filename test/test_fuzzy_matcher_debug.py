import asyncio
import logging
from src.optimized_game_fuzzy_matcher import OptimizedGameFuzzyMatcher

# Configure logging to see detailed output
logging.basicConfig(level=logging.INFO)

async def test_fuzzy_matcher():
    """
    Test the fuzzy matcher with a problematic query and a controlled set of candidates.
    """
    matcher = OptimizedGameFuzzyMatcher(threshold=0.1)
    
    query = "daiblo"
    candidates = ["Cyberpunk 2077", "Diablo IV"]
    
    print(f"Testing query: '{query}'")
    print(f"Candidates: {candidates}")
    
    # Test the fuzzy match directly
    for candidate in candidates:
        score = matcher.fuzzy_match_with_variants(query, candidate)
        print(f"Score for '{candidate}': {score:.3f}")
        
    # Test the full find_best_match logic
    best_match_result = await matcher.find_best_match(query, candidates)
    
    if best_match_result:
        best_match, confidence = best_match_result
        print(f"\nBest match found: '{best_match}' with confidence {confidence:.3f}")
    else:
        print("\nNo confident match found.")

if __name__ == "__main__":
    asyncio.run(test_fuzzy_matcher())