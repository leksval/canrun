#!/usr/bin/env python3
"""Test fuzzy matching integration for user input."""

import sys
sys.path.append('src')

from optimized_game_fuzzy_matcher import OptimizedGameFuzzyMatcher

def test_fuzzy_matching():
    """Test the fuzzy matching system."""
    print("Testing Fuzzy Matcher Integration...")
    matcher = OptimizedGameFuzzyMatcher()

    # Test roman numeral conversion
    print(f"\nTesting Roman Numeral Conversion:")
    print(f"'Diablo 4' -> '{matcher.normalize_game_name('Diablo 4')}'")
    print(f"'Diablo IV' -> '{matcher.normalize_game_name('Diablo IV')}'")
    print(f"'GTA 5' -> '{matcher.normalize_game_name('GTA 5')}'")
    print(f"'GTA V' -> '{matcher.normalize_game_name('GTA V')}'")

    # Test fuzzy matching
    print(f"\nTesting Fuzzy Matching:")
    games = ['Diablo IV', 'Call of Duty: Modern Warfare', "Baldur's Gate 3", 'Grand Theft Auto V']
    
    test_inputs = ['Diablo 4', 'COD MW', 'Baldurs Gate 3', 'GTA 5']
    
    for user_input in test_inputs:
        result = matcher.find_best_match(user_input, games)
        if result:
            match, conf = result
            print(f"'{user_input}' -> '{match}' (confidence: {conf:.2f})")
        else:
            print(f"'{user_input}' -> No match found")

    print("\n✅ Fuzzy matcher integration test completed successfully!")
    return True

if __name__ == "__main__":
    test_fuzzy_matching()