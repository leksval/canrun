"""
Optimized Game Fuzzy Matcher for G-Assist LLM Integration
Handles game name variations, prioritizes Steam API data, and provides intelligent matching.
"""

import re
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import asyncio


class OptimizedGameFuzzyMatcher:
    """
    Ultra-optimized fuzzy matcher for video game titles
    Designed for G-Assist LLM integration with game-specific optimizations
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self.cache = {}  # Unified cache for all preprocessing
        self.logger = logging.getLogger(__name__)

        # Expanded game-specific mappings for common abbreviations
        self.acronym_map = {
            'gta': ['grand', 'theft', 'auto'],
            'cod': ['call', 'of', 'duty'],
            'cs': ['counter', 'strike'],
            'csgo': ['counter', 'strike', 'global', 'offensive'],
            'pubg': ['playerunknowns', 'battlegrounds'],
            'ac': ['assassins', 'creed'],
            'ds': ['dark', 'souls'],
            'gow': ['god', 'of', 'war'],
            'hzd': ['horizon', 'zero', 'dawn'],
            'botw': ['breath', 'of', 'the', 'wild'],
            'mw': ['modern', 'warfare'],
            'nfs': ['need', 'for', 'speed'],
            'ff': ['final', 'fantasy'],
            'lol': ['league', 'of', 'legends'],
            'wow': ['world', 'of', 'warcraft']
        }

        # Number normalization patterns (critical for Diablo 4 -> Diablo IV)
        self.roman_map = {
            'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5',
            'vi': '6', 'vii': '7', 'viii': '8', 'ix': '9', 'x': '10',
            'xi': '11', 'xii': '12', 'xiii': '13', 'xiv': '14', 'xv': '15'
        }
        
        # Reverse mapping for number to roman conversion
        self.number_to_roman = {v: k for k, v in self.roman_map.items()}

        # Common game subtitles/editions to handle gracefully
        self.edition_words = {
            'edition', 'remastered', 'remake', 'definitive', 'ultimate',
            'goty', 'complete', 'deluxe', 'special', 'anniversary', 'enhanced'
        }

    def preprocess_title(self, title: str) -> List[str]:
        """Preprocess and tokenize game title with aggressive normalization."""
        cache_key = f"prep_{title}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Lowercase and remove special chars
        clean = re.sub(r'[^a-z0-9\s]', ' ', title.lower())
        
        # Split into tokens
        tokens = clean.split()

        # Process each token
        processed_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Check if token is a known acronym
            if token in self.acronym_map:
                processed_tokens.extend(self.acronym_map[token])
            
            # Check for multi-token acronyms (like cs:go -> csgo)
            elif i + 1 < len(tokens):
                combined = token + tokens[i + 1]
                if combined in self.acronym_map:
                    processed_tokens.extend(self.acronym_map[combined])
                    i += 1  # Skip next token
                else:
                    processed_tokens.append(self.normalize_token(token))
            else:
                processed_tokens.append(self.normalize_token(token))

            i += 1

        self.cache[cache_key] = processed_tokens
        return processed_tokens

    def normalize_token(self, token: str) -> str:
        """Normalize individual token with bidirectional roman/number conversion."""
        # Convert Roman numerals to numbers
        if token in self.roman_map:
            return self.roman_map[token]
        
        # Convert numbers to roman numerals for reverse matching
        if token in self.number_to_roman:
            return self.number_to_roman[token]

        # Handle year formats (23 -> 2023)
        if token.isdigit() and len(token) == 2:
            year = int(token)
            if year < 50:
                return f"20{token}"
            else:
                return f"19{token}"

        return token

    def fuzzy_match_with_variants(self, query: str, target: str) -> float:
        """
        Enhanced fuzzy matching that creates multiple variants for comparison.
        Specifically handles cases like "Diablo 4" -> "Diablo IV"
        """
        # Quick exact match check
        if query.lower() == target.lower():
            return 1.0

        # Generate variants of both query and target
        query_variants = self.generate_variants(query)
        target_variants = self.generate_variants(target)

        # Find best match among all combinations
        max_score = 0.0
        
        for q_variant in query_variants:
            for t_variant in target_variants:
                score = self.basic_fuzzy_match(q_variant, t_variant)
                max_score = max(max_score, score)
                
                # Early termination for perfect matches
                if score >= 0.95:
                    return score

        return max_score

    def generate_variants(self, title: str) -> List[str]:
        """Generate multiple variants of a game title for robust matching."""
        variants = [title]
        
        # Original preprocessing
        tokens = self.preprocess_title(title)
        if tokens:
            variants.append(' '.join(tokens))
        
        # Roman/Number conversion variants
        lower_title = title.lower()
        
        # Convert numbers to roman numerals
        for num, roman in self.number_to_roman.items():
            if num in lower_title:
                variant = lower_title.replace(num, roman)
                variants.append(variant)
        
        # Convert roman numerals to numbers  
        for roman, num in self.roman_map.items():
            if roman in lower_title:
                variant = lower_title.replace(roman, num)
                variants.append(variant)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)
        
        return unique_variants

    def basic_fuzzy_match(self, title1: str, title2: str) -> float:
        """Basic fuzzy matching with token-based similarity."""
        tokens1 = self.preprocess_title(title1)
        tokens2 = self.preprocess_title(title2)
        
        if not tokens1 or not tokens2:
            return 0.0

        set1, set2 = set(tokens1), set(tokens2)
        
        # Calculate Jaccard similarity
        intersection = set1 & set2
        union = set1 | set2
        
        if not union:
            return 0.0

        # Base score
        jaccard = len(intersection) / len(union)

        # Weight adjustments for game-specific patterns
        weight = 1.0

        # Boost score if main game words match
        main_words = intersection - self.edition_words
        if main_words:
            weight += 0.2

        # Small penalty for missing edition words
        edition_diff = (set1 ^ set2) & self.edition_words
        if edition_diff:
            weight -= 0.05 * len(edition_diff)

        return min(1.0, jaccard * weight)

    def normalize_game_name(self, game_name: str) -> str:
        """
        Normalize game name for consistent caching and matching.
        
        Args:
            game_name: Original game name
            
        Returns:
            Normalized game name with roman numerals and standardized format
        """
        # Use the preprocess_title method and join back the tokens
        tokens = self.preprocess_title(game_name)
        return ' '.join(tokens)

    def find_best_match(self, query: str, candidates: List[str],
                       steam_priority: bool = True) -> Optional[Tuple[str, float]]:
        """
        Find the best match with Steam API prioritization.
        
        Args:
            query: Game name to search for
            candidates: List of candidate game names
            steam_priority: Whether to prioritize results that look like Steam data
            
        Returns:
            Tuple of (best_match, confidence_score) or None
        """
        if not candidates:
            return None

        matches = []
        
        for candidate in candidates:
            score = self.fuzzy_match_with_variants(query, candidate)
            
            if score >= self.threshold:
                # Apply Steam priority boost
                if steam_priority and self.looks_like_steam_title(candidate):
                    score = min(1.0, score + 0.1)  # Small boost for Steam-like titles
                
                matches.append((candidate, score))

        if not matches:
            return None

        # Sort by score and return best match
        matches.sort(key=lambda x: x[1], reverse=True)
        best_match, best_score = matches[0]
        
        self.logger.info(f"Fuzzy match: '{query}' -> '{best_match}' (score: {best_score:.3f})")
        return best_match, best_score

    def looks_like_steam_title(self, title: str) -> bool:
        """Heuristic to identify Steam-style game titles."""
        # Steam titles tend to be more formal and complete
        return (
            len(title) > 5 and  # Not just abbreviations
            not any(abbrev in title.lower() for abbrev in ['gta', 'cod', 'cs']) and
            ':' not in title  # Steam tends to use cleaner formatting
        )

    async def match_with_steam_fallback(self, query: str, steam_candidates: List[str], 
                                      cache_candidates: List[str]) -> Optional[Tuple[str, float, str]]:
        """
        Match with Steam API prioritization and local cache fallback.
        
        Returns:
            Tuple of (matched_name, confidence_score, source) or None
        """
        # First try Steam API candidates
        if steam_candidates:
            steam_match = self.find_best_match(query, steam_candidates, steam_priority=True)
            if steam_match and steam_match[1] >= self.threshold:
                return steam_match[0], steam_match[1], "Steam API"

        # Fallback to local cache
        if cache_candidates:
            cache_match = self.find_best_match(query, cache_candidates, steam_priority=False)
            if cache_match and cache_match[1] >= self.threshold:
                return cache_match[0], cache_match[1], "Local Cache"

        # Log failed match for debugging
        self.logger.warning(f"No fuzzy match found for '{query}' in {len(steam_candidates)} Steam + {len(cache_candidates)} cache candidates")
        return None


# Singleton instance for use across the application
game_fuzzy_matcher = OptimizedGameFuzzyMatcher(threshold=0.7)
