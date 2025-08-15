"""
CanRun Game Matcher - Standalone module for NVIDIA G-Assist Plugin
Optimized fuzzy matching for video game titles with Steam API integration.
Flattened structure with all dependencies bundled for PyInstaller compatibility.
"""

import re
import logging
import asyncio
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class GameMatchResult:
    """Result of game matching operation."""
    matched_name: str
    confidence: float
    source: str  # "Steam", "Cache", "Database"
    message: str = ""  # Disambiguation or info message


class CanRunGameMatcher:
    """
    Ultra-optimized fuzzy matcher for video game titles.
    Designed for G-Assist integration with game-specific optimizations.
    Data loaded once in constructor for efficiency.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self.cache = {}  # Unified cache for all preprocessing  
        self.logger = logging.getLogger(__name__)
        
        # Load static data once in constructor for efficiency
        self._load_static_data()
        
        self.logger.info("CanRun game matcher initialized")
    
    def _load_static_data(self):
        """Load all static matching data once (following user's pattern)."""
        
        # Simple query preferences - let Steam API provide actual game names
        self.query_preferences = {
            'diablo': 'diablo',
            'gta': 'grand theft auto 5',  # Most recent GTA
            'call of duty': 'call of duty modern warfare',
        }
        
        # Disambiguation messages for ambiguous queries
        self.disambiguation_messages = {
            'diablo': "Multiple Diablo games found. Using Steam's most relevant result. Try 'diablo 2', 'diablo 4', or 'diablo iv' for specific versions.",
            'gta': "Multiple GTA games found. Using Steam's most relevant result. Try 'gta 3', 'gta 4', or 'gta 5' for specific versions.", 
            'call of duty': "Multiple Call of Duty games found. Try being more specific like 'call of duty modern warfare'.",
        }

        # Expanded game-specific mappings for common abbreviations (loaded once)
        self.acronym_map = {
            'gta': ['grand', 'theft', 'auto'],
            'cod': ['call', 'of', 'duty'],
            'cs': ['counter', 'strike'],
            'csgo': ['counter', 'strike', 'global', 'offensive'],
            'cs2': ['counter', 'strike', '2'],
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
            'wow': ['world', 'of', 'warcraft'],
            'diablo': ['diablo'],
            'valorant': ['valorant'],
            'fortnite': ['fortnite'],
            'apex': ['apex', 'legends'],
            'ow': ['overwatch'],
            'ow2': ['overwatch', '2']
        }

        # Number normalization patterns (critical for Diablo 4 -> Diablo IV) - loaded once
        self.roman_map = {
            'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5',
            'vi': '6', 'vii': '7', 'viii': '8', 'ix': '9', 'x': '10',
            'xi': '11', 'xii': '12', 'xiii': '13', 'xiv': '14', 'xv': '15'
        }
        
        # Reverse mapping for number to roman conversion (loaded once)
        self.number_to_roman = {v: k for k, v in self.roman_map.items()}

        # Common game subtitles/editions to handle gracefully (loaded once)
        self.edition_words = {
            'edition', 'remastered', 'remake', 'definitive', 'ultimate',
            'goty', 'complete', 'deluxe', 'special', 'anniversary', 'enhanced',
            'directors', 'cut', 'gold', 'platinum', 'legendary'
        }

    def preprocess_title(self, title: str) -> List[str]:
        """Preprocess and tokenize game title with aggressive normalization."""
        cache_key = f"prep_{title}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Lowercase and remove special chars
        clean = re.sub(r'[^a-z0-9\s]', ' ', title.lower())
        
        # Keep original game name with numbers intact
        original_tokens = clean.split()
        
        # Split into tokens
        tokens = clean.split()
        
        # Always preserve original game name with number intact
        if len(original_tokens) >= 2 and any(t.isdigit() or t in self.roman_map for t in original_tokens):
            self.logger.debug(f"Preserving numbered game title: {title}")
            self.cache[cache_key] = original_tokens
            return original_tokens

        # Process each token using pre-loaded acronym map
        processed_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Check if token is a known acronym (from pre-loaded data)
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
        # Convert Roman numerals to numbers (using pre-loaded data)
        if token in self.roman_map:
            return self.roman_map[token]
        
        # Convert numbers to roman numerals for reverse matching (using pre-loaded data)
        if token in self.number_to_roman:
            return self.number_to_roman[token]

        # Handle year formats (23 -> 2023)
        if token.isdigit() and len(token) == 2:
            year = int(token)
            if year < 50:
                return f"20{token}"
            else:
                return f"19{token}"

        # Preserve numbers to avoid losing them in game titles
        if token.isdigit() or token in self.roman_map:
            return token
            
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
        
        # Roman/Number conversion variants using pre-loaded maps
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

        # Weight adjustments for game-specific patterns using pre-loaded edition words
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

    def should_show_disambiguation(self, query: str, candidates: List[str]) -> bool:
        """
        Determine if a query should show disambiguation message.
        
        Args:
            query: The game query
            candidates: Available game candidates from Steam search
            
        Returns:
            True if disambiguation message should be shown
        """
        query_lower = query.lower().strip()
        
        # Check if this is a known ambiguous query (using pre-loaded data)
        if query_lower in self.disambiguation_messages:
            # Count how many candidates contain the base query
            matching_candidates = []
            for candidate in candidates:
                if query_lower in candidate.lower():
                    matching_candidates.append(candidate)
            
            # Show disambiguation if multiple matches found
            return len(matching_candidates) > 1
        
        return False

    async def find_best_match(self, query: str, candidates: List[str],
                       steam_priority: bool = True) -> Optional[GameMatchResult]:
        """
        Find the best match with Steam API prioritization and disambiguation support.
        
        Args:
            query: Game name to search for
            candidates: List of candidate game names
            steam_priority: Whether to prioritize results that look like Steam data
            
        Returns:
            GameMatchResult with match details or None
        """
        if not candidates:
            return None
            
        # For ambiguous queries, find the most similar match to what user typed
        query_lower = query.lower().strip()
        if query_lower in self.disambiguation_messages:
            # Find all relevant candidates
            relevant_candidates = []
            for candidate in candidates:
                candidate_lower = candidate.lower()
                if query_lower in candidate_lower:
                    relevant_candidates.append(candidate)
            
            if len(relevant_candidates) > 1:
                # For "diablo" specifically, prefer the most recent/popular version
                # Steam API returns results in order of relevance/popularity
                if query_lower == "diablo":
                    # Take the first Steam result which is most relevant
                    best_match = relevant_candidates[0]
                    self.logger.info(f"Diablo disambiguation: Steam returned {len(relevant_candidates)} results, selecting first: {best_match}")
                    
                    options_text = ", ".join(relevant_candidates[:3])
                    if len(relevant_candidates) > 3:
                        options_text += f" and {len(relevant_candidates) - 3} more"
                    
                    disambiguation_msg = f"Multiple Diablo games found: {options_text}. Selected most relevant: {best_match}."
                    return GameMatchResult(best_match, 1.0, "Steam", disambiguation_msg)
                
                # For other games, find the most similar match to the user's query
                best_match = None
                best_similarity = 0.0
                
                for candidate in relevant_candidates:
                    candidate_lower = candidate.lower()
                    # Calculate similarity - prefer shorter, more exact matches
                    if candidate_lower == query_lower:
                        # Exact match (e.g., "diablo" matches "diablo")
                        similarity = 1.0
                    elif candidate_lower.startswith(query_lower + " "):
                        # Direct prefix match (e.g., "diablo" matches "diablo 2")
                        # Shorter titles get higher scores
                        word_count = len(candidate_lower.split())
                        similarity = 0.9 / word_count  # Prefer shorter titles
                    else:
                        # Contains match but not prefix
                        similarity = 0.5
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = candidate
                
                if best_match:
                    options_text = ", ".join(relevant_candidates[:3])
                    if len(relevant_candidates) > 3:
                        options_text += f" and {len(relevant_candidates) - 3} more"
                    
                    disambiguation_msg = f"Multiple {query.title()} games found: {options_text}. Selected most similar: {best_match}."
                    return GameMatchResult(best_match, best_similarity, "Steam", disambiguation_msg)
        
        # Simple direct matching - let Steam API results speak for themselves
        query_lower = query.lower()
                    
        # Look for exact match in candidates
        for candidate in candidates:
            if candidate.lower() == query_lower:
                return GameMatchResult(candidate, 1.0, "Steam", "")
                
        # For games with numbers, preserve the exact numbered version from the query
        query_words = query_lower.split()
        if len(query_words) >= 2 and any(w.isdigit() or w in self.roman_map for w in query_words):
            # Keep the exact query if it contains a number
            self.logger.debug(f"Preserving numbered query: {query}")
            
            # Check if any candidate contains both the base name and the number
            for candidate in candidates:
                candidate_lower = candidate.lower()
                # Check if candidate contains all words from the query
                if all(word in candidate_lower for word in query_words):
                    return GameMatchResult(candidate, 1.0, "Steam", "")
                 
            # Do NOT strip numbers for partial matching - numbered games are distinct entries
            self.logger.info(f"No exact match for numbered game: '{query}'")
            return None
        
        # For simple fuzzy matching, use the best candidate with high enough score
        matches = []
        for candidate in candidates:
            # Simple similarity score
            query_set = set(query_lower.split())
            candidate_set = set(candidate.lower().split())
            
            intersection = query_set & candidate_set
            if intersection and len(intersection) / len(query_set) > 0.5:
                score = len(intersection) / max(len(query_set), len(candidate_set))
                matches.append((candidate, score))
        
        if matches:
            # Sort by score and return best match
            matches.sort(key=lambda x: x[1], reverse=True)
            best_match, best_score = matches[0]
            if best_score > 0.6:
                return GameMatchResult(best_match, best_score, "Steam", "")
                
        # If all else fails, return the first candidate
        if candidates:
            return GameMatchResult(candidates[0], 0.5, "Steam", "")
            
        return None

    def looks_like_steam_title(self, title: str) -> bool:
        """Heuristic to identify Steam-style game titles."""
        # Steam titles tend to be more formal and complete
        return (
            len(title) > 5 and  # Not just abbreviations
            not any(abbrev in title.lower() for abbrev in ['gta', 'cod', 'cs']) and
            ':' not in title  # Steam tends to use cleaner formatting
        )

    async def match_with_steam_fallback(self, query: str, steam_candidates: List[str],
                                      cache_candidates: List[str]) -> Optional[GameMatchResult]:
        """
        Match with Steam as default source and local cache as fallback only.
        
        Returns:
            GameMatchResult with match details and source information or None
        """
        # Steam is the default - try Steam search results first
        if steam_candidates:
            steam_match = await self.find_best_match(query, steam_candidates, steam_priority=True)
            if steam_match and steam_match.confidence >= self.threshold:
                steam_match.source = "Steam"
                return steam_match

        # Only fallback to local cache if Steam completely fails
        if cache_candidates:
            cache_match = await self.find_best_match(query, cache_candidates, steam_priority=False)
            if cache_match and cache_match.confidence >= self.threshold:
                cache_match.source = "Cache"
                return cache_match

        # Log failed match for debugging
        self.logger.warning(f"No match found for '{query}' - Steam returned {len(steam_candidates)} candidates, cache has {len(cache_candidates)} candidates")
        return None

    def quick_match_simple(self, query: str, candidates: List[str]) -> Optional[str]:
        """Quick synchronous matching for simple cases."""
        query_lower = query.lower().strip()
        
        # Direct exact match
        for candidate in candidates:
            if candidate.lower() == query_lower:
                return candidate
        
        # Partial match with high confidence
        for candidate in candidates:
            if query_lower in candidate.lower() and len(query_lower) > 3:
                return candidate
                
        return None