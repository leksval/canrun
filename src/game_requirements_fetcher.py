"""
Game Requirements Fetcher Module for CanRun
Fetches game requirements from multiple sources including Steam API,
PCGameBenchmark, and local cache with optimized fuzzy matching.
"""

import json
import logging
import asyncio
import aiohttp
from typing import Dict, List, Union, Optional
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re
import time
import sys
import os

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable
        base_path = sys._MEIPASS
    else:
        # Running as normal Python script
        base_path = Path(__file__).parent.parent
    return os.path.join(base_path, relative_path)

# Import the optimized fuzzy matcher
from optimized_game_fuzzy_matcher import OptimizedGameFuzzyMatcher

# Create a global instance for use throughout the module
game_fuzzy_matcher = OptimizedGameFuzzyMatcher()


@dataclass
class GameRequirements:
    """Data class for storing game requirements."""
    game_name: str
    minimum_cpu: str
    minimum_gpu: str
    minimum_ram_gb: int
    minimum_vram_gb: int
    minimum_storage_gb: int
    minimum_directx: str = "DirectX 11"
    minimum_os: str = "Windows 10"
    recommended_cpu: str = "Unknown"
    recommended_gpu: str = "Unknown"
    recommended_ram_gb: int = 0
    recommended_vram_gb: int = 0
    recommended_storage_gb: int = 0
    recommended_directx: str = "DirectX 12"
    recommended_os: str = "Windows 11"
    source: str = "Unknown"
    last_updated: str = ""


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    async def fetch(self, game_name: str) -> Optional[GameRequirements]:
        """Fetch game requirements from the source."""
        pass


class SteamAPISource(DataSource):
    """Steam Store API source for game requirements."""
    
    def __init__(self, llm_analyzer=None):
        self.base_url = "https://store.steampowered.com/api"
        self.search_url = "https://steamcommunity.com/actions/SearchApps"
        self.logger = logging.getLogger(__name__)
        self.llm_analyzer = llm_analyzer
    
    async def fetch(self, game_name: str) -> Optional[GameRequirements]:
        """Fetch game requirements from Steam API."""
        try:
            # First, search for the game to get its Steam ID
            steam_id = await self._search_game(game_name)
            if not steam_id:
                return None
            
            # Fetch detailed app info
            app_info = await self._get_app_info(steam_id)
            if not app_info:
                return None
            
            # Parse requirements from app info
            requirements = self._parse_requirements(app_info, game_name)
            return requirements
            
        except Exception as e:
            self.logger.error(f"Steam API fetch failed for {game_name}: {e}")
            return None
    
    async def _search_game(self, game_name: str) -> Optional[str]:
        """Search for a game and return its Steam ID."""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'text': game_name,
                    'max_results': 10
                }
                
                async with session.get(self.search_url, params=params) as response:
                    if response.status == 200:
                        try:
                            # Try to parse as JSON first (Steam Community API)
                            data = await response.json()
                            if isinstance(data, list) and len(data) > 0:
                                # Return the first match's app ID
                                app_data = data[0]
                                if 'appid' in app_data:
                                    return str(app_data['appid'])
                        except Exception as json_e:
                            # Fallback to text parsing if JSON fails
                            text = await response.text()
                            
                            # Use LLM to extract Steam app ID from HTML
                            if self.llm_analyzer:
                                try:
                                    prompt = f"""
                                    Extract the Steam app ID from this content. Look for app IDs in JSON format or data-ds-appid attributes.
                                    Return only the numeric app ID, nothing else.
                                    
                                    Content:
                                    {text[:2000]}  # Limit to first 2000 chars for LLM processing
                                    """
                                    
                                    app_id = await self.llm_analyzer.analyze_text(prompt)
                                    if app_id and app_id.strip().isdigit():
                                        return app_id.strip()
                                except Exception as e:
                                    self.logger.debug(f"LLM parsing failed: {e}")
                            
                            # Fallback regex parsing for HTML format
                            match = re.search(r'data-ds-appid="(\d+)"', text)
                            if match:
                                return match.group(1)
                            
                            # Try to find JSON-like app ID patterns
                            match = re.search(r'"appid":\s*(\d+)', text)
                            if match:
                                return match.group(1)
                                
        except Exception as e:
            self.logger.debug(f"Steam search failed: {e}")
        
        return None
    
    async def _get_app_info(self, steam_id: str) -> Optional[Dict]:
        """Get detailed app information from Steam Store API (public, no key required)."""
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{self.base_url}/appdetails"
                params = {
                    'appids': steam_id,
                    'cc': 'US',
                    'l': 'english'
                }
                
                # Add retry logic for reliability
                for attempt in range(3):
                    try:
                        self.logger.debug(f"Fetching Steam app info for ID {steam_id}, attempt {attempt + 1}")
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                if steam_id in data and data[steam_id].get('success'):
                                    self.logger.debug(f"Successfully fetched app info for {steam_id}")
                                    return data[steam_id]['data']
                                else:
                                    self.logger.warning(f"Steam API returned unsuccessful response for {steam_id}")
                                    return None
                            elif response.status == 429:  # Rate limited
                                wait_time = 2 ** attempt
                                self.logger.warning(f"Rate limited by Steam API, waiting {wait_time}s")
                                if attempt < 2:
                                    await asyncio.sleep(wait_time)
                                continue
                            else:
                                self.logger.warning(f"Steam API returned status {response.status}")
                                return None
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Steam API timeout, attempt {attempt + 1}/3")
                        if attempt < 2:
                            await asyncio.sleep(1)
                        continue
                    except Exception as e:
                        self.logger.warning(f"Steam API error: {e}, attempt {attempt + 1}/3")
                        if attempt < 2:
                            await asyncio.sleep(1)
                        continue
                    
                    break  # Success or final failure
                        
        except Exception as e:
            self.logger.error(f"Steam app info fetch failed: {e}")
        
        return None
    
    def _parse_requirements(self, app_info: Dict, game_name: str) -> Optional[GameRequirements]:
        """Parse requirements from Steam app info."""
        try:
            pc_requirements = app_info.get('pc_requirements', {})
            if not pc_requirements:
                return None
            
            minimum = self._parse_requirement_text(pc_requirements.get('minimum', ''))
            recommended = self._parse_requirement_text(pc_requirements.get('recommended', ''))
            
            return GameRequirements(
                game_name=game_name,
                **self._dict_to_dataclass_fields(minimum, recommended),
                source='Steam API',
                last_updated=str(int(time.time()))
            )
        except Exception as e:
            self.logger.debug(f"Steam requirements parsing failed: {e}")
            return None
    
    def _parse_requirement_text(self, text: str) -> Dict[str, str]:
        """Parse requirement text into structured format."""
        requirements = {}
        
        # Clean HTML tags first
        clean_text = re.sub(r'<[^>]+>', '\n', text)
        clean_text = re.sub(r'&nbsp;', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Improved requirement patterns that stop at the next field
        patterns = {
            'os': r'OS:\s*([^<>\n]*?)(?=\s*(?:Processor|Memory|Graphics|DirectX|Storage|Sound|Additional|$))',
            'processor': r'Processor:\s*([^<>\n]*?)(?=\s*(?:Memory|Graphics|DirectX|Storage|Sound|Additional|$))',
            'memory': r'Memory:\s*([^<>\n]*?)(?=\s*(?:Graphics|DirectX|Storage|Sound|Additional|$))',
            'graphics': r'Graphics:\s*([^<>\n]*?)(?=\s*(?:DirectX|Storage|Sound|Additional|$))',
            'directx': r'DirectX:\s*([^<>\n]*?)(?=\s*(?:Storage|Sound|Additional|$))',
            'storage': r'Storage:\s*([^<>\n]*?)(?=\s*(?:Sound|Additional|$))',
            'sound': r'Sound Card:\s*([^<>\n]*?)(?=\s*(?:Additional|$))'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, clean_text, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                # Remove any trailing punctuation and extra whitespace
                value = re.sub(r'[.,:;]+$', '', value).strip()
                if value:
                    requirements[key] = value
        
        # If no structured parsing worked, try a simpler approach
        if not requirements:
            # Split by common delimiters and try to extract key-value pairs
            lines = re.split(r'[<>]|(?:\s*(?:Processor|Memory|Graphics|DirectX|Storage|Sound)\s*:)', clean_text)
            current_key = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this line contains a requirement key
                if re.match(r'^(OS|Processor|Memory|Graphics|DirectX|Storage|Sound)', line, re.IGNORECASE):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().lower()
                        value = parts[1].strip()
                        if key in ['os', 'processor', 'memory', 'graphics', 'directx', 'storage', 'sound']:
                            requirements[key] = value
        
        return requirements

    def _dict_to_dataclass_fields(self, minimum: Dict[str, str], recommended: Dict[str, str]) -> Dict[str, any]:
        """Convert old dict format to new dataclass field format."""
        def parse_storage(value: str) -> int:
            """Parse storage value like '25 GB' to integer."""
            if not value:
                return 0
            # Extract number from strings like "25 GB", "2.5 GB", etc.
            match = re.search(r'(\d+\.?\d*)', str(value))
            return int(float(match.group(1))) if match else 0
        
        def parse_ram(value: str) -> int:
            """Parse RAM value like '8 GB RAM' to integer."""
            if not value:
                return 0
            # Extract number from strings like "8 GB", "16 GB RAM", etc.
            match = re.search(r'(\d+)', str(value))
            return int(match.group(1)) if match else 0
        
        return {
            'minimum_cpu': minimum.get('processor', 'Unknown'),
            'minimum_gpu': minimum.get('graphics', 'Unknown'),
            'minimum_ram_gb': parse_ram(minimum.get('memory', '0')),
            'minimum_vram_gb': 0,  # Not typically in Steam API data
            'minimum_storage_gb': parse_storage(minimum.get('storage', '0')),
            'minimum_directx': minimum.get('directx', 'DirectX 11'),
            'minimum_os': minimum.get('os', 'Windows 10'),
            'recommended_cpu': recommended.get('processor', 'Unknown'),
            'recommended_gpu': recommended.get('graphics', 'Unknown'),
            'recommended_ram_gb': parse_ram(recommended.get('memory', '0')),
            'recommended_vram_gb': 0,  # Not typically in Steam API data
            'recommended_storage_gb': parse_storage(recommended.get('storage', '0')),
            'recommended_directx': recommended.get('directx', 'DirectX 12'),
            'recommended_os': recommended.get('os', 'Windows 11')
        }


class PCGameBenchmarkSource(DataSource):
    """PCGameBenchmark community source for game requirements."""
    
    def __init__(self):
        self.base_url = "https://www.pcgamebenchmark.com"
        self.logger = logging.getLogger(__name__)
    
    async def fetch(self, game_name: str) -> Optional[GameRequirements]:
        """Fetch game requirements from PCGameBenchmark."""
        try:
            # This is a placeholder implementation
            # In a real implementation, you would scrape the website
            # or use their API if available
            self.logger.info(f"PCGameBenchmark fetch for {game_name} - placeholder")
            return None
        except Exception as e:
            self.logger.error(f"PCGameBenchmark fetch failed for {game_name}: {e}")
            return None


class LocalCacheSource(DataSource):
    """Local cache source for game requirements."""
    
    def __init__(self, cache_path: Optional[Path] = None):
        if cache_path is None:
            cache_path = Path(get_resource_path("data/game_requirements.json"))
        self.cache_path = cache_path
        self.logger = logging.getLogger(__name__)
        self._cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cached game requirements."""
        try:
            if self.cache_path.exists():
                with open(self.cache_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
        return {}
    
    async def fetch(self, game_name: str) -> Optional[GameRequirements]:
        """Fetch game requirements from local cache with optimized fuzzy matching."""
        try:
            games = self._cache.get('games', {})
            
            # Try exact match first
            if game_name in games:
                game_data = games[game_name]
                minimum = game_data.get('minimum', {})
                recommended = game_data.get('recommended', {})
                
                # Parse storage values
                def parse_storage(value: str) -> int:
                    if not value:
                        return 0
                    match = re.search(r'(\d+\.?\d*)', str(value))
                    return int(float(match.group(1))) if match else 0
                
                # Parse RAM values
                def parse_ram(value: str) -> int:
                    if not value:
                        return 0
                    match = re.search(r'(\d+)', str(value))
                    return int(match.group(1)) if match else 0
                
                return GameRequirements(
                    game_name=game_name,
                    minimum_cpu=minimum.get('processor', 'Unknown'),
                    minimum_gpu=minimum.get('graphics', 'Unknown'),
                    minimum_ram_gb=parse_ram(minimum.get('memory', '0')),
                    minimum_vram_gb=0,
                    minimum_storage_gb=parse_storage(minimum.get('storage', '0')),
                    minimum_directx=minimum.get('directx', 'DirectX 11'),
                    minimum_os=minimum.get('os', 'Windows 10'),
                    recommended_cpu=recommended.get('processor', 'Unknown'),
                    recommended_gpu=recommended.get('graphics', 'Unknown'),
                    recommended_ram_gb=parse_ram(recommended.get('memory', '0')),
                    recommended_vram_gb=0,
                    recommended_storage_gb=parse_storage(recommended.get('storage', '0')),
                    recommended_directx=recommended.get('directx', 'DirectX 12'),
                    recommended_os=recommended.get('os', 'Windows 11'),
                    source='Local Cache',
                    last_updated=str(int(time.time()))
                )
            
            # Use optimized fuzzy matching
            cache_candidates = list(games.keys())
            match_result = game_fuzzy_matcher.find_best_match(game_name, cache_candidates, steam_priority=False)
            
            if match_result:
                matched_name, confidence = match_result
                game_data = games[matched_name]
                self.logger.info(f"Fuzzy matched '{game_name}' to '{matched_name}' (confidence: {confidence:.3f})")
                
                minimum = game_data.get('minimum', {})
                recommended = game_data.get('recommended', {})
                
                # Parse storage values
                def parse_storage(value: str) -> int:
                    if not value:
                        return 0
                    match = re.search(r'(\d+\.?\d*)', str(value))
                    return int(float(match.group(1))) if match else 0
                
                # Parse RAM values
                def parse_ram(value: str) -> int:
                    if not value:
                        return 0
                    match = re.search(r'(\d+)', str(value))
                    return int(match.group(1)) if match else 0
                
                return GameRequirements(
                    game_name=matched_name,
                    minimum_cpu=minimum.get('processor', 'Unknown'),
                    minimum_gpu=minimum.get('graphics', 'Unknown'),
                    minimum_ram_gb=parse_ram(minimum.get('memory', '0')),
                    minimum_vram_gb=0,
                    minimum_storage_gb=parse_storage(minimum.get('storage', '0')),
                    minimum_directx=minimum.get('directx', 'DirectX 11'),
                    minimum_os=minimum.get('os', 'Windows 10'),
                    recommended_cpu=recommended.get('processor', 'Unknown'),
                    recommended_gpu=recommended.get('graphics', 'Unknown'),
                    recommended_ram_gb=parse_ram(recommended.get('memory', '0')),
                    recommended_vram_gb=0,
                    recommended_storage_gb=parse_storage(recommended.get('storage', '0')),
                    recommended_directx=recommended.get('directx', 'DirectX 12'),
                    recommended_os=recommended.get('os', 'Windows 11'),
                    source='Local Cache',
                    last_updated=str(int(time.time()))
                )
            
            return None
        except Exception as e:
            self.logger.error(f"Local cache fetch failed for {game_name}: {e}")
            return None
    
    # Old fuzzy matching methods removed - using optimized_game_fuzzy_matcher instead
    
    def save_to_cache(self, requirements: GameRequirements):
        """Save requirements to local cache."""
        try:
            if 'games' not in self._cache:
                self._cache['games'] = {}
            
            self._cache['games'][requirements.game_name] = {
                'minimum': requirements.minimum,
                'recommended': requirements.recommended
            }
            
            # Save to file
            with open(self.cache_path, 'w') as f:
                json.dump(self._cache, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save to cache: {e}")


class GameRequirementsFetcher:
    """Main game requirements fetcher that coordinates multiple sources."""
    
    def __init__(self, llm_analyzer=None):
        self.logger = logging.getLogger(__name__)
        self.llm_analyzer = llm_analyzer
        self.steam_source = SteamAPISource(llm_analyzer)
        self.cache_source = LocalCacheSource()
        self.sources = [
            self.steam_source,   # Primary source - most up-to-date requirements
            self.cache_source,   # Fallback for offline/cached data
        ]
    
    async def fetch_requirements(self, game_name: str) -> Optional[GameRequirements]:
        """
        Fetch game requirements with Steam API prioritization and optimized fuzzy matching.
        Uses the optimized fuzzy matcher as the primary LLM tool for intelligent game matching.
        """
        
        try:
            # Step 1: Gather candidates from all sources
            steam_candidates = []
            cache_candidates = list(self.cache_source._cache.get('games', {}).keys())
            
            # Try to get Steam API candidates (if available)
            try:
                # For now, we'll use cache candidates as Steam isn't implemented fully
                # In production, this would query Steam API for game search results
                pass
            except Exception as e:
                self.logger.debug(f"Steam API search failed: {e}")
            
            # Step 2: Use optimized fuzzy matcher with Steam prioritization
            if steam_candidates or cache_candidates:
                match_result = await game_fuzzy_matcher.match_with_steam_fallback(
                    game_name, steam_candidates, cache_candidates
                )
                
                if match_result:
                    matched_name, confidence, source_type = match_result
                    self.logger.info(f"Optimized fuzzy match: '{game_name}' -> '{matched_name}' (confidence: {confidence:.3f}, source: {source_type})")
                    
                    # Fetch the requirements for the matched game
                    if source_type == "Local Cache":
                        games = self.cache_source._cache.get('games', {})
                        if matched_name in games:
                            game_data = games[matched_name]
                            minimum = game_data.get('minimum', {})
                            recommended = game_data.get('recommended', {})
                            
                            # Parse storage values
                            def parse_storage(value: str) -> int:
                                if not value:
                                    return 0
                                match = re.search(r'(\d+\.?\d*)', str(value))
                                return int(float(match.group(1))) if match else 0
                            
                            # Parse RAM values
                            def parse_ram(value: str) -> int:
                                if not value:
                                    return 0
                                match = re.search(r'(\d+)', str(value))
                                return int(match.group(1)) if match else 0
                            
                            return GameRequirements(
                                game_name=matched_name,
                                minimum_cpu=minimum.get('processor', 'Unknown'),
                                minimum_gpu=minimum.get('graphics', 'Unknown'),
                                minimum_ram_gb=parse_ram(minimum.get('memory', '0')),
                                minimum_vram_gb=0,
                                minimum_storage_gb=parse_storage(minimum.get('storage', '0')),
                                minimum_directx=minimum.get('directx', 'DirectX 11'),
                                minimum_os=minimum.get('os', 'Windows 10'),
                                recommended_cpu=recommended.get('processor', 'Unknown'),
                                recommended_gpu=recommended.get('graphics', 'Unknown'),
                                recommended_ram_gb=parse_ram(recommended.get('memory', '0')),
                                recommended_vram_gb=0,
                                recommended_storage_gb=parse_storage(recommended.get('storage', '0')),
                                recommended_directx=recommended.get('directx', 'DirectX 12'),
                                recommended_os=recommended.get('os', 'Windows 11'),
                                source='Local Cache (Fuzzy Matched)',
                                last_updated=str(int(time.time()))
                            )
                    
                    # For Steam API results, would fetch from Steam here
                    # elif source_type == "Steam API":
                    #     return await self.steam_source.fetch(matched_name)
            
            # Step 3: Fallback to traditional source fetching if fuzzy matching fails
            self.logger.info(f"Fuzzy matching failed for '{game_name}', trying direct source fetching")
            
            for source in self.sources:
                try:
                    requirements = await source.fetch(game_name)
                    if requirements:
                        self.logger.info(f"Found requirements for {game_name} from {requirements.source}")
                        
                        # Cache the result if it's from a remote source
                        if requirements.source != 'Local Cache':
                            await self._cache_requirements(requirements)
                        
                        return requirements
                except Exception as e:
                    self.logger.error(f"Source {source.__class__.__name__} failed: {e}")
            
            self.logger.warning(f"No requirements found for '{game_name}' - tried fuzzy matching and direct fetching")
            return None
            
        except Exception as e:
            self.logger.error(f"fetch_requirements failed for '{game_name}': {e}")
            return None
    
    async def _cache_requirements(self, requirements: GameRequirements):
        """Cache requirements locally."""
        try:
            cache_source = LocalCacheSource()
            cache_source.save_to_cache(requirements)
        except Exception as e:
            self.logger.error(f"Failed to cache requirements: {e}")
    
    async def batch_fetch(self, game_names: List[str]) -> Dict[str, Optional[GameRequirements]]:
        """Fetch requirements for multiple games concurrently."""
        tasks = []
        for game_name in game_names:
            task = asyncio.create_task(self.fetch_requirements(game_name))
            tasks.append((game_name, task))
        
        results = {}
        for game_name, task in tasks:
            try:
                results[game_name] = await task
            except Exception as e:
                self.logger.error(f"Batch fetch failed for {game_name}: {e}")
                results[game_name] = None
        
        return results
    
    def add_source(self, source: DataSource):
        """Add a new data source."""
        self.sources.append(source)
    
    def get_supported_games(self) -> List[str]:
        """Get list of games that have cached requirements."""
        try:
            cache_source = LocalCacheSource()
            return list(cache_source._cache.get('games', {}).keys())
        except Exception as e:
            self.logger.error(f"Failed to get supported games: {e}")
            return []


async def main():
    """Test the game requirements fetcher."""
    fetcher = GameRequirementsFetcher()
    
    # Test single game fetch
    print("Testing single game fetch...")
    requirements = await fetcher.fetch_requirements("Cyberpunk 2077")
    if requirements:
        print(f"Game: {requirements.game_name}")
        print(f"Source: {requirements.source}")
        print(f"Minimum: {requirements.minimum}")
        print(f"Recommended: {requirements.recommended}")
    else:
        print("No requirements found")
    
    # Test batch fetch
    print("\nTesting batch fetch...")
    games = ["Cyberpunk 2077", "Elden Ring", "Baldur's Gate 3"]
    results = await fetcher.batch_fetch(games)
    
    for game, req in results.items():
        if req:
            print(f"{game}: Found ({req.source})")
        else:
            print(f"{game}: Not found")
    
    # Show supported games
    print(f"\nSupported games: {fetcher.get_supported_games()}")


if __name__ == "__main__":
    asyncio.run(main())