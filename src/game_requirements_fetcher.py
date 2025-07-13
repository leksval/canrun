"""
Game Requirements Fetcher Module for CanRun
Fetches game requirements from multiple sources including Steam API, 
PCGameBenchmark, and local cache.
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


@dataclass
class GameRequirements:
    """Data class for storing game requirements."""
    game_name: str
    minimum: Dict[str, str]
    recommended: Dict[str, str]
    source: str
    last_updated: str


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
        self.search_url = "https://store.steampowered.com/search/suggest"
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
                    'term': game_name,
                    'f': 'games',
                    'cc': 'US',
                    'l': 'english'
                }
                
                async with session.get(self.search_url, params=params) as response:
                    if response.status == 200:
                        text = await response.text()
                        # Use LLM to extract Steam app ID from HTML
                        if self.llm_analyzer:
                            try:
                                prompt = f"""
                                Extract the Steam app ID from this HTML content. Look for data-ds-appid attributes in links.
                                Return only the numeric app ID, nothing else.
                                
                                HTML content:
                                {text[:2000]}  # Limit to first 2000 chars for LLM processing
                                """
                                
                                app_id = await self.llm_analyzer.analyze_text(prompt)
                                if app_id and app_id.strip().isdigit():
                                    return app_id.strip()
                            except Exception as e:
                                self.logger.debug(f"LLM parsing failed: {e}")
                        
                        # Fallback regex parsing if LLM fails
                        match = re.search(r'data-ds-appid="(\d+)"', text)
                        if match:
                            return match.group(1)
        except Exception as e:
            self.logger.debug(f"Steam search failed: {e}")
        
        return None
    
    async def _get_app_info(self, steam_id: str) -> Optional[Dict]:
        """Get detailed app information from Steam API."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/appdetails"
                params = {
                    'appids': steam_id,
                    'cc': 'US',
                    'l': 'english'
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if steam_id in data and data[steam_id].get('success'):
                            return data[steam_id]['data']
        except Exception as e:
            self.logger.debug(f"Steam app info fetch failed: {e}")
        
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
                minimum=minimum,
                recommended=recommended,
                source='Steam API',
                last_updated=str(int(time.time()))
            )
        except Exception as e:
            self.logger.debug(f"Steam requirements parsing failed: {e}")
            return None
    
    def _parse_requirement_text(self, text: str) -> Dict[str, str]:
        """Parse requirement text into structured format."""
        requirements = {}
        
        # Common requirement patterns
        patterns = {
            'os': r'OS:\s*(.+?)(?:<br>|\n|$)',
            'processor': r'Processor:\s*(.+?)(?:<br>|\n|$)',
            'memory': r'Memory:\s*(.+?)(?:<br>|\n|$)',
            'graphics': r'Graphics:\s*(.+?)(?:<br>|\n|$)',
            'directx': r'DirectX:\s*(.+?)(?:<br>|\n|$)',
            'storage': r'Storage:\s*(.+?)(?:<br>|\n|$)',
            'sound': r'Sound Card:\s*(.+?)(?:<br>|\n|$)'
        }
        
        # Clean HTML tags
        clean_text = re.sub(r'<[^>]+>', ' ', text)
        
        for key, pattern in patterns.items():
            match = re.search(pattern, clean_text, re.IGNORECASE)
            if match:
                requirements[key] = match.group(1).strip()
        
        return requirements


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
            cache_path = Path(__file__).parent.parent / "data" / "game_requirements.json"
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
        """Fetch game requirements from local cache."""
        try:
            games = self._cache.get('games', {})
            
            # Try exact match first
            if game_name in games:
                game_data = games[game_name]
                return GameRequirements(
                    game_name=game_name,
                    minimum=game_data.get('minimum', {}),
                    recommended=game_data.get('recommended', {}),
                    source='Local Cache',
                    last_updated=str(int(time.time()))
                )
            
            # Try fuzzy matching
            for cached_name, game_data in games.items():
                if self._fuzzy_match(game_name, cached_name):
                    return GameRequirements(
                        game_name=cached_name,
                        minimum=game_data.get('minimum', {}),
                        recommended=game_data.get('recommended', {}),
                        source='Local Cache',
                        last_updated=str(int(time.time()))
                    )
            
            return None
        except Exception as e:
            self.logger.error(f"Local cache fetch failed for {game_name}: {e}")
            return None
    
    def _fuzzy_match(self, query: str, target: str, threshold: float = 0.8) -> bool:
        """Simple fuzzy matching for game names."""
        query_lower = query.lower()
        target_lower = target.lower()
        
        # Check if query is contained in target or vice versa
        if query_lower in target_lower or target_lower in query_lower:
            return True
        
        # Simple character overlap check
        query_chars = set(query_lower.replace(' ', ''))
        target_chars = set(target_lower.replace(' ', ''))
        
        if len(query_chars) == 0 or len(target_chars) == 0:
            return False
        
        overlap = len(query_chars & target_chars)
        union = len(query_chars | target_chars)
        
        return (overlap / union) >= threshold
    
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
        self.sources = [
            SteamAPISource(llm_analyzer),  # Primary source - most up-to-date requirements
            LocalCacheSource(),           # Fallback for offline/cached data
        ]
    
    async def fetch_requirements(self, game_name: str) -> Optional[GameRequirements]:
        """Fetch game requirements from all available sources."""
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
        
        self.logger.warning(f"No requirements found for {game_name}")
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