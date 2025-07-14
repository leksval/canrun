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
        self.store_search_url = "https://store.steampowered.com/search/suggest"
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
        """Search for a game and return its Steam ID using multiple search methods."""
        self.logger.debug(f"Searching Steam for game: {game_name}")
        
        # Try multiple search methods in order of reliability
        search_methods = [
            self._search_steam_store_suggest,
            self._search_steam_community,
            self._search_steam_store_direct
        ]
        
        for method in search_methods:
            try:
                steam_id = await method(game_name)
                if steam_id:
                    self.logger.info(f"Found Steam ID {steam_id} for '{game_name}' using {method.__name__}")
                    return steam_id
            except Exception as e:
                self.logger.debug(f"Search method {method.__name__} failed: {e}")
                continue
        
        self.logger.warning(f"All Steam search methods failed for '{game_name}'")
        return None
    
    async def _search_steam_store_suggest(self, game_name: str) -> Optional[str]:
        """Search using Steam Store suggest API."""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'term': game_name,
                    'f': 'games',
                    'cc': 'US',
                    'l': 'english'
                }
                
                async with session.get(self.store_search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, list) and len(data) > 0:
                            # Return the first match's app ID
                            app_data = data[0]
                            if 'id' in app_data:
                                return str(app_data['id'])
                            elif 'appid' in app_data:
                                return str(app_data['appid'])
        except Exception as e:
            self.logger.debug(f"Steam store suggest search failed: {e}")
        
        return None
    
    async def _search_steam_community(self, game_name: str) -> Optional[str]:
        """Search using Steam Community API."""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'text': game_name,
                    'max_results': 10
                }
                
                async with session.get(self.search_url, params=params) as response:
                    if response.status == 200:
                        try:
                            # Try to parse as JSON first
                            data = await response.json()
                            if isinstance(data, list) and len(data) > 0:
                                app_data = data[0]
                                if 'appid' in app_data:
                                    return str(app_data['appid'])
                        except Exception:
                            # Fallback to text parsing if JSON fails
                            text = await response.text()
                            
                            # Use LLM to extract Steam app ID from HTML
                            if self.llm_analyzer:
                                try:
                                    prompt = f"""
                                    Extract the Steam app ID from this content. Look for app IDs in JSON format or data-ds-appid attributes.
                                    Return only the numeric app ID, nothing else.
                                    
                                    Content:
                                    {text[:2000]}
                                    """
                                    
                                    app_id = await self.llm_analyzer.analyze_text(prompt)
                                    if app_id and app_id.strip().isdigit():
                                        return app_id.strip()
                                except Exception as e:
                                    self.logger.debug(f"LLM parsing failed: {e}")
                            
                            # Fallback regex parsing
                            patterns = [
                                r'data-ds-appid="(\d+)"',
                                r'"appid":\s*(\d+)',
                                r'app/(\d+)/',
                                r'appid=(\d+)'
                            ]
                            
                            for pattern in patterns:
                                match = re.search(pattern, text)
                                if match:
                                    return match.group(1)
                                    
        except Exception as e:
            self.logger.debug(f"Steam community search failed: {e}")
        
        return None
    
    async def _search_steam_store_direct(self, game_name: str) -> Optional[str]:
        """Direct search on Steam store page."""
        try:
            # This is a more aggressive search method
            search_url = f"https://store.steampowered.com/search/?term={game_name.replace(' ', '+')}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url) as response:
                    if response.status == 200:
                        text = await response.text()
                        
                        # Look for app IDs in the search results
                        patterns = [
                            r'data-ds-appid="(\d+)"',
                            r'app/(\d+)/',
                            r'appid=(\d+)'
                        ]
                        
                        for pattern in patterns:
                            match = re.search(pattern, text)
                            if match:
                                return match.group(1)
                                
        except Exception as e:
            self.logger.debug(f"Steam store direct search failed: {e}")
        
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
            
            # Convert GameRequirements dataclass back to the expected cache format
            self._cache['games'][requirements.game_name] = {
                'minimum': {
                    'processor': requirements.minimum_cpu,
                    'graphics': requirements.minimum_gpu,
                    'memory': f"{requirements.minimum_ram_gb} GB",
                    'storage': f"{requirements.minimum_storage_gb} GB",
                    'directx': requirements.minimum_directx,
                    'os': requirements.minimum_os
                },
                'recommended': {
                    'processor': requirements.recommended_cpu,
                    'graphics': requirements.recommended_gpu,
                    'memory': f"{requirements.recommended_ram_gb} GB",
                    'storage': f"{requirements.recommended_storage_gb} GB",
                    'directx': requirements.recommended_directx,
                    'os': requirements.recommended_os
                }
            }
            
            # Save to file
            with open(self.cache_path, 'w') as f:
                json.dump(self._cache, f, indent=2)
            
            self.logger.debug(f"Successfully cached requirements for {requirements.game_name}")
            
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
        Fetch game requirements with Steam API prioritization and LLM-enhanced parsing.
        Steam API is the primary source, with intelligent fallback to local cache.
        """
        
        try:
            self.logger.info(f"Fetching requirements for '{game_name}' - Steam API priority enabled")
            
            # Step 1: Try Steam API first (primary source)
            try:
                steam_requirements = await self.steam_source.fetch(game_name)
                if steam_requirements:
                    self.logger.info(f"Successfully fetched '{game_name}' from Steam API")
                    
                    # Cache the Steam result for future use
                    await self._cache_requirements(steam_requirements)
                    return steam_requirements
                else:
                    self.logger.info(f"Steam API returned no results for '{game_name}', trying LLM-enhanced search")
                    
                    # Try LLM-enhanced game matching with Steam
                    if self.llm_analyzer:
                        enhanced_result = await self._llm_enhanced_steam_search(game_name)
                        if enhanced_result:
                            await self._cache_requirements(enhanced_result)
                            return enhanced_result
                            
            except Exception as e:
                self.logger.warning(f"Steam API failed for '{game_name}': {e}")
            
            # Step 2: Fallback to local cache with fuzzy matching
            self.logger.info(f"Trying local cache with fuzzy matching for '{game_name}'")
            cache_requirements = await self.cache_source.fetch(game_name)
            if cache_requirements:
                self.logger.info(f"Found '{game_name}' in local cache")
                return cache_requirements
            
            # Step 3: Try LLM-enhanced cache interpretation
            if self.llm_analyzer:
                self.logger.info(f"Trying LLM-enhanced cache interpretation for '{game_name}'")
                llm_result = await self._llm_enhanced_cache_search(game_name)
                if llm_result:
                    return llm_result
            
            self.logger.warning(f"No requirements found for '{game_name}' after trying all sources")
            return None
            
        except Exception as e:
            self.logger.error(f"fetch_requirements failed for '{game_name}': {e}")
            return None
    
    async def _llm_enhanced_steam_search(self, game_name: str) -> Optional[GameRequirements]:
        """Use LLM to enhance Steam search with intelligent game name variations."""
        try:
            if not self.llm_analyzer:
                return None
                
            self.logger.info(f"Using LLM to enhance Steam search for '{game_name}'")
            
            # Use LLM to generate game name variations
            variations = await self._generate_game_name_variations(game_name)
            
            # Try each variation with Steam API
            for variation in variations:
                try:
                    result = await self.steam_source.fetch(variation)
                    if result:
                        self.logger.info(f"LLM-enhanced Steam search successful: '{game_name}' -> '{variation}'")
                        # Update the game name to the original query
                        result.game_name = game_name
                        return result
                except Exception as e:
                    self.logger.debug(f"Steam search failed for variation '{variation}': {e}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"LLM-enhanced Steam search failed: {e}")
            return None
    
    async def _llm_enhanced_cache_search(self, game_name: str) -> Optional[GameRequirements]:
        """Use LLM to intelligently search and interpret cache data."""
        try:
            if not self.llm_analyzer:
                return None
                
            self.logger.info(f"Using LLM for intelligent cache interpretation of '{game_name}'")
            
            # Get all available games from cache
            available_games = self.cache_source._cache.get('games', {})
            if not available_games:
                return None
            
            # Use LLM to interpret and match game requirements
            llm_result = await self.llm_analyzer.interpret_game_requirements(game_name, available_games)
            
            if llm_result and 'matched_game' in llm_result:
                matched_name = llm_result['matched_game']
                
                if matched_name in available_games:
                    game_data = available_games[matched_name]
                    minimum = game_data.get('minimum', {})
                    recommended = game_data.get('recommended', {})
                    
                    self.logger.info(f"LLM successfully matched '{game_name}' to '{matched_name}'")
                    
                    return GameRequirements(
                        game_name=game_name,  # Use original query name
                        minimum_cpu=minimum.get('processor', 'Unknown'),
                        minimum_gpu=minimum.get('graphics', 'Unknown'),
                        minimum_ram_gb=self._parse_ram_value(minimum.get('memory', '0')),
                        minimum_vram_gb=0,
                        minimum_storage_gb=self._parse_storage_value(minimum.get('storage', '0')),
                        minimum_directx=minimum.get('directx', 'DirectX 11'),
                        minimum_os=minimum.get('os', 'Windows 10'),
                        recommended_cpu=recommended.get('processor', 'Unknown'),
                        recommended_gpu=recommended.get('graphics', 'Unknown'),
                        recommended_ram_gb=self._parse_ram_value(recommended.get('memory', '0')),
                        recommended_vram_gb=0,
                        recommended_storage_gb=self._parse_storage_value(recommended.get('storage', '0')),
                        recommended_directx=recommended.get('directx', 'DirectX 12'),
                        recommended_os=recommended.get('os', 'Windows 11'),
                        source='Local Cache (LLM Enhanced)',
                        last_updated=str(int(time.time()))
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"LLM-enhanced cache search failed: {e}")
            return None
    
    async def _generate_game_name_variations(self, game_name: str) -> List[str]:
        """Generate intelligent game name variations using LLM."""
        try:
            if not self.llm_analyzer:
                return [game_name]
            
            # Create prompt for LLM to generate variations
            prompt = f"""
            Generate alternative names and variations for the game: "{game_name}"
            
            Include common variations like:
            - Roman numeral conversions (4 ↔ IV, 2 ↔ II)
            - Subtitle variations
            - Abbreviations and full names
            - Common misspellings
            - Regional name differences
            
            Return only the game names, one per line, maximum 5 variations.
            """
            
            response = await self.llm_analyzer.analyze_text(prompt)
            
            # Parse response into list of variations
            variations = [game_name]  # Always include original
            if response:
                lines = response.strip().split('\n')
                for line in lines:
                    variation = line.strip().strip('-').strip()
                    if variation and variation != game_name:
                        variations.append(variation)
            
            return variations[:6]  # Limit to 6 total variations
            
        except Exception as e:
            self.logger.error(f"Failed to generate game name variations: {e}")
            return [game_name]
    
    def _parse_ram_value(self, ram_str: str) -> int:
        """Parse RAM value from string to integer GB."""
        if not ram_str:
            return 0
        match = re.search(r'(\d+)', str(ram_str))
        return int(match.group(1)) if match else 0
    
    def _parse_storage_value(self, storage_str: str) -> int:
        """Parse storage value from string to integer GB."""
        if not storage_str:
            return 0
        match = re.search(r'(\d+\.?\d*)', str(storage_str))
        return int(float(match.group(1))) if match else 0

    async def _cache_requirements(self, requirements: GameRequirements):
        """Cache requirements locally."""
        try:
            self.cache_source.save_to_cache(requirements)
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