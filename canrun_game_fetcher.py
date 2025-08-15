"""
CanRun Game Fetcher - Standalone module for NVIDIA G-Assist Plugin
Fetches game requirements from Steam API with comprehensive parsing.
Flattened structure with all dependencies bundled for PyInstaller compatibility.
"""

import json
import logging
import asyncio
import aiohttp
import re
import time
import sys
import os
from typing import Dict, List, Union, Optional, Any
from pathlib import Path
from dataclasses import dataclass


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable - use temporary extraction path
        base_path = sys._MEIPASS
        data_path = os.path.join(base_path, relative_path)
        return data_path
    else:
        # Running as normal Python script - use relative path from project root
        base_path = Path(__file__).parent
        return os.path.join(base_path, relative_path)


@dataclass
class GameRequirements:
    """
    Data class for storing game requirements fetched from Steam API.
    Supports the S-A-B-C-D-F tier system for performance prediction.
    """
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
    steam_api_name: str = ""  # Actual name returned by Steam API for reference


class CanRunGameFetcher:
    """
    Main game requirements fetcher using Steam API exclusively.
    No caching - always fetches fresh data for accurate performance calculations.
    Data loaded once in constructor for efficiency.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load static data once in constructor (following user's efficiency pattern)
        self._load_static_data()
        
        self.logger.info("CanRun game fetcher initialized")
    
    def _load_static_data(self):
        """Load all static parsing data once (following user's pattern)."""
        
        # Steam API endpoints for comprehensive game search (loaded once)
        self.base_url = "https://store.steampowered.com/api"
        self.search_url = "https://steamcommunity.com/actions/SearchApps"
        self.store_search_url = "https://store.steampowered.com/search/suggest"
        
        # Regex patterns to extract specific requirement fields (loaded once)
        self.requirement_patterns = {
            'os': r'OS:\s*([^<>\n]*?)(?=\s*(?:Processor|Memory|Graphics|DirectX|Storage|Sound|Additional|$))',
            'processor': r'Processor:\s*([^<>\n]*?)(?=\s*(?:Memory|Graphics|DirectX|Storage|Sound|Additional|$))',
            'memory': r'Memory:\s*([^<>\n]*?)(?=\s*(?:Graphics|DirectX|Storage|Sound|Additional|$))',
            'graphics': r'Graphics:\s*([^<>\n]*?)(?=\s*(?:DirectX|Storage|Sound|Additional|$))',
            'directx': r'DirectX:\s*([^<>\n]*?)(?=\s*(?:Storage|Sound|Additional|$))',
            'storage': r'Storage:\s*([^<>\n]*?)(?=\s*(?:Sound|Additional|$))',
            'sound': r'Sound Card:\s*([^<>\n]*?)(?=\s*(?:Additional|$))'
        }
        
        # App ID extraction patterns for various Steam responses (loaded once)
        self.app_id_patterns = [
            r'data-ds-appid="(\d+)"',
            r'"appid":\s*(\d+)',
            r'app/(\d+)/',
            r'appid=(\d+)'
        ]
        
        # Basic VRAM fallbacks (detailed VRAM estimation handled by hardware detector)
        self.vram_fallbacks = {
            'rtx': 8, 'gtx': 4, 'amd': 6, 'radeon': 6
        }

    async def fetch_requirements(self, game_name: str) -> Optional[GameRequirements]:
        """
        Fetch game requirements from Steam API only.
        Always returns fresh data - no caching involved.
        """
        try:
            # Direct Steam API call with timeout for G-Assist compatibility
            steam_requirements = await asyncio.wait_for(
                self._fetch_from_steam(game_name),
                timeout=15.0
            )
            
            if steam_requirements:
                # Preserve both original query and Steam API name for reference
                steam_api_name = steam_requirements.game_name
                steam_requirements.steam_api_name = steam_api_name
                steam_requirements.game_name = game_name  # Use original user query
                
                self.logger.info(f"Successfully fetched requirements for '{game_name}' from Steam API")
                return steam_requirements
            
            self.logger.warning(f"No requirements found for '{game_name}' in Steam API")
            return None

        except Exception as e:
            self.logger.error(f"Steam API fetch failed for '{game_name}': {e}")
            return None

    async def _fetch_from_steam(self, game_name: str) -> Optional[GameRequirements]:
        """
        Fetch game requirements from Steam API using multiple search strategies.
        Returns fresh data every time - no caching involved.
        """
        try:
            # Preserve number information for accurate game matching (e.g., "Diablo 4" vs "Diablo IV")
            has_number = any(c.isdigit() for c in game_name)
            
            # Step 1: Search for the game to get Steam App ID
            steam_id = await self._search_game(game_name)
            if not steam_id:
                self.logger.warning(f"No Steam ID found for '{game_name}'")
                return None
            
            # Step 2: Fetch detailed app information using the Steam ID
            app_info = await self._get_app_info(steam_id)
            if not app_info:
                self.logger.warning(f"No app info found for Steam ID {steam_id}")
                return None
            
            # Step 3: Parse requirements from Steam's structured data
            requirements = self._parse_requirements(app_info, game_name)
            
            # Preserve original query format if it contained numbers
            if has_number and requirements:
                requirements.game_name = game_name
                
            return requirements
            
        except Exception as e:
            self.logger.error(f"Steam API fetch failed for {game_name}: {e}")
            return None

    async def _search_game(self, game_name: str) -> Optional[str]:
        """
        Search for a game using multiple Steam API endpoints for maximum success rate.
        Returns the Steam App ID if found, None otherwise.
        """
        self.logger.debug(f"Searching Steam for game: {game_name}")
        
        # Multiple search strategies ordered by reliability and speed
        search_methods = [
            self._search_steam_store_suggest,    # Fastest, most accurate for exact matches
            self._search_steam_community,        # Good for partial matches
            self._search_steam_store_direct      # Fallback for difficult cases
        ]
        
        # Try each search method until one succeeds
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
        """
        Search using Steam Store suggest API - fastest and most accurate method.
        This endpoint is used by Steam's own search suggestions.
        """
        try:
            # Short timeout for G-Assist compatibility
            timeout = aiohttp.ClientTimeout(total=5, connect=3)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                params = {
                    'term': game_name,
                    'f': 'games',        # Filter to games only
                    'cc': 'US',          # Country code for consistent results
                    'l': 'english'       # Language for parsing
                }
                
                async with session.get(self.store_search_url, params=params) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '').lower()
                        
                        if 'application/json' in content_type:
                            # Parse JSON response (preferred format)
                            try:
                                data = await response.json()
                                if isinstance(data, list) and len(data) > 0:
                                    app_data = data[0]  # Take first (best) match
                                    if 'id' in app_data:
                                        return str(app_data['id'])
                                    elif 'appid' in app_data:
                                        return str(app_data['appid'])
                            except json.JSONDecodeError:
                                return None
                        else:
                            # Parse HTML response as fallback using pre-loaded patterns
                            text = await response.text()
                            for pattern in self.app_id_patterns:
                                match = re.search(pattern, text)
                                if match:
                                    return match.group(1)
                        
        except Exception as e:
            self.logger.debug(f"Steam store suggest search failed for '{game_name}': {e}")
        
        return None

    async def _search_steam_community(self, game_name: str) -> Optional[str]:
        """
        Search using Steam Community API - good for partial matches and fuzzy search.
        """
        try:
            timeout = aiohttp.ClientTimeout(total=5, connect=3)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                params = {
                    'text': game_name,
                    'max_results': 10
                }
                
                async with session.get(self.search_url, params=params) as response:
                    if response.status == 200:
                        try:
                            # Try JSON parsing first
                            data = await response.json()
                            if isinstance(data, list) and len(data) > 0:
                                app_data = data[0]
                                if 'appid' in app_data:
                                    return str(app_data['appid'])
                        except Exception:
                            # Fallback to HTML parsing using pre-loaded patterns
                            text = await response.text()
                            for pattern in self.app_id_patterns:
                                match = re.search(pattern, text)
                                if match:
                                    return match.group(1)
                                    
        except Exception as e:
            self.logger.debug(f"Steam community search failed for '{game_name}': {e}")
        
        return None

    async def _search_steam_store_direct(self, game_name: str) -> Optional[str]:
        """
        Direct search on Steam store page - most comprehensive but slowest method.
        Used as final fallback when other methods fail.
        """
        try:
            # Construct direct search URL
            search_url = f"https://store.steampowered.com/search/?term={game_name.replace(' ', '+')}"
            
            timeout = aiohttp.ClientTimeout(total=8, connect=3)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(search_url) as response:
                    if response.status == 200:
                        text = await response.text()
                        
                        # Extract app ID using pre-loaded patterns
                        for pattern in self.app_id_patterns:
                            match = re.search(pattern, text)
                            if match:
                                return match.group(1)
                                
        except Exception as e:
            self.logger.debug(f"Steam store direct search failed for '{game_name}': {e}")
        
        return None

    async def _get_app_info(self, steam_id: str) -> Optional[Dict]:
        """
        Get detailed app information from Steam Store API.
        Includes retry logic for reliability and rate limiting handling.
        """
        try:
            timeout = aiohttp.ClientTimeout(total=8)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{self.base_url}/appdetails"
                params = {
                    'appids': steam_id,
                    'cc': 'US',          # Country code for consistent pricing/content
                    'l': 'english'       # Language for requirements text
                }
                
                # Retry logic for handling temporary failures and rate limits
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
                                wait_time = 2 ** attempt  # Exponential backoff
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
        """
        Parse requirements from Steam app info into structured GameRequirements object.
        Handles both minimum and recommended specifications for the S-A-B-C-D-F tier system.
        """
        try:
            pc_requirements = app_info.get('pc_requirements', {})
            if not pc_requirements:
                self.logger.debug(f"No PC requirements found for {game_name}")
                return None
            
            # Parse minimum and recommended requirements separately
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
        """
        Parse requirement text from Steam into structured format.
        Handles HTML tags, special characters, and various formatting styles.
        """
        requirements = {}
        
        # Clean HTML tags and normalize whitespace
        clean_text = re.sub(r'<[^>]+>', '\n', text)
        clean_text = re.sub(r'&nbsp;', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Apply each pattern using pre-loaded patterns to extract requirements
        for key, pattern in self.requirement_patterns.items():
            match = re.search(pattern, clean_text, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                # Clean up trailing punctuation
                value = re.sub(r'[.,:;]+$', '', value).strip()
                if value:
                    requirements[key] = value
        
        # Fallback parsing if structured patterns fail
        if not requirements:
            self._fallback_requirement_parsing(clean_text, requirements)
        
        return requirements

    def _fallback_requirement_parsing(self, clean_text: str, requirements: Dict[str, str]):
        """
        Fallback method for parsing requirements when structured patterns fail.
        Uses line-by-line analysis to extract key-value pairs.
        """
        lines = re.split(r'[<>]|(?:\s*(?:Processor|Memory|Graphics|DirectX|Storage|Sound)\s*:)', clean_text)
        
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

    def _dict_to_dataclass_fields(self, minimum: Dict[str, str], recommended: Dict[str, str]) -> Dict[str, any]:
        """
        Convert parsed requirement dictionaries to GameRequirements dataclass fields.
        Implements intelligent parsing for the S-A-B-C-D-F tier system calculations.
        """
        def parse_storage(value: str) -> int:
            """Parse storage value like '25 GB' to integer GB."""
            if not value:
                return 0
            # Extract number from strings like "25 GB", "2.5 GB", etc.
            match = re.search(r'(\d+\.?\d*)', str(value))
            return int(float(match.group(1))) if match else 0
        
        def parse_ram(value: str) -> int:
            """
            Parse RAM value with proper MB/GB unit handling.
            Critical for accurate performance tier calculations.
            """
            if not value:
                return 0
                
            value_upper = str(value).upper()
            
            # Handle MB specifications (convert to GB)
            if 'MB' in value_upper:
                mb_match = re.search(r'(\d+\.?\d*)\s*MB', value_upper)
                if mb_match:
                    mb_value = float(mb_match.group(1))
                    if mb_value < 512:
                        return 1  # Minimum 1GB for very small values
                    else:
                        return max(1, int(mb_value / 1024))  # Convert MB to GB
            
            # Handle GB specifications
            gb_match = re.search(r'(\d+\.?\d*)\s*G?B?', value_upper)
            if gb_match:
                return int(float(gb_match.group(1)))
                
            return 0

        def estimate_vram_from_gpu(gpu_str: str) -> int:
            """
            Estimate VRAM from GPU model string using pre-loaded database.
            Based on known GPU specifications and market data.
            """
            if not gpu_str:
                return 2  # Conservative default
            
            gpu_lower = gpu_str.lower()
            
            # Look for explicit VRAM mention first
            vram_match = re.search(r'(\d+)\s*gb', gpu_lower)
            if vram_match:
                return int(vram_match.group(1))
            
            # Use pre-loaded VRAM fallbacks for common GPU types
            for gpu_type, vram in self.vram_fallbacks.items():
                if gpu_type in gpu_lower:
                    return vram
            
            return 2  # Conservative fallback
        
        # Calculate VRAM estimates for both minimum and recommended using pre-loaded data
        min_vram = estimate_vram_from_gpu(minimum.get('graphics', ''))
        rec_vram = estimate_vram_from_gpu(recommended.get('graphics', ''))
        
        # Ensure recommended VRAM is at least as high as minimum
        if rec_vram < min_vram:
            rec_vram = min_vram
        
        return {
            'minimum_cpu': minimum.get('processor', 'Unknown'),
            'minimum_gpu': minimum.get('graphics', 'Unknown'),
            'minimum_ram_gb': parse_ram(minimum.get('memory', '0')),
            'minimum_vram_gb': min_vram,
            'minimum_storage_gb': parse_storage(minimum.get('storage', '0')),
            'minimum_directx': minimum.get('directx', 'DirectX 11'),
            'minimum_os': minimum.get('os', 'Windows 10'),
            'recommended_cpu': recommended.get('processor', 'Unknown'),
            'recommended_gpu': recommended.get('graphics', 'Unknown'),
            'recommended_ram_gb': parse_ram(recommended.get('memory', '0')),
            'recommended_vram_gb': rec_vram,
            'recommended_storage_gb': parse_storage(recommended.get('storage', '0')),
            'recommended_directx': recommended.get('directx', 'DirectX 12'),
            'recommended_os': recommended.get('os', 'Windows 11')
        }

    async def batch_fetch(self, game_names: List[str]) -> Dict[str, Optional[GameRequirements]]:
        """
        Fetch requirements for multiple games concurrently.
        Each game gets fresh Steam API data - no caching.
        """
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

    def quick_requirements_summary(self, requirements: GameRequirements) -> Dict[str, Any]:
        """Quick summary of requirements for G-Assist responses."""
        if not requirements:
            return {}
            
        return {
            'game_name': requirements.game_name,
            'min_ram': f"{requirements.minimum_ram_gb}GB",
            'min_vram': f"{requirements.minimum_vram_gb}GB", 
            'min_gpu': requirements.minimum_gpu,
            'min_cpu': requirements.minimum_cpu,
            'rec_ram': f"{requirements.recommended_ram_gb}GB" if requirements.recommended_ram_gb > 0 else "N/A",
            'rec_vram': f"{requirements.recommended_vram_gb}GB" if requirements.recommended_vram_gb > 0 else "N/A",
            'source': requirements.source
        }


async def test_game_fetcher():
    """Test the game requirements fetcher with Steam API only."""
    fetcher = CanRunGameFetcher()
    
    print("Testing Steam API-only game fetch...")
    requirements = await fetcher.fetch_requirements("Cyberpunk 2077")
    if requirements:
        summary = fetcher.quick_requirements_summary(requirements)
        print(f"Game: {summary['game_name']}")
        print(f"Source: {summary['source']}")
        print(f"Min GPU: {summary['min_gpu']}")
        print(f"Min RAM: {summary['min_ram']}")
        print(f"Min VRAM: {summary['min_vram']}")
    else:
        print("No requirements found")


if __name__ == "__main__":
    asyncio.run(test_game_fetcher())