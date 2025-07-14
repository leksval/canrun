"""
Steam API Integration Tests for CanRun
Tests Steam API integration for game requirements fetching.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os
import aiohttp
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from game_requirements_fetcher import SteamAPISource, GameRequirements, GameRequirementsFetcher


class TestSteamAPIIntegration(unittest.TestCase):
    """Test suite for Steam API integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.steam_source = SteamAPISource()
        self.fetcher = GameRequirementsFetcher()
        
        # Mock Steam API responses
        self.mock_steam_search_response = '''
        <html>
        <div class="search_result" data-ds-appid="1091500">
            <div class="responsive_search_name_combined">
                <div class="col search_name ellipsis">
                    <span class="title">Cyberpunk 2077</span>
                </div>
            </div>
        </div>
        </html>
        '''
        
        self.mock_steam_app_details = {
            "1091500": {
                "success": True,
                "data": {
                    "name": "Cyberpunk 2077",
                    "pc_requirements": {
                        "minimum": "<strong>Minimum:</strong><br><ul class=\"bb_ul\"><li><strong>Requires a 64-bit processor and operating system</strong></li><li><strong>OS:</strong> Windows 10 64-bit</li><li><strong>Processor:</strong> Intel Core i5-3570K or AMD FX-8310</li><li><strong>Memory:</strong> 8 GB RAM</li><li><strong>Graphics:</strong> NVIDIA GeForce GTX 970 or AMD Radeon RX 470</li><li><strong>DirectX:</strong> Version 12</li><li><strong>Storage:</strong> 70 GB available space</li></ul>",
                        "recommended": "<strong>Recommended:</strong><br><ul class=\"bb_ul\"><li><strong>Requires a 64-bit processor and operating system</strong></li><li><strong>OS:</strong> Windows 10 64-bit</li><li><strong>Processor:</strong> Intel Core i7-4790 or AMD Ryzen 3 3200G</li><li><strong>Memory:</strong> 12 GB RAM</li><li><strong>Graphics:</strong> NVIDIA GeForce GTX 1060 6GB or AMD Radeon R9 Fury</li><li><strong>DirectX:</strong> Version 12</li><li><strong>Storage:</strong> 70 GB available space</li></ul>"
                    }
                }
            }
        }
        
        self.mock_invalid_app_details = {
            "999999": {
                "success": False
            }
        }
    
    async def test_steam_id_search_success(self):
        """Test successful Steam ID search."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value=self.mock_steam_search_response)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            steam_id = await self.steam_source._search_game("Cyberpunk 2077")
            self.assertEqual(steam_id, "1091500")
    
    async def test_steam_id_search_failure(self):
        """Test Steam ID search failure."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 404
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            steam_id = await self.steam_source._search_game("NonExistentGame")
            self.assertIsNone(steam_id)
    
    async def test_steam_app_info_success(self):
        """Test successful Steam app info retrieval."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=self.mock_steam_app_details)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            app_info = await self.steam_source._get_app_info("1091500")
            self.assertIsNotNone(app_info)
            self.assertEqual(app_info["name"], "Cyberpunk 2077")
            self.assertIn("pc_requirements", app_info)
    
    async def test_steam_app_info_failure(self):
        """Test Steam app info retrieval failure."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=self.mock_invalid_app_details)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            app_info = await self.steam_source._get_app_info("999999")
            self.assertIsNone(app_info)
    
    async def test_steam_requirements_parsing(self):
        """Test parsing Steam requirements."""
        app_info = self.mock_steam_app_details["1091500"]["data"]
        requirements = self.steam_source._parse_requirements(app_info, "Cyberpunk 2077")
        
        self.assertIsNotNone(requirements)
        self.assertEqual(requirements.game_name, "Cyberpunk 2077")
        self.assertEqual(requirements.source, "Steam API")
        
        # Check minimum requirements
        self.assertIn("os", requirements.minimum)
        self.assertIn("processor", requirements.minimum)
        self.assertIn("memory", requirements.minimum)
        self.assertIn("graphics", requirements.minimum)
        self.assertIn("storage", requirements.minimum)
        
        # Check recommended requirements
        self.assertIn("os", requirements.recommended)
        self.assertIn("processor", requirements.recommended)
        self.assertIn("memory", requirements.recommended)
        self.assertIn("graphics", requirements.recommended)
        self.assertIn("storage", requirements.recommended)
        
        # Verify specific values
        self.assertEqual(requirements.minimum["os"], "Windows 10 64-bit")
        self.assertEqual(requirements.minimum["memory"], "8 GB RAM")
        self.assertEqual(requirements.recommended["memory"], "12 GB RAM")
    
    async def test_steam_requirements_parsing_rtx_support(self):
        """Test parsing Steam requirements with RTX support detection."""
        app_info = self.mock_steam_app_details["1091500"]["data"]
        requirements = self.steam_source._parse_requirements(app_info, "Cyberpunk 2077")
        
        # Check for RTX support in graphics requirements
        min_gpu = requirements.minimum.get("graphics", "")
        rec_gpu = requirements.recommended.get("graphics", "")
        
        # Should detect NVIDIA GPUs
        self.assertIn("NVIDIA", min_gpu)
        self.assertIn("NVIDIA", rec_gpu)
        
        # Should detect specific GPU models
        self.assertIn("GTX 970", min_gpu)
        self.assertIn("GTX 1060", rec_gpu)
    
    async def test_steam_api_error_handling(self):
        """Test Steam API error handling."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Test connection error
            mock_session.return_value.__aenter__.return_value.get.side_effect = aiohttp.ClientError("Connection failed")
            
            steam_id = await self.steam_source._get_steam_id("Test Game")
            self.assertIsNone(steam_id)
            
            app_info = await self.steam_source._get_app_info("123456")
            self.assertIsNone(app_info)
    
    async def test_steam_api_rate_limiting(self):
        """Test Steam API rate limiting handling."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Test rate limiting response
            mock_response = Mock()
            mock_response.status = 429  # Too Many Requests
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            steam_id = await self.steam_source._get_steam_id("Test Game")
            self.assertIsNone(steam_id)
    
    async def test_game_requirements_fetcher_steam_integration(self):
        """Test GameRequirementsFetcher with Steam API integration."""
        with patch.object(self.steam_source, '_search_game', return_value="1091500"):
            with patch.object(self.steam_source, '_get_app_info', return_value=self.mock_steam_app_details["1091500"]["data"]):
                
                # Add Steam source to fetcher
                self.fetcher.sources = [self.steam_source]
                
                requirements = await self.fetcher.fetch_requirements("Cyberpunk 2077")
                
                self.assertIsNotNone(requirements)
                self.assertEqual(requirements.game_name, "Cyberpunk 2077")
                self.assertEqual(requirements.source, "Steam API")
                
                # Verify requirements structure
                self.assertIsInstance(requirements.minimum, dict)
                self.assertIsInstance(requirements.recommended, dict)
    
    async def test_steam_api_fallback_behavior(self):
        """Test Steam API fallback behavior when Steam fails."""
        with patch.object(self.steam_source, '_search_game', return_value=None):
            requirements = await self.steam_source.fetch("Cyberpunk 2077")
            self.assertIsNone(requirements)
    
    async def test_steam_api_improved_parsing(self):
        """Test improved Steam API requirements parsing."""
        mock_html_requirements = {
            "minimum": "<strong>Minimum:</strong><br><ul><li><strong>OS:</strong> Windows 10 64-bit</li><li><strong>Processor:</strong> Intel Core i5-3570K</li><li><strong>Memory:</strong> 8 GB RAM</li><li><strong>Graphics:</strong> NVIDIA GeForce GTX 970</li><li><strong>DirectX:</strong> Version 12</li><li><strong>Storage:</strong> 70 GB available space</li></ul>",
            "recommended": "<strong>Recommended:</strong><br><ul><li><strong>OS:</strong> Windows 10 64-bit</li><li><strong>Processor:</strong> Intel Core i7-4790</li><li><strong>Memory:</strong> 12 GB RAM</li><li><strong>Graphics:</strong> NVIDIA GeForce GTX 1060</li><li><strong>DirectX:</strong> Version 12</li><li><strong>Storage:</strong> 70 GB available space</li></ul>"
        }
        
        app_info = {"name": "Test Game", "pc_requirements": mock_html_requirements}
        requirements = self.steam_source._parse_requirements(app_info, "Test Game")
        
        self.assertIsNotNone(requirements)
        self.assertEqual(requirements.game_name, "Test Game")
        
        # Test that parsing correctly separates requirements
        self.assertEqual(requirements.minimum["os"], "Windows 10 64-bit")
        self.assertEqual(requirements.minimum["processor"], "Intel Core i5-3570K")
        self.assertEqual(requirements.minimum["memory"], "8 GB RAM")
        self.assertEqual(requirements.minimum["graphics"], "NVIDIA GeForce GTX 970")
        
        self.assertEqual(requirements.recommended["processor"], "Intel Core i7-4790")
        self.assertEqual(requirements.recommended["memory"], "12 GB RAM")
    
    async def test_steam_api_fallback_to_cache(self):
        """Test fallback mechanism from Steam API to local cache."""
        # Mock Steam API failure
        with patch.object(self.steam_source, '_search_game', side_effect=Exception("Steam API failed")):
            # Mock cache hit
            with patch.object(self.fetcher.sources[1], 'fetch') as mock_cache:
                mock_cache.return_value = GameRequirements(
                    game_name="Cyberpunk 2077",
                    minimum={"os": "Windows 10", "processor": "Intel i5"},
                    recommended={"os": "Windows 10", "processor": "Intel i7"},
                    source="Local Cache",
                    last_updated="123456789"
                )
                
                requirements = await self.fetcher.fetch_requirements("Cyberpunk 2077")
                
                self.assertIsNotNone(requirements)
                self.assertEqual(requirements.source, "Local Cache")
                self.assertEqual(requirements.game_name, "Cyberpunk 2077")
    
    async def test_steam_api_retry_mechanism(self):
        """Test Steam API retry mechanism for rate limiting."""
        with patch('aiohttp.ClientSession') as mock_session:
            # First two attempts return 429 (rate limited), third succeeds
            mock_responses = [
                Mock(status=429),
                Mock(status=429),
                Mock(status=200, json=AsyncMock(return_value=self.mock_steam_app_details))
            ]
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_responses[0]
            
            # This should eventually succeed after retries
            with patch('asyncio.sleep'):  # Mock sleep to speed up test
                app_info = await self.steam_source._get_app_info("1091500")
                
                # Should have retried and eventually succeeded
                self.assertIsNotNone(app_info)
    
    async def test_steam_api_malformed_response(self):
        """Test Steam API with malformed response."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Test malformed JSON response
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(side_effect=ValueError("Invalid JSON"))
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            app_info = await self.steam_source._get_app_info("1091500")
            self.assertIsNone(app_info)
    
    async def test_steam_requirements_parsing_edge_cases(self):
        """Test Steam requirements parsing edge cases."""
        # Test empty requirements
        empty_app_info = {"name": "Test Game", "pc_requirements": {}}
        requirements = self.steam_source._parse_requirements(empty_app_info, "Test Game")
        self.assertIsNone(requirements)
        
        # Test missing pc_requirements
        no_pc_app_info = {"name": "Test Game"}
        requirements = self.steam_source._parse_requirements(no_pc_app_info, "Test Game")
        self.assertIsNone(requirements)
        
        # Test malformed requirements text
        malformed_app_info = {
            "name": "Test Game",
            "pc_requirements": {
                "minimum": "Invalid HTML format",
                "recommended": "Also invalid"
            }
        }
        requirements = self.steam_source._parse_requirements(malformed_app_info, "Test Game")
        self.assertIsNotNone(requirements)  # Should still create basic structure
    
    async def test_steam_api_timeout_handling(self):
        """Test Steam API timeout handling."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Test timeout error
            mock_session.return_value.__aenter__.return_value.get.side_effect = asyncio.TimeoutError("Request timeout")
            
            steam_id = await self.steam_source._get_steam_id("Test Game")
            self.assertIsNone(steam_id)
            
            app_info = await self.steam_source._get_app_info("123456")
            self.assertIsNone(app_info)
    
    async def test_steam_api_performance_metrics(self):
        """Test Steam API performance metrics."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=self.mock_steam_app_details)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            import time
            start_time = time.time()
            
            app_info = await self.steam_source._get_app_info("1091500")
            
            end_time = time.time()
            response_time = end_time - start_time
            
            self.assertIsNotNone(app_info)
            self.assertLess(response_time, 5.0)  # Should respond within 5 seconds
    
    def test_steam_api_initialization(self):
        """Test Steam API source initialization."""
        source = SteamAPISource()
        self.assertIsNotNone(source)
        self.assertEqual(source.base_url, "https://store.steampowered.com/api")
        self.assertIsNotNone(source.logger)


# Test runner for async tests
def run_async_test(test_func):
    """Helper to run async tests."""
    def wrapper(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(test_func(self))
        finally:
            loop.close()
    return wrapper


# Apply async test runner to all async test methods
for attr_name in dir(TestSteamAPIIntegration):
    if attr_name.startswith('test_') and 'async' in attr_name:
        attr = getattr(TestSteamAPIIntegration, attr_name)
        if asyncio.iscoroutinefunction(attr):
            setattr(TestSteamAPIIntegration, attr_name, run_async_test(attr))


if __name__ == '__main__':
    unittest.main()