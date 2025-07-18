"""
Test Steam Scraping and LLM Evaluation Fixes
Comprehensive test to verify the Steam API integration and LLM evaluation work correctly.
"""

import unittest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from game_requirements_fetcher import GameRequirementsFetcher, GameRequirements, SteamAPISource
from canrun_engine import CanRunEngine
from rtx_llm_analyzer import GAssistLLMAnalyzer


class TestSteamScrapingFix(unittest.TestCase):
    """Test suite for Steam scraping and LLM evaluation fixes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_analyzer = Mock()
        self.mock_llm_analyzer.analyze_text = AsyncMock()
        self.mock_llm_analyzer.interpret_game_requirements = AsyncMock()
        
        # Mock Steam API responses
        self.mock_steam_app_details = {
            "1091500": {
                "success": True,
                "data": {
                    "name": "Cyberpunk 2077",
                    "pc_requirements": {
                        "minimum": "<strong>Minimum:</strong><br><ul><li><strong>OS:</strong> Windows 10 64-bit</li><li><strong>Processor:</strong> Intel Core i5-3570K or AMD FX-8310</li><li><strong>Memory:</strong> 8 GB RAM</li><li><strong>Graphics:</strong> NVIDIA GeForce GTX 970 or AMD Radeon RX 470</li><li><strong>DirectX:</strong> Version 12</li><li><strong>Storage:</strong> 70 GB available space</li></ul>",
                        "recommended": "<strong>Recommended:</strong><br><ul><li><strong>OS:</strong> Windows 10 64-bit</li><li><strong>Processor:</strong> Intel Core i7-4790 or AMD Ryzen 3 3200G</li><li><strong>Memory:</strong> 12 GB RAM</li><li><strong>Graphics:</strong> NVIDIA GeForce GTX 1060 6GB or AMD Radeon R9 Fury</li><li><strong>DirectX:</strong> Version 12</li><li><strong>Storage:</strong> 70 GB available space</li></ul>"
                    }
                }
            }
        }
    
    async def test_steam_api_bypass_removed(self):
        """Test that Steam API bypass has been removed and Steam is actually called."""
        fetcher = GameRequirementsFetcher(self.mock_llm_analyzer)
        
        with patch.object(fetcher.steam_source, 'fetch') as mock_steam_fetch:
            mock_steam_fetch.return_value = GameRequirements(
                game_name="Cyberpunk 2077",
                minimum_cpu="Intel Core i5-3570K",
                minimum_gpu="NVIDIA GeForce GTX 970",
                minimum_ram_gb=8,
                minimum_vram_gb=4,
                minimum_storage_gb=70,
                minimum_directx="DirectX 12",
                minimum_os="Windows 10 64-bit",
                recommended_cpu="Intel Core i7-4790",
                recommended_gpu="NVIDIA GeForce GTX 1060 6GB",
                recommended_ram_gb=12,
                recommended_vram_gb=6,
                recommended_storage_gb=70,
                recommended_directx="DirectX 12",
                recommended_os="Windows 10 64-bit",
                source="Steam API",
                last_updated="1234567890"
            )
            
            # Test that Steam API is actually called
            result = await fetcher.fetch_requirements("Cyberpunk 2077")
            
            # Verify Steam API was called
            mock_steam_fetch.assert_called_once_with("Cyberpunk 2077")
            
            # Verify result is correct
            self.assertIsNotNone(result)
            self.assertEqual(result.source, "Steam API")
            self.assertEqual(result.game_name, "Cyberpunk 2077")
            self.assertEqual(result.minimum_cpu, "Intel Core i5-3570K")
            
        print("✅ Steam API bypass removal test passed")
    
    async def test_llm_enhanced_steam_search(self):
        """Test LLM-enhanced Steam search functionality."""
        fetcher = GameRequirementsFetcher(self.mock_llm_analyzer)
        
        # Mock LLM to generate game variations
        self.mock_llm_analyzer.analyze_text.return_value = "Cyberpunk 2077\nCP2077\nCyberpunk\nCyberpunk: 2077"
        
        with patch.object(fetcher.steam_source, 'fetch') as mock_steam_fetch:
            # First call returns None, second call (with variation) returns result
            mock_steam_fetch.side_effect = [
                None,  # Original name fails
                GameRequirements(  # Variation succeeds
                    game_name="Cyberpunk 2077",
                    minimum_cpu="Intel Core i5-3570K",
                    minimum_gpu="NVIDIA GeForce GTX 970",
                    minimum_ram_gb=8,
                    minimum_vram_gb=4,
                    minimum_storage_gb=70,
                    source="Steam API",
                    last_updated="1234567890"
                )
            ]
            
            # Test LLM-enhanced search
            result = await fetcher._llm_enhanced_steam_search("Cyberpunk")
            
            # Verify LLM was called to generate variations
            self.mock_llm_analyzer.analyze_text.assert_called()
            
            # Verify result is correct
            self.assertIsNotNone(result)
            self.assertEqual(result.game_name, "Cyberpunk")  # Should use original query name
            
        print("✅ LLM-enhanced Steam search test passed")
    
    async def test_data_structure_fix(self):
        """Test that GameRequirements dataclass works correctly with cache."""
        fetcher = GameRequirementsFetcher(self.mock_llm_analyzer)
        
        # Create a GameRequirements object
        requirements = GameRequirements(
            game_name="Test Game",
            minimum_cpu="Intel i5",
            minimum_gpu="GTX 1060",
            minimum_ram_gb=8,
            minimum_vram_gb=4,
            minimum_storage_gb=50,
            minimum_directx="DirectX 11",
            minimum_os="Windows 10",
            recommended_cpu="Intel i7",
            recommended_gpu="GTX 1070",
            recommended_ram_gb=16,
            recommended_vram_gb=6,
            recommended_storage_gb=50,
            recommended_directx="DirectX 12",
            recommended_os="Windows 11",
            source="Test",
            last_updated="1234567890"
        )
        
        # Test that save_to_cache works without errors
        try:
            fetcher.cache_source.save_to_cache(requirements)
            print("✅ Data structure fix test passed - no errors saving to cache")
        except Exception as e:
            self.fail(f"save_to_cache failed with error: {e}")
    
    async def test_steam_search_methods(self):
        """Test multiple Steam search methods."""
        steam_source = SteamAPISource(self.mock_llm_analyzer)
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock successful response
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=[{"id": "1091500", "name": "Cyberpunk 2077"}])
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            # Test Steam store suggest search
            steam_id = await steam_source._search_steam_store_suggest("Cyberpunk 2077")
            
            self.assertEqual(steam_id, "1091500")
            
        print("✅ Steam search methods test passed")
    
    async def test_canrun_engine_integration(self):
        """Test full CanRun engine integration with Steam fixes."""
        with patch('src.privacy_aware_hardware_detector.PrivacyAwareHardwareDetector') as mock_detector:
            # Mock hardware specs
            mock_hardware = Mock()
            mock_hardware.is_nvidia_gpu = True
            mock_hardware.gpu_model = "RTX 3080"
            mock_hardware.cpu_model = "Intel i7-10700K"
            mock_hardware.ram_total_gb = 32
            mock_detector.return_value.get_hardware_specs = AsyncMock(return_value=mock_hardware)
            
            # Create engine with LLM enabled
            engine = CanRunEngine(enable_llm=True)
            
            # Verify LLM analyzer is passed to requirements fetcher
            self.assertIsNotNone(engine.requirements_fetcher.llm_analyzer)
            
        print("✅ CanRun engine integration test passed")
    
    def test_dynamic_performance_predictor_integration(self):
        """Test that the new dynamic performance predictor works correctly."""
        from dynamic_performance_predictor import DynamicPerformancePredictor
        
        predictor = DynamicPerformancePredictor()
        
        # Test RTX 3080 + i7-10700K system (should get A-tier)
        hardware_specs = {
            'gpu_model': 'RTX 3080',
            'cpu_model': 'Intel i7-10700K',
            'ram_total_gb': 16,
            'cpu_cores': 8,
            'cpu_threads': 16,
            'cpu_frequency': 3800,
            'gpu_vram_gb': 10
        }
        
        # Test performance assessment
        assessment = predictor.assess_performance(hardware_specs=hardware_specs)
        
        # Verify mid-high-end hardware gets good tier
        self.assertGreaterEqual(assessment.score, 70, "RTX 3080 + i7-10700K should get 70+ score")
        self.assertIn(assessment.tier.name, ['A', 'B', 'S'], "Should get good tier")
        self.assertIsNotNone(assessment.recommended_settings, "Should have settings recommendation")
        self.assertIsNotNone(assessment.recommended_resolution, "Should have resolution recommendation")
        
        print("✅ Dynamic performance predictor integration test passed")


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
for attr_name in dir(TestSteamScrapingFix):
    if attr_name.startswith('test_') and 'async' in attr_name:
        attr = getattr(TestSteamScrapingFix, attr_name)
        if asyncio.iscoroutinefunction(attr):
            setattr(TestSteamScrapingFix, attr_name, run_async_test(attr))


if __name__ == '__main__':
    print("🧪 Running Steam Scraping and LLM Evaluation Fix Tests")
    print("=" * 60)
    
    # Run the tests
    unittest.main(verbosity=2)