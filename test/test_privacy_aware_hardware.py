"""
Privacy-Aware Hardware Detection Tests
Tests for the privacy-focused hardware detection system with security validation.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import hashlib
import secrets

# Mock hardware detection libraries to avoid import issues
import sys
sys.modules['GPUtil'] = MagicMock()
sys.modules['cpuinfo'] = MagicMock()
sys.modules['pynvml'] = MagicMock()
sys.modules['winreg'] = MagicMock()
sys.modules['wmi'] = MagicMock()
sys.modules['rtx_llm_analyzer'] = MagicMock()

from canrun.src.privacy_aware_hardware_detector import (
    PrivacyAwareHardwareDetector,
    PrivacyAwareHardwareSpecs,
    PrivacyAwareCache
)


class TestPrivacyAwareHardwareSpecs(unittest.TestCase):
    """Test privacy-aware hardware specifications data structure."""
    
    def test_privacy_aware_specs_structure(self):
        """Test privacy-aware hardware specs structure validation."""
        # Test that the dataclass exists and has the expected structure
        self.assertTrue(hasattr(PrivacyAwareHardwareSpecs, '__dataclass_fields__'))
        
        # Test required fields exist
        fields = PrivacyAwareHardwareSpecs.__dataclass_fields__
        required_fields = ['gpu_model', 'gpu_vram_gb', 'cpu_cores', 'cpu_threads', 'ram_total_gb']
        
        for field in required_fields:
            self.assertIn(field, fields, f"Required field '{field}' missing from dataclass")
        
        # Verify no sensitive data fields exist in the structure
        sensitive_fields = ['serial_number', 'mac_address', 'user_name', 'computer_name']
        for field in sensitive_fields:
            self.assertNotIn(field, fields, f"Sensitive field '{field}' found in dataclass")


class TestPrivacyAwareCache(unittest.TestCase):
    """Test privacy-aware caching system."""
    
    def setUp(self):
        """Set up test cache."""
        # Cache now standardized to 15 minutes regardless of input parameter
        self.cache = PrivacyAwareCache()
    
    def test_cache_store_and_retrieve(self):
        """Test storing and retrieving cached data."""
        test_key = "test_key"
        test_value = {"test": "data"}
        
        # Store data
        self.cache.store(test_key, test_value)
        
        # Retrieve data
        retrieved_value = self.cache.get(test_key)
        self.assertEqual(retrieved_value, test_value)
    
    def test_cache_key_anonymization(self):
        """Test that cache keys are anonymized."""
        test_key = "sensitive_system_identifier"
        test_value = {"test": "data"}
        
        # Store data
        self.cache.store(test_key, test_value)
        
        # Verify key is anonymized in internal storage
        self.assertNotIn(test_key, self.cache.data.keys())
        
        # But can still retrieve with original key
        retrieved_value = self.cache.get(test_key)
        self.assertEqual(retrieved_value, test_value)
    
    def test_cache_expiration(self):
        """Test cache expiration functionality."""
        test_key = "test_key"
        test_value = {"test": "data"}
        
        # Cache is standardized to 15 minutes - test with manual cache expiry manipulation
        short_cache = PrivacyAwareCache()
        
        # Store data
        short_cache.store(test_key, test_value)
        
        # Verify data is there immediately
        self.assertEqual(short_cache.get(test_key), test_value)
        
        # Simulate time passing by manually setting old timestamp
        from datetime import datetime, timedelta
        old_time = datetime.now() - timedelta(hours=1)
        anonymous_key = short_cache._anonymize_key(test_key)
        if anonymous_key in short_cache.cache_timestamps:
            short_cache.cache_timestamps[anonymous_key] = old_time
        
        # Verify data is cleaned up
        self.assertIsNone(short_cache.get(test_key))
    
    def test_cache_cleanup(self):
        """Test automatic cache cleanup."""
        test_keys = ["key1", "key2", "key3"]
        test_value = {"test": "data"}
        
        # Store multiple items
        for key in test_keys:
            self.cache.store(key, test_value)
        
        # Verify all are stored
        for key in test_keys:
            self.assertEqual(self.cache.get(key), test_value)
        
        # Manually expire some items
        from datetime import datetime, timedelta
        old_time = datetime.now() - timedelta(hours=2)
        
        for key in test_keys[:2]:  # Expire first two
            anonymous_key = self.cache._anonymize_key(key)
            if anonymous_key in self.cache.cache_timestamps:
                self.cache.cache_timestamps[anonymous_key] = old_time
        
        # Access cache to trigger cleanup
        self.cache.get("key3")
        
        # Verify expired items are removed
        self.assertIsNone(self.cache.get("key1"))
        self.assertIsNone(self.cache.get("key2"))
        self.assertEqual(self.cache.get("key3"), test_value)


class TestPrivacyAwareHardwareDetector(unittest.TestCase):
    """Test privacy-aware hardware detector."""
    
    def setUp(self):
        """Set up test detector."""
        # GAssistLLMAnalyzer is now imported lazily, no need to mock
        self.detector = PrivacyAwareHardwareDetector()
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        self.assertIsInstance(self.detector.cache, PrivacyAwareCache)
        # LLM analyzer is now initialized lazily, so it starts as None
        self.assertIsNone(self.detector.llm_analyzer)
        
    def test_llm_analyzer_initialization(self):
        """Test LLM analyzer lazy initialization."""
        # Initially None
        self.assertIsNone(self.detector.llm_analyzer)
        
        # Test lazy initialization
        llm_analyzer = self.detector._get_llm_analyzer()
        
        # Should now be initialized (or still None if import fails gracefully)
        self.assertIsNotNone(self.detector.llm_analyzer)
        
        # Test that LLM analyzer methods are available
        if self.detector.llm_analyzer:
            self.assertTrue(hasattr(self.detector.llm_analyzer, 'model_available'))
    
    def test_gpu_name_cleaning(self):
        """Test GPU name cleaning."""
        test_cases = [
            ("NVIDIA GeForce RTX 4070", "RTX 4070"),
            ("GeForce RTX 4070", "RTX 4070"),
            ("NVIDIA RTX 4070 (some details)", "RTX 4070"),
            ("RTX 4070", "RTX 4070"),
        ]
        
        for input_name, expected_output in test_cases:
            with self.subTest(input_name=input_name):
                result = self.detector._clean_gpu_name(input_name)
                self.assertEqual(result, expected_output)
    
    def test_cpu_name_cleaning(self):
        """Test CPU name cleaning."""
        test_cases = [
            ("Intel Core i7-12700K @ 3.60GHz", "Intel Core i7-12700K"),
            ("AMD Ryzen 7 5800X 3.8GHz", "AMD Ryzen 7 5800X"),
            ("Intel Core i7-12700K", "Intel Core i7-12700K"),
        ]
        
        for input_name, expected_output in test_cases:
            with self.subTest(input_name=input_name):
                result = self.detector._clean_cpu_name(input_name)
                self.assertEqual(result, expected_output)
    
    def test_llm_hardware_analysis(self):
        """Test LLM-based hardware analysis."""
        with patch.object(self.detector, 'llm_analyzer') as mock_llm:
            mock_llm.model_available = True
            
            # Test GPU analysis
            result = self.detector._analyze_hardware_with_llm('gpu', 'RTX 4070')
            self.assertIsInstance(result, dict)
            
            # Test CPU analysis
            result = self.detector._analyze_hardware_with_llm('cpu', 'Intel i7-12700K')
            self.assertIsInstance(result, dict)
    
    def test_hardware_specs_parsing(self):
        """Test hardware specifications parsing."""
        # Test GPU parsing
        gpu_specs = self.detector._parse_hardware_specs('gpu', 'RTX 4070')
        self.assertIn('vram_gb', gpu_specs)
        self.assertEqual(gpu_specs['vram_gb'], 12)
        
        # Test CPU parsing
        cpu_specs = self.detector._parse_hardware_specs('cpu', 'Intel i7-12700K')
        self.assertIn('cores', cpu_specs)
        self.assertIn('threads', cpu_specs)
        
        # Test unknown GPU - should not provide fake values
        unknown_gpu = self.detector._parse_hardware_specs('gpu', 'Unknown Brand GPU X1000')
        self.assertNotIn('vram_gb', unknown_gpu)  # No fake values for unknown GPUs
    
    def test_no_misleading_defaults(self):
        """Test that no misleading default values are provided."""
        # Test that unknown hardware doesn't get fake specs
        unknown_gpu = self.detector._parse_hardware_specs('gpu', 'UnknownGPU9000')
        self.assertEqual(len(unknown_gpu), 0)  # Should be empty, not contain fake values
        
        unknown_cpu = self.detector._parse_hardware_specs('cpu', 'UnknownCPU9000')
        self.assertEqual(len(unknown_cpu), 0)  # Should be empty, not contain fake values
        
        # Test that known hardware gets real specs (not fake defaults)
        rtx_4070 = self.detector._parse_hardware_specs('gpu', 'RTX 4070')
        self.assertIn('vram_gb', rtx_4070)
        self.assertEqual(rtx_4070['vram_gb'], 12)  # Real spec, not fake default
        
        # Test that partial detection doesn't fill with fake values
        mystery_gpu = self.detector._parse_hardware_specs('gpu', 'NVIDIA Some New GPU 9999')
        self.assertEqual(len(mystery_gpu), 0)  # No fake specs for unknown models


class TestPrivacyCompliance(unittest.TestCase):
    """Test privacy compliance features."""
    
    def setUp(self):
        """Set up privacy compliance tests."""
        # GAssistLLMAnalyzer is now imported lazily, no need to mock
        self.detector = PrivacyAwareHardwareDetector()
    
    def test_cache_privacy(self):
        """Test cache privacy features."""
        # Test that cache keys are anonymized
        cache = PrivacyAwareCache()
        test_key = "sensitive_hardware_identifier"
        test_value = {"hardware": "data"}
        
        cache.store(test_key, test_value)
        
        # Verify original key is not stored
        self.assertNotIn(test_key, cache.data.keys())
        
        # Verify data can be retrieved with original key
        self.assertEqual(cache.get(test_key), test_value)
    
    def test_no_personal_identification(self):
        """Test that no personal identification data is collected."""
        # Test that dataclass structure doesn't include personal identifiers
        fields = PrivacyAwareHardwareSpecs.__dataclass_fields__
        
        # Verify no personal data fields in dataclass structure
        forbidden_fields = [
            'serial_number', 'mac_address', 'user_name', 'computer_name',
            'ip_address', 'hostname', 'domain', 'network_adapter'
        ]
        
        for field in forbidden_fields:
            self.assertNotIn(field, fields,
                           f"Personal identifier field '{field}' found in hardware specs structure")
        
        # Verify only gaming-relevant hardware data fields are defined
        expected_fields = ['gpu_model', 'gpu_vram_gb', 'cpu_cores', 'cpu_threads', 'ram_total_gb']
        for field in expected_fields:
            self.assertIn(field, fields, f"Expected field '{field}' missing from hardware specs structure")
    
    def test_temporal_data_limitation(self):
        """Test temporal limitation of data storage."""
        # Test that cache respects time limits - cache is standardized to 15 minutes
        short_cache = PrivacyAwareCache()
        
        test_data = {'test': 'data'}
        cache_key = "test_key"
        
        # Store data
        short_cache.store(cache_key, test_data)
        
        # Verify immediate retrieval works
        self.assertEqual(short_cache.get(cache_key), test_data)
        
        # Simulate time passing
        import time
        time.sleep(0.1)  # 100ms
        
        # Manually expire by setting old timestamp
        from datetime import datetime, timedelta
        old_time = datetime.now() - timedelta(hours=1)
        anonymous_key = short_cache._anonymize_key(cache_key)
        if anonymous_key in short_cache.cache_timestamps:
            short_cache.cache_timestamps[anonymous_key] = old_time
        
        # Data should be expired
        self.assertIsNone(short_cache.get(cache_key))


if __name__ == '__main__':
    unittest.main()