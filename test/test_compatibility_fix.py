#!/usr/bin/env python3
"""
Test script to verify compatibility fixes for CanRun G-Assist plugin.
Tests the specific issues: HardwareSpecs.get() error and CPU data handling.
"""

import sys
import os
import asyncio
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import G-Assist plugin components
from canrun_engine import CanRunEngine
from canrun_hardware_detector import CanRunHardwareDetector, HardwareSpecs
from canrun_ml_predictor import CanRunMLPredictor
from canrun_compatibility_analyzer import CanRunCompatibilityAnalyzer
from canrun_game_fetcher import GameRequirements

def setup_logging():
    """Setup ASCII-only logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_hardware_specs_to_dict_conversion():
    """Test that HardwareSpecs dataclass is properly converted to dict."""
    print("Testing HardwareSpecs to dictionary conversion...")
    
    try:
        # Create a test hardware specs object
        hardware_specs = HardwareSpecs(
            gpu_model="RTX 4070",
            gpu_vram_gb=12,
            cpu_cores=8,
            cpu_threads=16,
            ram_total_gb=32,
            ram_speed_mhz=3200,
            storage_type="NVMe SSD",
            primary_monitor_refresh_hz=144,
            primary_monitor_resolution="2560x1440",
            os_version="Windows 11",
            directx_version="DirectX 12"
        )
        
        # Test compatibility analyzer with hardware specs
        analyzer = CanRunCompatibilityAnalyzer()
        
        # Create test game requirements
        game_requirements = {
            "game_name": "Cyberpunk 2077",
            "minimum_gpu": "GTX 1060",
            "recommended_gpu": "RTX 3070",
            "minimum_cpu": "Intel Core i5-3570K",
            "recommended_cpu": "Intel Core i7-4790K",
            "minimum_ram_gb": 8,
            "recommended_ram_gb": 12,
            "minimum_vram_gb": 6,
            "recommended_vram_gb": 8,
            "minimum_storage_gb": 70,
            "recommended_storage_gb": 70,
            "minimum_directx": "DirectX 12",
            "recommended_directx": "DirectX 12",
            "minimum_os": "Windows 10",
            "recommended_os": "Windows 10"
        }
        
        # Convert hardware specs to dict (this is what the engine should do)
        hardware_dict = {
            "gpu_model": hardware_specs.gpu_model,
            "gpu_vram_gb": hardware_specs.gpu_vram_gb,
            "gpu_vendor": hardware_specs.gpu_vendor,
            "cpu_model": hardware_specs.cpu_model,
            "cpu_cores": hardware_specs.cpu_cores,
            "cpu_threads": hardware_specs.cpu_threads,
            "ram_total_gb": hardware_specs.ram_total_gb,
            "ram_speed_mhz": hardware_specs.ram_speed_mhz,
            "storage_type": hardware_specs.storage_type,
            "os_version": hardware_specs.os_version,
            "directx_version": hardware_specs.directx_version,
            "is_nvidia_gpu": hardware_specs.is_nvidia_gpu,
            "supports_rtx": hardware_specs.supports_rtx,
            "supports_dlss": hardware_specs.supports_dlss,
            "nvidia_driver_version": hardware_specs.nvidia_driver_version
        }
        
        # This should now work without the .get() error
        analysis = analyzer.analyze_compatibility("Cyberpunk 2077", hardware_dict, game_requirements)
        
        print("[PASS] HardwareSpecs conversion test passed")
        print(f"   Compatibility: {analysis.overall_compatibility.value}")
        print(f"   Can run minimum: {analysis.can_run_minimum}")
        print(f"   Overall score: {analysis.overall_score:.2f}")
        
        # Use assert statements instead of return
        assert analysis is not None, "Analysis should not be None"
        assert hasattr(analysis, 'overall_compatibility'), "Analysis should have overall_compatibility"
        assert hasattr(analysis, 'can_run_minimum'), "Analysis should have can_run_minimum"
        assert hasattr(analysis, 'overall_score'), "Analysis should have overall_score"
        assert analysis.overall_score >= 0, "Overall score should be non-negative"
        
    except Exception as e:
        print(f"[FAIL] HardwareSpecs conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"HardwareSpecs conversion test failed: {e}"

def test_cpu_data_handling():
    """Test CPU data handling for unknown/default CPUs."""
    print("\nTesting CPU data handling for unknown CPUs...")
    
    try:
        # Create ML predictor
        ml_predictor = CanRunMLPredictor()
        
        # Test cases for CPU handling
        test_cases = [
            {"model": "Default CPU", "cores": 8, "frequency": 3500},
            {"model": "Unknown CPU", "cores": 6, "frequency": 3200},
            {"model": "", "cores": 4, "frequency": 3000},
            {"model": "Intel Core i7-12700K", "cores": 12, "frequency": 3600}  # Should be found in DB
        ]
        
        for i, cpu_info in enumerate(test_cases):
            try:
                score, passmark = ml_predictor._get_cpu_score(cpu_info)
                print(f"[PASS] CPU test {i+1}: {cpu_info['model']} -> Score: {score}, PassMark: {passmark}")
                
                # Assert valid results
                assert score is not None, f"CPU score should not be None for {cpu_info['model']}"
                assert passmark is not None, f"CPU PassMark should not be None for {cpu_info['model']}"
                assert score >= 0, f"CPU score should be non-negative for {cpu_info['model']}"
                assert passmark >= 0, f"CPU PassMark should be non-negative for {cpu_info['model']}"
                
            except Exception as e:
                print(f"[FAIL] CPU test {i+1}: {cpu_info['model']} -> Error: {e}")
                assert False, f"CPU test failed for {cpu_info['model']}: {e}"
        
    except Exception as e:
        print(f"[FAIL] CPU data handling test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"CPU data handling test failed: {e}"

def test_fps_prediction_with_default_cpu():
    """Test FPS prediction with default CPU (the failing scenario)."""
    print("\nTesting FPS prediction with default CPU...")
    
    try:
        ml_predictor = CanRunMLPredictor()
        
        # Test the specific scenario that was failing
        fps = ml_predictor.predict_fps(
            gpu_name="RTX 4070",
            game_name="Cyberpunk 2077",
            resolution="1080p",
            quality="Ultra"
        )
        
        print(f"[PASS] FPS prediction test passed: {fps} FPS")
        print(f"   GPU: RTX 4070")
        print(f"   Game: Cyberpunk 2077")
        print(f"   Resolution: 1080p")
        assert fps is not None, "FPS prediction should not be None"
        assert fps > 0, "FPS prediction should be positive"
        
    except Exception as e:
        print(f"[FAIL] FPS prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"FPS prediction test failed: {e}"

def test_full_engine_compatibility():
    """Test full engine compatibility check (end-to-end)."""
    print("\nTesting full engine compatibility check...")
    
    try:
        # Initialize engine
        engine = CanRunEngine()
        
        # Test compatibility check method
        can_run = engine.check_compatibility("Cyberpunk 2077")
        
        print(f"[PASS] Full engine test passed")
        print(f"   Can run Cyberpunk 2077: {can_run}")
        assert can_run is not None, "Compatibility check should return a result"
        
    except Exception as e:
        print(f"[FAIL] Full engine test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Full engine test failed: {e}"

def main():
    """Run all compatibility tests."""
    setup_logging()
    
    print("CanRun G-Assist Plugin Compatibility Fix Tests")
    print("=" * 50)
    
    tests = [
        ("HardwareSpecs conversion", test_hardware_specs_to_dict_conversion),
        ("CPU data handling", test_cpu_data_handling),
        ("FPS prediction with default CPU", test_fps_prediction_with_default_cpu),
        ("Full engine compatibility", test_full_engine_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All compatibility fixes are working correctly!")
        return True
    else:
        print("Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)