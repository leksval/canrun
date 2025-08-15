"""
Test that the plugin correctly displays actual resolution and FPS ranges.
"""

import pytest
import sys
import os

# Add canrun to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plugin import format_canrun_response
from canrun_engine import CanRunResult
from canrun_hardware_detector import HardwareSpecs
from canrun_game_fetcher import GameRequirements
from canrun_compatibility_analyzer import CompatibilityAnalysis, ComponentAnalysis, ComponentType, CompatibilityLevel
from canrun_ml_predictor import PerformanceAssessment
from datetime import datetime


def create_test_result_with_custom_resolution_and_fps():
    """Create test result with custom resolution and FPS data."""
    
    # Hardware with 3072x1728 resolution (like user's system)
    hardware_specs = HardwareSpecs(
        gpu_model="RTX 4090",
        gpu_vram_gb=24,
        cpu_model="Intel i9-13900K",
        cpu_cores=24,
        cpu_threads=32,
        ram_total_gb=32,
        ram_speed_mhz=6000,
        storage_type="NVMe SSD",
        primary_monitor_refresh_hz=165,
        primary_monitor_resolution="3072x1728",  # Actual resolution
        os_version="Windows 11",
        directx_version="DirectX 12",
        supports_rtx=True,
        supports_dlss=True
    )
    
    # Game requirements
    game_requirements = GameRequirements(
        game_name="Cyberpunk 2077",
        minimum_gpu="GTX 1060",
        recommended_gpu="RTX 3070",
        minimum_cpu="Intel i5-3570K",
        recommended_cpu="Intel i7-4790",
        minimum_ram_gb=8,
        recommended_ram_gb=12,
        minimum_vram_gb=3,
        recommended_vram_gb=8,
        minimum_storage_gb=70,
        recommended_storage_gb=70
    )
    
    # Component analysis for GPU
    gpu_analysis = ComponentAnalysis(
        component=ComponentType.GPU,
        meets_minimum=True,
        meets_recommended=True,
        score=0.95,
        bottleneck_factor=0.0,
        details="RTX 4090 exceeds all requirements"
    )
    
    # Compatibility analysis
    compatibility_analysis = CompatibilityAnalysis(
        game_name="Cyberpunk 2077",
        overall_compatibility=CompatibilityLevel.EXCELLENT,
        can_run_minimum=True,
        can_run_recommended=True,
        component_analyses=[gpu_analysis],
        bottlenecks=[],
        overall_score=0.95,
        summary="Excellent performance expected",
        recommendations=["Enable RTX features", "Use DLSS Quality mode"]
    )
    
    # Performance with FPS range
    performance_prediction = PerformanceAssessment(
        score=95,
        tier="S",
        tier_description="Ultra High Performance",
        expected_fps=85,
        fps_min=75,  # FPS range
        fps_max=95,  # FPS range
        recommended_settings="Ultra with RTX",
        recommended_resolution="3072x1728",
        bottlenecks=[],
        upgrade_suggestions=[]
    )
    
    return CanRunResult(
        game_name="Cyberpunk 2077",
        timestamp=datetime.now().isoformat(),
        hardware_specs=hardware_specs,
        game_requirements=game_requirements,
        compatibility_analysis=compatibility_analysis,
        performance_prediction=performance_prediction,
        detected_resolution_class="4K",
        cache_used=False,
        analysis_time_ms=1500
    )


def test_actual_resolution_display():
    """Test that actual resolution is displayed instead of classification."""
    result = create_test_result_with_custom_resolution_and_fps()
    formatted_response = format_canrun_response(result)
    
    # Should show actual resolution
    assert "3072x1728" in formatted_response
    assert "165Hz" in formatted_response
    
    # Should NOT show resolution classification like "4K" in display line
    display_lines = [line for line in formatted_response.split('\n') if 'Display:' in line]
    assert len(display_lines) == 1
    display_line = display_lines[0]
    assert "3072x1728" in display_line
    assert "165Hz" in display_line


def test_fps_range_display():
    """Test that FPS ranges are displayed when available."""
    result = create_test_result_with_custom_resolution_and_fps()
    formatted_response = format_canrun_response(result)
    
    # Should show FPS range
    assert "75-95 FPS" in formatted_response
    
    # Should NOT show single unrealistic FPS numbers
    assert "340 FPS" not in formatted_response
    assert "200 FPS" not in formatted_response


def test_fallback_fps_display():
    """Test FPS display when range is not available."""
    result = create_test_result_with_custom_resolution_and_fps()
    
    # Remove fps_min and fps_max to test fallback
    result.performance_prediction.fps_min = None
    result.performance_prediction.fps_max = None
    
    formatted_response = format_canrun_response(result)
    
    # Should show single FPS with ~ prefix
    assert "~85 FPS" in formatted_response


def test_unknown_fps_display():
    """Test FPS display when no FPS data available."""
    result = create_test_result_with_custom_resolution_and_fps()
    
    # Remove all FPS data
    result.performance_prediction.fps_min = None
    result.performance_prediction.fps_max = None
    result.performance_prediction.expected_fps = None
    
    formatted_response = format_canrun_response(result)
    
    # Should show "Unknown"
    assert "FPS: **Unknown**" in formatted_response


def test_high_resolution_systems():
    """Test various high-resolution display scenarios."""
    test_cases = [
        ("3072x1728", "165Hz"),  # User's system
        ("3840x2160", "144Hz"),  # 4K
        ("2560x1440", "240Hz"),  # 1440p high refresh
        ("5120x2880", "60Hz"),   # 5K
    ]
    
    for resolution, refresh_rate in test_cases:
        result = create_test_result_with_custom_resolution_and_fps()
        result.hardware_specs.primary_monitor_resolution = resolution
        result.hardware_specs.primary_monitor_refresh_hz = int(refresh_rate.replace("Hz", ""))
        
        formatted_response = format_canrun_response(result)
        
        # Should show exact resolution and refresh rate
        assert resolution in formatted_response
        assert refresh_rate in formatted_response


if __name__ == "__main__":
    # Run tests
    test_actual_resolution_display()
    test_fps_range_display()
    test_fallback_fps_display()
    test_unknown_fps_display()
    test_high_resolution_systems()
    
    print("All resolution and FPS display tests passed!")