"""
Test that the plugin correctly displays actual resolution and FPS ranges.
"""

import pytest
import sys
import os

# Add canrun to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plugin import _fps_range_text, format_canrun_response
from canrun_steam_fps import graphics_feature_lines
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
    assert formatted_response.startswith("🎮 **Game:**")
    assert formatted_response.index("🎯 **VERDICT:**") < formatted_response.index("Estimate only.")


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


def test_steam_player_source_label():
    """Steam measurements name the player sample and skip the DLSS side note."""
    result = create_test_result_with_custom_resolution_and_fps()
    result.performance_prediction.fps_source = "steam"
    formatted_response = format_canrun_response(result)
    assert "75-95 FPS (Steam players)" in formatted_response
    assert "FPS more" not in formatted_response
    assert "Ray tracing:" not in formatted_response
    assert "Path tracing:" not in formatted_response


def test_similar_pc_source_label():
    """Scaled estimates say they come from a similar PC and add a DLSS side note."""
    result = create_test_result_with_custom_resolution_and_fps()
    result.performance_prediction.fps_source = "steam_similar"
    formatted_response = format_canrun_response(result)
    assert "75-95 FPS (estimated from a similar PC, native, ray tracing off)" in formatted_response
    assert "DLSS 4.5: Quality, Balanced, Performance, Ultra Performance." in formatted_response
    assert "DLSS 4.5 frame gen: around 85 FPS more (2×)." in formatted_response
    assert "Path tracing: around 45 to 55 FPS less (typically about 2× to 3× heavier)." in formatted_response
    assert "Ray tracing:" not in formatted_response


def test_scaled_steam_estimate_is_not_called_native():
    """Other players' mixed settings are not a native figure, so no DLSS note is added."""
    result = create_test_result_with_custom_resolution_and_fps()
    result.performance_prediction.fps_source = "steam_scaled"
    formatted_response = format_canrun_response(result)
    assert "75-95 FPS (scaled from Steam players on a similar GPU, mixed settings)" in formatted_response
    assert "native" not in formatted_response
    assert "FPS more" not in formatted_response


def test_dlss_note_follows_nvidia_multipliers():
    """Side notes use NVIDIA's multipliers and the GPU's actual features."""
    quality_1080 = graphics_feature_lines(
        "requirements", 60, 48, 72, "1080p", "RTX 2060", 6, True, True
    )
    quality_text = "\n".join(quality_1080)
    assert quality_1080[0] == "DLSS 4: Quality, Balanced, Performance."
    assert "Path tracing: around 30 to 40 FPS less (typically about 2× to 3× heavier)." in quality_text
    assert "frame gen" not in quality_text
    assert "Ray tracing:" not in quality_text
    assert all("\n" not in line for line in quality_1080)

    high_rate = graphics_feature_lines(
        "steam_similar", 160, 140, 180, "1080p", "RTX 4070", 12, True, True
    )
    high_text = "\n".join(high_rate)
    assert "DLSS 4.5: Quality, Balanced, Performance." in high_text
    assert "DLSS 4.5 frame gen: around 160 FPS more (2×)." in high_text
    assert "Path tracing: around 80 to 105 FPS less (typically about 2× to 3× heavier)." in high_text

    blackwell = graphics_feature_lines(
        "steam_similar", 80, 64, 96, "4K", "RTX 5070", 12, True, True, None, 144
    )
    blackwell_text = "\n".join(blackwell)
    assert "DLSS 4.5: Quality, Balanced, Performance, Ultra Performance." in blackwell_text
    assert "DLSS 4.5 multi frame gen: around 240 to 400 FPS more (4× to 6×)." in blackwell_text
    assert "DLSS 4.5 Dynamic mode aims at this 144 Hz screen." in blackwell_text
    assert "Path tracing: around 40 to 55 FPS less (typically about 2× to 3× heavier)." in blackwell_text

    assert graphics_feature_lines("steam", 80, 70, 90, "4K", "RTX 4090", 24, True, True) == []
    assert graphics_feature_lines("steam_scaled", 80, 64, 96, "4K", "RTX 4090", 24, True, True) == []
    assert graphics_feature_lines("steam_similar", 80, 70, 90, "1080p", "GTX 1660", 6, False, False) == []

    capped = graphics_feature_lines("requirements", 240, 192, 288, "1080p", "RTX 4090", 24, True, True)
    assert capped == []

    older = graphics_feature_lines("requirements", 90, 72, 108, "720p", "RTX 3060", 12, True, True)
    assert older == [
        "DLSS 4: Quality, Balanced, Performance.",
        "Path tracing: around 45 to 60 FPS less (typically about 2× to 3× heavier).",
    ]
    no_rtx = graphics_feature_lines("requirements", 90, 72, 108, "1080p", "RTX 3060", 12, True, False)
    assert no_rtx == ["DLSS 4: Quality, Balanced, Performance."]


def test_answer_starts_with_the_formatted_card():
    """The plain score header is not repeated ahead of the formatted card."""
    result = create_test_result_with_custom_resolution_and_fps()
    formatted = format_canrun_response(result)
    assert formatted.startswith("🎮 **Game:**")
    assert "can run on this PC" not in formatted.split("🎮 **Game:**", 1)[0]
    assert formatted.count("Estimate only.") == 1
    assert formatted.rstrip().endswith("or that you will see this frame rate.")


def test_fps_max_stops_at_this_screen_refresh_rate():
    """This monitor's refresh rate is the ceiling. A faster screen can show the rest."""
    assert _fps_range_text(120, 200, 144, " (Steam players)") == (
        "120-144 FPS on this 144 Hz screen (Steam players). On a faster monitor, about 120-200."
    )
    assert _fps_range_text(160, 180, 144, "") == (
        "144 FPS on this 144 Hz screen. On a faster monitor, about 160-180."
    )
    assert _fps_range_text(75, 95, 165, " (Steam players)") == "75-95 FPS (Steam players)"

    result = create_test_result_with_custom_resolution_and_fps()
    result.hardware_specs.primary_monitor_refresh_hz = 144
    result.performance_prediction.fps_min = 120
    result.performance_prediction.fps_max = 200
    result.performance_prediction.fps_source = "steam"
    formatted = format_canrun_response(result)
    assert "120-144 FPS on this 144 Hz screen (Steam players)." in formatted
    assert "On a faster monitor, about 120-200." in formatted


def test_heavier_display_notes_the_fps_class():
    """A monitor above the FPS class is labeled so the range is not read as exact."""
    result = create_test_result_with_custom_resolution_and_fps()
    result.detected_resolution_class = "1440p"
    result.hardware_specs.primary_monitor_resolution = "3072x1728"
    formatted_response = format_canrun_response(result)
    assert "FPS estimate uses 1440p" in formatted_response


if __name__ == "__main__":
    # Run tests
    test_actual_resolution_display()
    test_fps_range_display()
    test_fallback_fps_display()
    test_unknown_fps_display()
    test_high_resolution_systems()
    test_steam_player_source_label()
    test_similar_pc_source_label()
    
    print("All resolution and FPS display tests passed!")