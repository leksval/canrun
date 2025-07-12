"""
Performance Prediction Module for CanRun
Heuristic-based performance predictions for RTX/GTX gaming systems with G-Assist integration.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from compatibility_analyzer import CompatibilityAnalysis, ComponentType, CompatibilityLevel


class PerformanceLevel(Enum):
    """Performance quality levels for gaming."""
    ULTRA = "Ultra"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    MINIMUM = "Minimum"


class Resolution(Enum):
    """Gaming resolutions supported by RTX/GTX GPUs."""
    HD_1080P = "1920x1080"
    QHD_1440P = "2560x1440"
    UHD_4K = "3840x2160"


@dataclass
class PerformanceSettings:
    """Gaming performance settings recommendation."""
    resolution: Resolution
    quality_preset: PerformanceLevel
    expected_fps: int
    dlss_enabled: bool
    rtx_enabled: bool
    settings_details: Dict[str, str]

    def __post_init__(self):
        """Validate performance settings after initialization."""
        assert self.expected_fps > 0, "Expected FPS must be greater than 0"
        assert self.settings_details, "Settings details cannot be empty"


@dataclass
class PerformancePrediction:
    """Complete gaming performance prediction result."""
    game_name: str
    predictions: List[PerformanceSettings]
    bottleneck_info: Dict[ComponentType, str]
    optimization_suggestions: List[str]
    dlss_support: bool
    rtx_support: bool
    overall_performance_score: float

    def __post_init__(self):
        """Validate performance prediction after initialization."""
        assert self.game_name.strip(), "Game name cannot be empty"
        assert 0.0 <= self.overall_performance_score <= 1.0, "Performance score must be between 0 and 1"
        assert isinstance(self.predictions, list), "Predictions must be a list"
        assert isinstance(self.optimization_suggestions, list), "Optimization suggestions must be a list"


class PerformancePredictor:
    """Heuristic-based performance predictor for RTX/GTX gaming systems."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # GPU performance tiers for FPS estimation (1080p Medium baseline)
        self.gpu_fps_baseline = {
            # RTX 40 Series
            'rtx 4090': 130, 'rtx 4080': 110, 'rtx 4070 ti': 95, 'rtx 4070': 80,
            'rtx 4060 ti': 70, 'rtx 4060': 60,
            # RTX 30 Series
            'rtx 3090': 105, 'rtx 3080': 90, 'rtx 3070': 75, 'rtx 3060 ti': 65,
            'rtx 3060': 55, 'rtx 3050': 45,
            # RTX 20 Series
            'rtx 2080 ti': 85, 'rtx 2080': 70, 'rtx 2070': 60, 'rtx 2060': 50,
            # GTX 16 Series
            'gtx 1660 ti': 50, 'gtx 1660': 45, 'gtx 1650': 35,
            # GTX 10 Series
            'gtx 1080 ti': 70, 'gtx 1080': 55, 'gtx 1070': 45, 'gtx 1060': 35,
            'gtx 1050': 25
        }
        
        # Resolution performance scaling factors
        self.resolution_scaling = {
            Resolution.HD_1080P: 1.0,
            Resolution.QHD_1440P: 0.60,
            Resolution.UHD_4K: 0.30
        }
        
        # Quality preset performance scaling factors
        self.quality_scaling = {
            PerformanceLevel.ULTRA: 0.65,
            PerformanceLevel.HIGH: 0.80,
            PerformanceLevel.MEDIUM: 1.0,
            PerformanceLevel.LOW: 1.30,
            PerformanceLevel.MINIMUM: 1.60
        }
        
        # DLSS performance boost factors
        self.dlss_boost = {
            'Quality': 1.35,
            'Balanced': 1.55,
            'Performance': 1.85,
            'Ultra Performance': 2.25
        }
        
        # RTX-capable GPUs for ray tracing
        self.rtx_capable_gpus = {
            'rtx 4090', 'rtx 4080', 'rtx 4070 ti', 'rtx 4070', 'rtx 4060 ti', 'rtx 4060',
            'rtx 3090', 'rtx 3080', 'rtx 3070', 'rtx 3060 ti', 'rtx 3060', 'rtx 3050',
            'rtx 2080 ti', 'rtx 2080', 'rtx 2070', 'rtx 2060'
        }
        
        # DLSS 3.0 Frame Generation capable GPUs
        self.dlss3_capable_gpus = {
            'rtx 4090', 'rtx 4080', 'rtx 4070 ti', 'rtx 4070', 'rtx 4060 ti', 'rtx 4060'
        }
        
        self.logger.info("Performance predictor initialized for RTX/GTX gaming systems")
    
    def predict_performance(self, compatibility_analysis: CompatibilityAnalysis, 
                          gpu_name: str, vram_gb: int, supports_rtx: bool, 
                          supports_dlss: bool) -> PerformancePrediction:
        """Generate comprehensive gaming performance prediction."""
        # Validate inputs
        assert compatibility_analysis.game_name.strip(), "Game name cannot be empty"
        assert gpu_name.strip(), "GPU name cannot be empty"
        assert vram_gb > 0, "VRAM must be greater than 0"
        assert self._is_supported_gpu(gpu_name), "RTX/GTX GPU required for performance prediction"
        
        gpu_lower = gpu_name.lower()
        
        # Generate performance settings for different scenarios
        predictions = self._generate_performance_settings(
            gpu_lower, vram_gb, compatibility_analysis, supports_rtx, supports_dlss
        )
        
        # Analyze bottlenecks and their performance impact
        bottleneck_info = self._analyze_bottleneck_impact(compatibility_analysis)
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(
            gpu_lower, compatibility_analysis, bottleneck_info, supports_rtx, supports_dlss
        )
        
        # Calculate overall performance score
        performance_score = self._calculate_performance_score(compatibility_analysis, predictions)
        
        return PerformancePrediction(
            game_name=compatibility_analysis.game_name,
            predictions=predictions,
            bottleneck_info=bottleneck_info,
            optimization_suggestions=optimization_suggestions,
            dlss_support=supports_dlss,
            rtx_support=supports_rtx,
            overall_performance_score=performance_score
        )
    
    def _generate_performance_settings(self, gpu_name: str, vram_gb: int,
                                     compatibility_analysis: CompatibilityAnalysis,
                                     supports_rtx: bool, supports_dlss: bool) -> List[PerformanceSettings]:
        """Generate gaming performance settings for different scenarios."""
        settings = []
        base_fps = self._get_gpu_baseline_fps(gpu_name)
        
        # Apply compatibility modifier
        compatibility_modifier = self._get_compatibility_modifier(compatibility_analysis)
        base_fps *= compatibility_modifier
        
        # Generate settings for each resolution and quality combination
        for resolution in Resolution:
            resolution_fps = base_fps * self.resolution_scaling[resolution]
            
            # Check VRAM requirements for higher resolutions
            if not self._check_vram_requirements(resolution, vram_gb):
                continue
            
            for quality in PerformanceLevel:
                quality_fps = resolution_fps * self.quality_scaling[quality]
                
                # Determine DLSS usage
                dlss_enabled = self._should_enable_dlss(
                    gpu_name, quality_fps, quality, resolution, supports_dlss
                )
                if dlss_enabled:
                    quality_fps *= self.dlss_boost['Balanced']
                
                # Determine RTX usage
                rtx_enabled = self._should_enable_rtx(
                    gpu_name, quality_fps, quality, supports_rtx
                )
                if rtx_enabled:
                    quality_fps *= 0.65  # RTX performance cost
                
                # Only include viable settings (30+ FPS or minimum preset)
                if quality_fps >= 30 or quality == PerformanceLevel.MINIMUM:
                    settings_details = self._generate_settings_details(
                        quality, dlss_enabled, rtx_enabled
                    )
                    
                    settings.append(PerformanceSettings(
                        resolution=resolution,
                        quality_preset=quality,
                        expected_fps=int(quality_fps),
                        dlss_enabled=dlss_enabled,
                        rtx_enabled=rtx_enabled,
                        settings_details=settings_details
                    ))
        
        # Sort by expected FPS (descending) and return top 8 settings
        settings.sort(key=lambda x: x.expected_fps, reverse=True)
        return settings[:8]
    
    def _get_gpu_baseline_fps(self, gpu_name: str) -> float:
        """Get GPU baseline FPS at 1080p Medium."""
        assert gpu_name.strip(), "GPU name cannot be empty"
        
        # Match against known GPU performance tiers
        for gpu_key, fps in self.gpu_fps_baseline.items():
            if gpu_key in gpu_name:
                return float(fps)
        
        # Fallback estimation based on GPU generation
        if 'rtx 40' in gpu_name:
            return 75.0  # Average RTX 40 series
        elif 'rtx 30' in gpu_name:
            return 65.0  # Average RTX 30 series
        elif 'rtx 20' in gpu_name:
            return 55.0  # Average RTX 20 series
        elif 'gtx 16' in gpu_name:
            return 45.0  # Average GTX 16 series
        elif 'gtx 10' in gpu_name:
            return 40.0  # Average GTX 10 series
        else:
            return 35.0  # Conservative estimate for older GPUs
    
    def _get_compatibility_modifier(self, compatibility_analysis: CompatibilityAnalysis) -> float:
        """Get performance modifier based on compatibility analysis."""
        if compatibility_analysis.overall_compatibility == CompatibilityLevel.EXCELLENT:
            return 1.25
        elif compatibility_analysis.overall_compatibility == CompatibilityLevel.GOOD:
            return 1.0
        elif compatibility_analysis.overall_compatibility == CompatibilityLevel.ADEQUATE:
            return 0.80
        elif compatibility_analysis.overall_compatibility == CompatibilityLevel.POOR:
            return 0.65
        else:
            return 0.45
    
    def _check_vram_requirements(self, resolution: Resolution, vram_gb: int) -> bool:
        """Check if VRAM is sufficient for resolution."""
        vram_requirements = {
            Resolution.HD_1080P: 4,
            Resolution.QHD_1440P: 8,
            Resolution.UHD_4K: 12
        }
        
        return vram_gb >= vram_requirements[resolution]
    
    def _should_enable_dlss(self, gpu_name: str, current_fps: float, 
                           quality: PerformanceLevel, resolution: Resolution,
                           supports_dlss: bool) -> bool:
        """Determine if DLSS should be enabled for better performance."""
        if not supports_dlss:
            return False
        
        # Enable DLSS if FPS is below 60 and quality is High or Ultra
        if current_fps < 60 and quality in [PerformanceLevel.HIGH, PerformanceLevel.ULTRA]:
            return True
        
        # Always enable DLSS for 4K gaming
        if resolution == Resolution.UHD_4K:
            return True
        
        # Enable DLSS for 1440p at Ultra settings
        if resolution == Resolution.QHD_1440P and quality == PerformanceLevel.ULTRA:
            return True
        
        return False
    
    def _should_enable_rtx(self, gpu_name: str, current_fps: float, 
                          quality: PerformanceLevel, supports_rtx: bool) -> bool:
        """Determine if RTX should be enabled."""
        if not supports_rtx:
            return False
        
        # Only enable RTX if we have good performance headroom (>70 FPS)
        if current_fps > 70 and quality in [PerformanceLevel.HIGH, PerformanceLevel.ULTRA]:
            # High-end RTX cards can handle RTX better
            high_end_rtx = ['rtx 4090', 'rtx 4080', 'rtx 3090', 'rtx 3080']
            if any(gpu in gpu_name for gpu in high_end_rtx):
                return True
        
        return False
    
    def _generate_settings_details(self, quality: PerformanceLevel, 
                                 dlss_enabled: bool, rtx_enabled: bool) -> Dict[str, str]:
        """Generate detailed settings recommendations."""
        settings = {
            'Texture Quality': quality.value,
            'Shadow Quality': quality.value,
            'Effects Quality': quality.value,
            'Post-Processing': quality.value,
            'Anti-Aliasing': 'TAA' if quality != PerformanceLevel.MINIMUM else 'FXAA'
        }
        
        if dlss_enabled:
            settings['DLSS'] = 'Balanced'
        
        if rtx_enabled:
            settings['Ray Tracing'] = 'On'
            settings['RTX Global Illumination'] = 'On'
            settings['RTX Reflections'] = 'On'
        
        return settings
    
    def _analyze_bottleneck_impact(self, compatibility_analysis: CompatibilityAnalysis) -> Dict[ComponentType, str]:
        """Analyze bottleneck impact on performance."""
        bottleneck_info = {}
        
        for component in compatibility_analysis.bottlenecks:
            if component == ComponentType.GPU:
                bottleneck_info[component] = "GPU is limiting factor - consider lower settings or DLSS"
            elif component == ComponentType.CPU:
                bottleneck_info[component] = "CPU may cause frame drops in CPU-intensive scenes"
            elif component == ComponentType.RAM:
                bottleneck_info[component] = "Low RAM may cause stuttering and texture streaming issues"
            elif component == ComponentType.STORAGE:
                bottleneck_info[component] = "Slow storage may cause longer loading times"
            elif component == ComponentType.OS:
                bottleneck_info[component] = "OS compatibility may affect performance"
            elif component == ComponentType.DIRECTX:
                bottleneck_info[component] = "DirectX version may limit graphics features"
        
        return bottleneck_info
    
    def _generate_optimization_suggestions(self, gpu_name: str, 
                                         compatibility_analysis: CompatibilityAnalysis,
                                         bottleneck_info: Dict[ComponentType, str],
                                         supports_rtx: bool, supports_dlss: bool) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []
        
        # DLSS suggestions
        if supports_dlss:
            suggestions.append("Enable DLSS for significant performance boost without quality loss")
            
            if gpu_name in self.dlss3_capable_gpus:
                suggestions.append("DLSS 3.0 Frame Generation available for even higher FPS")
        
        # RTX suggestions
        if supports_rtx:
            suggestions.append("RTX ray tracing available - enable if performance allows")
        
        # Bottleneck-specific suggestions
        if ComponentType.GPU in bottleneck_info:
            suggestions.append("Lower texture quality and effects for better performance")
            if supports_dlss:
                suggestions.append("Use DLSS Performance mode for maximum FPS gain")
        
        if ComponentType.CPU in bottleneck_info:
            suggestions.append("Reduce CPU-intensive settings like draw distance and NPC count")
        
        if ComponentType.RAM in bottleneck_info:
            suggestions.append("Close background applications to free up RAM")
            suggestions.append("Consider upgrading to 16GB+ RAM for modern games")
        
        # Resolution scaling suggestions
        if compatibility_analysis.overall_compatibility == CompatibilityLevel.POOR:
            suggestions.append("Consider lowering resolution to 1080p for better performance")
        
        # G-Assist specific suggestions
        suggestions.append("Use G-Assist for real-time game-specific optimization tips")
        
        return suggestions
    
    def _calculate_performance_score(self, compatibility_analysis: CompatibilityAnalysis,
                                   predictions: List[PerformanceSettings]) -> float:
        """Calculate overall performance score."""
        if not predictions:
            return 0.0
        
        # Base score from compatibility analysis
        base_score = compatibility_analysis.overall_score
        
        # Bonus for high FPS predictions
        max_fps = max(pred.expected_fps for pred in predictions)
        fps_bonus = min(max_fps / 60.0, 2.0)  # Cap at 2x bonus
        
        # Bonus for advanced features
        feature_bonus = 0.0
        if any(pred.dlss_enabled for pred in predictions):
            feature_bonus += 0.1
        if any(pred.rtx_enabled for pred in predictions):
            feature_bonus += 0.1
        
        return min(base_score * fps_bonus + feature_bonus, 1.0)
    
    def _is_supported_gpu(self, gpu_name: str) -> bool:
        """Check if GPU is supported for performance prediction."""
        gpu_lower = gpu_name.lower()
        return any(keyword in gpu_lower for keyword in ['rtx', 'gtx', 'geforce'])
    
    def get_performance_summary(self, prediction: PerformancePrediction) -> str:
        """Generate human-readable performance summary."""
        assert prediction.game_name.strip(), "Game name cannot be empty"
        
        if not prediction.predictions:
            return f"Performance analysis unavailable for {prediction.game_name}"
        
        best_setting = prediction.predictions[0]
        
        summary = f"{prediction.game_name} Performance Summary:\n"
        summary += f"Best Performance: {best_setting.expected_fps} FPS at {best_setting.resolution.value} {best_setting.quality_preset.value}\n"
        
        if best_setting.dlss_enabled:
            summary += "DLSS: Enabled for enhanced performance\n"
        
        if best_setting.rtx_enabled:
            summary += "RTX: Enabled for enhanced visuals\n"
        
        if prediction.optimization_suggestions:
            summary += f"Top Suggestion: {prediction.optimization_suggestions[0]}\n"
        
        return summary