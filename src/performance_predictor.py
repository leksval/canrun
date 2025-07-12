"""
Performance Prediction Module for CanRun
Provides heuristic-based performance predictions for games on NVIDIA RTX/GTX systems.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .compatibility_analyzer import CompatibilityAnalysis, ComponentType, CompatibilityLevel


class PerformanceLevel(Enum):
    """Performance quality levels."""
    ULTRA = "Ultra"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    MINIMUM = "Minimum"


class Resolution(Enum):
    """Supported resolutions."""
    HD_1080P = "1920x1080"
    QHD_1440P = "2560x1440"
    UHD_4K = "3840x2160"


@dataclass
class PerformanceSettings:
    """Recommended performance settings."""
    resolution: Resolution
    quality_preset: PerformanceLevel
    expected_fps: int
    dlss_enabled: bool
    rtx_enabled: bool
    settings_details: Dict[str, str]


@dataclass
class PerformancePrediction:
    """Complete performance prediction for a game."""
    game_name: str
    predictions: List[PerformanceSettings]
    bottleneck_info: Dict[ComponentType, str]
    optimization_suggestions: List[str]
    dlss_support: bool
    rtx_support: bool
    overall_performance_score: float


class PerformancePredictor:
    """Heuristic-based performance predictor for NVIDIA RTX/GTX systems."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # GPU performance tiers for FPS estimation
        self.gpu_fps_base = {
            # RTX 40 series
            'rtx 4090': 120,
            'rtx 4080': 100,
            'rtx 4070 ti': 85,
            'rtx 4070': 75,
            'rtx 4060': 60,
            
            # RTX 30 series
            'rtx 3090': 95,
            'rtx 3080': 85,
            'rtx 3070': 70,
            'rtx 3060': 55,
            
            # RTX 20 series
            'rtx 2080': 65,
            'rtx 2070': 60,
            'rtx 2060': 50,
            
            # GTX 16 series
            'gtx 1660': 45,
            'gtx 1650': 40,
            
            # GTX 10 series
            'gtx 1080': 50,
            'gtx 1070': 45,
            'gtx 1060': 35,
            
            # Older GTX
            'gtx 970': 30,
            'gtx 960': 25,
        }
        
        # Resolution scaling factors
        self.resolution_scaling = {
            Resolution.HD_1080P: 1.0,
            Resolution.QHD_1440P: 0.65,
            Resolution.UHD_4K: 0.35
        }
        
        # Quality preset scaling factors
        self.quality_scaling = {
            PerformanceLevel.ULTRA: 0.7,
            PerformanceLevel.HIGH: 0.85,
            PerformanceLevel.MEDIUM: 1.0,
            PerformanceLevel.LOW: 1.25,
            PerformanceLevel.MINIMUM: 1.5
        }
        
        # DLSS performance boost factors
        self.dlss_boost = {
            'Quality': 1.3,
            'Balanced': 1.5,
            'Performance': 1.8,
            'Ultra Performance': 2.2
        }
        
        # RTX-capable GPUs
        self.rtx_capable = {
            'rtx 4090', 'rtx 4080', 'rtx 4070 ti', 'rtx 4070', 'rtx 4060',
            'rtx 3090', 'rtx 3080', 'rtx 3070', 'rtx 3060',
            'rtx 2080', 'rtx 2070', 'rtx 2060'
        }
        
        # DLSS 3.0 capable GPUs (Frame Generation)
        self.dlss3_capable = {
            'rtx 4090', 'rtx 4080', 'rtx 4070 ti', 'rtx 4070', 'rtx 4060'
        }
    
    def predict_performance(self, compatibility_analysis: CompatibilityAnalysis, 
                          gpu_name: str) -> PerformancePrediction:
        """Generate comprehensive performance prediction."""
        try:
            self.logger.info(f"Predicting performance for {compatibility_analysis.game_name}")
            
            # Get GPU performance characteristics
            gpu_lower = gpu_name.lower()
            is_nvidia = self._is_nvidia_gpu(gpu_lower)
            
            if not is_nvidia:
                return self._create_non_nvidia_prediction(compatibility_analysis.game_name)
            
            # Generate performance settings for different scenarios
            predictions = self._generate_performance_settings(gpu_lower, compatibility_analysis)
            
            # Analyze bottlenecks and their impact
            bottleneck_info = self._analyze_bottleneck_impact(compatibility_analysis)
            
            # Generate optimization suggestions
            optimization_suggestions = self._generate_optimization_suggestions(
                gpu_lower, compatibility_analysis, bottleneck_info
            )
            
            # Determine RTX and DLSS support
            dlss_support = self._supports_dlss(gpu_lower)
            rtx_support = self._supports_rtx(gpu_lower)
            
            # Calculate overall performance score
            performance_score = self._calculate_performance_score(
                compatibility_analysis, predictions
            )
            
            return PerformancePrediction(
                game_name=compatibility_analysis.game_name,
                predictions=predictions,
                bottleneck_info=bottleneck_info,
                optimization_suggestions=optimization_suggestions,
                dlss_support=dlss_support,
                rtx_support=rtx_support,
                overall_performance_score=performance_score
            )
            
        except Exception as e:
            self.logger.error(f"Performance prediction failed: {e}")
            return self._create_fallback_prediction(compatibility_analysis.game_name)
    
    def _generate_performance_settings(self, gpu_name: str, 
                                     compatibility_analysis: CompatibilityAnalysis) -> List[PerformanceSettings]:
        """Generate performance settings for different quality levels."""
        settings = []
        base_fps = self._get_base_fps(gpu_name)
        
        # Apply compatibility modifier
        compatibility_modifier = self._get_compatibility_modifier(compatibility_analysis)
        base_fps *= compatibility_modifier
        
        # Generate settings for each resolution
        for resolution in Resolution:
            resolution_fps = base_fps * self.resolution_scaling[resolution]
            
            # Generate quality presets that achieve good performance
            for quality in PerformanceLevel:
                quality_fps = resolution_fps * self.quality_scaling[quality]
                
                # Check if DLSS should be enabled
                dlss_enabled = self._should_enable_dlss(gpu_name, quality_fps, quality)
                if dlss_enabled:
                    quality_fps *= self.dlss_boost['Balanced']  # Use balanced DLSS
                
                # Check if RTX should be enabled
                rtx_enabled = self._should_enable_rtx(gpu_name, quality_fps, quality)
                if rtx_enabled:
                    quality_fps *= 0.7  # RTX performance cost
                
                # Only include settings that achieve reasonable performance
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
        
        # Sort by expected FPS (descending)
        settings.sort(key=lambda x: x.expected_fps, reverse=True)
        
        # Return top 6 most relevant settings
        return settings[:6]
    
    def _get_base_fps(self, gpu_name: str) -> float:
        """Get base FPS for GPU at 1080p Medium settings."""
        for gpu_key, fps in self.gpu_fps_base.items():
            if gpu_key in gpu_name:
                return fps
        
        # Fallback for unknown NVIDIA GPUs
        return 40.0
    
    def _get_compatibility_modifier(self, compatibility_analysis: CompatibilityAnalysis) -> float:
        """Get performance modifier based on compatibility analysis."""
        if compatibility_analysis.overall_compatibility == CompatibilityLevel.EXCELLENT:
            return 1.2
        elif compatibility_analysis.overall_compatibility == CompatibilityLevel.GOOD:
            return 1.0
        elif compatibility_analysis.overall_compatibility == CompatibilityLevel.ADEQUATE:
            return 0.8
        elif compatibility_analysis.overall_compatibility == CompatibilityLevel.POOR:
            return 0.6
        else:
            return 0.4
    
    def _should_enable_dlss(self, gpu_name: str, current_fps: float, 
                           quality: PerformanceLevel) -> bool:
        """Determine if DLSS should be enabled for better performance."""
        if not self._supports_dlss(gpu_name):
            return False
        
        # Enable DLSS if FPS is below 60 and quality is High or Ultra
        if current_fps < 60 and quality in [PerformanceLevel.HIGH, PerformanceLevel.ULTRA]:
            return True
        
        # Enable DLSS for 4K at any quality level
        return False  # Resolution context needed for better decision
    
    def _should_enable_rtx(self, gpu_name: str, current_fps: float, 
                          quality: PerformanceLevel) -> bool:
        """Determine if RTX should be enabled."""
        if not self._supports_rtx(gpu_name):
            return False
        
        # Only enable RTX if we have headroom (>80 FPS) and quality is High or Ultra
        if current_fps > 80 and quality in [PerformanceLevel.HIGH, PerformanceLevel.ULTRA]:
            # High-end RTX cards can handle RTX better
            if any(gpu in gpu_name for gpu in ['rtx 4090', 'rtx 4080', 'rtx 3090', 'rtx 3080']):
                return True
        
        return False
    
    def _generate_settings_details(self, quality: PerformanceLevel, 
                                 dlss_enabled: bool, rtx_enabled: bool) -> Dict[str, str]:
        """Generate detailed settings recommendations."""
        settings = {
            'Texture Quality': quality.value,
            'Shadow Quality': quality.value,
            'Effects Quality': quality.value,
            'Post-Processing': quality.value
        }
        
        if dlss_enabled:
            settings['DLSS'] = 'Balanced'
        
        if rtx_enabled:
            settings['Ray Tracing'] = 'On'
            settings['RTX Global Illumination'] = 'On'
        
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
        
        return bottleneck_info
    
    def _generate_optimization_suggestions(self, gpu_name: str, 
                                         compatibility_analysis: CompatibilityAnalysis,
                                         bottleneck_info: Dict[ComponentType, str]) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []
        
        # DLSS suggestions
        if self._supports_dlss(gpu_name):
            suggestions.append("Enable DLSS for significant performance boost without quality loss")
            
            if gpu_name in self.dlss3_capable:
                suggestions.append("DLSS 3.0 Frame Generation available for even higher FPS")
        
        # RTX suggestions
        if self._supports_rtx(gpu_name):
            suggestions.append("RTX features available - enable if performance allows")
        
        # Bottleneck-specific suggestions
        if ComponentType.GPU in bottleneck_info:
            suggestions.append("Lower texture quality and effects for better performance")
        
        if ComponentType.CPU in bottleneck_info:
            suggestions.append("Reduce CPU-intensive settings like draw distance and NPC density")
        
        if ComponentType.RAM in bottleneck_info:
            suggestions.append("Close background applications to free up RAM")
        
        # Resolution scaling suggestions
        if compatibility_analysis.overall_compatibility == CompatibilityLevel.POOR:
            suggestions.append("Consider lowering resolution to 1080p for better performance")
        
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
        
        # Bonus for DLSS/RTX support
        feature_bonus = 0.0
        if any(pred.dlss_enabled for pred in predictions):
            feature_bonus += 0.1
        if any(pred.rtx_enabled for pred in predictions):
            feature_bonus += 0.1
        
        return min(base_score * fps_bonus + feature_bonus, 1.0)
    
    def _is_nvidia_gpu(self, gpu_name: str) -> bool:
        """Check if GPU is NVIDIA."""
        return 'nvidia' in gpu_name or 'geforce' in gpu_name or 'rtx' in gpu_name or 'gtx' in gpu_name
    
    def _supports_dlss(self, gpu_name: str) -> bool:
        """Check if GPU supports DLSS."""
        return any(gpu in gpu_name for gpu in self.rtx_capable)
    
    def _supports_rtx(self, gpu_name: str) -> bool:
        """Check if GPU supports RTX features."""
        return any(gpu in gpu_name for gpu in self.rtx_capable)
    
    def _create_non_nvidia_prediction(self, game_name: str) -> PerformancePrediction:
        """Create prediction for non-NVIDIA GPUs."""
        return PerformancePrediction(
            game_name=game_name,
            predictions=[],
            bottleneck_info={},
            optimization_suggestions=[
                "G-Assist requires an NVIDIA RTX or GTX GPU",
                "Consider upgrading to an NVIDIA GPU for full CanRun compatibility"
            ],
            dlss_support=False,
            rtx_support=False,
            overall_performance_score=0.0
        )
    
    def _create_fallback_prediction(self, game_name: str) -> PerformancePrediction:
        """Create fallback prediction when analysis fails."""
        return PerformancePrediction(
            game_name=game_name,
            predictions=[],
            bottleneck_info={},
            optimization_suggestions=["Performance analysis unavailable"],
            dlss_support=False,
            rtx_support=False,
            overall_performance_score=0.5
        )


def main():
    """Test the performance predictor."""
    print("NVIDIA RTX/GTX performance predictor module loaded successfully")


if __name__ == "__main__":
    main()