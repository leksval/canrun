"""
Performance Prediction Module for CanRun
Advanced tiered performance predictions (S-A-B-C-D-F) for RTX/GTX gaming systems with G-Assist integration.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from compatibility_analyzer import CompatibilityAnalysis, ComponentType, CompatibilityLevel


class PerformanceTier(Enum):
    """Performance tier classifications"""
    S = (90, 100, "Exceptional - Ultra settings, 4K@60fps+")
    A = (80, 89, "Excellent - High settings, 1440p@60fps")
    B = (70, 79, "Good - High settings, 1080p@60fps")
    C = (60, 69, "Adequate - Medium settings, 1080p@30fps")
    D = (50, 59, "Minimum - Low settings, 720p@30fps")
    F = (0, 49, "Below Minimum - Unable to run acceptably")


@dataclass
class PerformanceAssessment:
    """Complete performance assessment result with S-A-B-C-D-F tier system"""
    score: int
    tier: PerformanceTier
    tier_description: str
    expected_fps: int
    recommended_settings: str
    recommended_resolution: str
    bottlenecks: list
    upgrade_suggestions: list


@dataclass
class PerformancePrediction:
    """Legacy performance prediction for backward compatibility"""
    game_name: str
    predictions: List[Dict]
    bottleneck_info: Dict[ComponentType, str]
    optimization_suggestions: List[str]
    dlss_support: bool
    rtx_support: bool
    overall_performance_score: float

    def __post_init__(self):
        """Validate performance prediction after initialization."""
        assert self.game_name.strip(), "Game name cannot be empty"
        assert 0.0 <= self.overall_performance_score <= 100.0, "Performance score must be between 0 and 100"


class PerformancePredictor:
    """Advanced tiered performance predictor for RTX/GTX gaming systems."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Comprehensive GPU hierarchy (higher score = better performance)
        self.gpu_hierarchy = {
            # NVIDIA RTX 40 Series
            "RTX 4090": 100, "RTX 4080": 95, "RTX 4070 Ti": 90,
            "RTX 4070": 85, "RTX 4060 Ti": 80, "RTX 4060": 75,
            
            # NVIDIA RTX 30 Series
            "RTX 3090 Ti": 88, "RTX 3090": 86, "RTX 3080 Ti": 84,
            "RTX 3080": 82, "RTX 3070 Ti": 78, "RTX 3070": 76,
            "RTX 3060 Ti": 72, "RTX 3060": 68, "RTX 3050": 60,
            
            # NVIDIA GTX Series
            "GTX 1660 Ti": 55, "GTX 1660": 52, "GTX 1650": 45,
            "GTX 1060": 48, "GTX 1050 Ti": 40, "GTX 1050": 35,
            
            # AMD RX Series
            "RX 7900 XTX": 93, "RX 7900 XT": 88, "RX 7800 XT": 83,
            "RX 7700 XT": 78, "RX 7600": 70, "RX 6900 XT": 81,
            "RX 6800 XT": 79, "RX 6800": 77, "RX 6700 XT": 73,
            "RX 6600 XT": 65, "RX 6600": 62, "RX 6500 XT": 50,
        }
        
        # Comprehensive CPU hierarchy
        self.cpu_hierarchy = {
            # Intel 13th Gen
            "i9-13900K": 100, "i7-13700K": 92, "i5-13600K": 85,
            "i5-13400": 78, "i3-13100": 65,
            
            # AMD Ryzen 7000
            "Ryzen 9 7950X": 98, "Ryzen 9 7900X": 94, "Ryzen 7 7700X": 90,
            "Ryzen 5 7600X": 84, "Ryzen 5 7600": 80,
            
            # AMD Ryzen 5000
            "Ryzen 9 5950X": 93, "Ryzen 9 5900X": 90, "Ryzen 7 5800X3D": 95,
            "Ryzen 7 5800X": 86, "Ryzen 5 5600X": 80, "Ryzen 5 5600": 76,
        }
        
        # Performance tier descriptions
        self.tier_descriptions = {
            PerformanceTier.S: "Exceptional - Ultra settings, 4K@60fps+",
            PerformanceTier.A: "Excellent - High settings, 1440p@60fps",
            PerformanceTier.B: "Good - High settings, 1080p@60fps",
            PerformanceTier.C: "Adequate - Medium settings, 1080p@30fps",
            PerformanceTier.D: "Minimum - Low settings, 720p@30fps",
            PerformanceTier.F: "Below Minimum - Unable to run acceptably"
        }
        
        self.logger.info("Performance predictor initialized for RTX/GTX gaming systems")
    
    def assess_performance(self, hardware_specs: Dict, game_requirements: Dict = None) -> PerformanceAssessment:
        """
        Generate advanced tiered performance assessment using S-A-B-C-D-F tier system.
        
        Args:
            hardware_specs: Dictionary containing hardware specifications
            game_requirements: Optional game requirements
            
        Returns:
            PerformanceAssessment with tier, score, FPS, and recommendations
        """
        self.logger.info("Generating advanced performance assessment")
        
        # Calculate component scores with weighted algorithm
        gpu_score = self._calculate_gpu_score(hardware_specs, game_requirements or {})
        cpu_score = self._calculate_cpu_score(hardware_specs, game_requirements or {})
        ram_score = self._calculate_ram_score(hardware_specs, game_requirements or {})
        
        # Weighted scoring: GPU 60%, CPU 25%, RAM 15%
        overall_score = int(gpu_score * 0.60 + cpu_score * 0.25 + ram_score * 0.15)
        
        # Determine performance tier
        tier = self._determine_tier(overall_score)
        
        # Calculate expected FPS
        expected_fps = self._calculate_expected_fps(tier, gpu_score, cpu_score)
        
        # Determine settings and resolution
        recommended_settings, recommended_resolution = self._determine_recommendations(tier, overall_score)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(cpu_score, gpu_score, ram_score)
        
        # Generate upgrade suggestions
        upgrade_suggestions = self._generate_upgrade_suggestions(
            hardware_specs, cpu_score, gpu_score, ram_score, tier
        )
        
        assessment = PerformanceAssessment(
            score=overall_score,
            tier=tier,
            tier_description=self.tier_descriptions[tier],
            expected_fps=expected_fps,
            recommended_settings=recommended_settings,
            recommended_resolution=recommended_resolution,
            bottlenecks=bottlenecks,
            upgrade_suggestions=upgrade_suggestions
        )
        
        self.logger.info(f"Performance assessment: Score {assessment.score}, Tier {assessment.tier.name}")
        
        return assessment
    
    def predict_advanced_performance(self, hardware_specs: Dict, game_requirements: Dict = None) -> PerformanceAssessment:
        """Alias for assess_performance for backward compatibility"""
        return self.assess_performance(hardware_specs, game_requirements)
    
    def predict_performance(self, compatibility_analysis: CompatibilityAnalysis, 
                          gpu_name: str, vram_gb: int, supports_rtx: bool, 
                          supports_dlss: bool) -> PerformancePrediction:
        """Legacy method for backward compatibility"""
        # Convert to hardware specs format
        hardware_specs = {
            'gpu_model': gpu_name,
            'gpu_vram_gb': vram_gb,
            'ram_total_gb': 16,  # Default fallback
            'cpu_model': 'Unknown CPU'
        }
        
        # Get advanced assessment
        assessment = self.assess_performance(hardware_specs)
        
        # Convert to legacy format
        predictions = [{
            'resolution': assessment.recommended_resolution,
            'settings': assessment.recommended_settings,
            'fps': assessment.expected_fps,
            'tier': assessment.tier.name
        }]
        
        bottleneck_info = {}
        if assessment.bottlenecks:
            for bottleneck in assessment.bottlenecks:
                if bottleneck == 'CPU':
                    bottleneck_info[ComponentType.CPU] = "CPU bottleneck detected"
                elif bottleneck == 'GPU':
                    bottleneck_info[ComponentType.GPU] = "GPU bottleneck detected"
                elif bottleneck == 'RAM':
                    bottleneck_info[ComponentType.MEMORY] = "RAM bottleneck detected"
        
        return PerformancePrediction(
            game_name=compatibility_analysis.game_name,
            predictions=predictions,
            bottleneck_info=bottleneck_info,
            optimization_suggestions=assessment.upgrade_suggestions,
            dlss_support=supports_dlss,
            rtx_support=supports_rtx,
            overall_performance_score=ssessment.score
        )
    
    def _calculate_gpu_score(self, hardware_specs: Dict, game_requirements: Dict) -> int:
        """Calculate GPU performance score"""
        gpu_name = getattr(hardware_specs, 'gpu_model', '') if hasattr(hardware_specs, 'gpu_model') else getattr(hardware_specs, 'gpu_model', '') if isinstance(hardware_specs, dict) else ''
        gpu_key = self._find_gpu_key(gpu_name)
        
        if not gpu_key:
            # Fallback: estimate based on VRAM
            vram = getattr(hardware_specs, 'gpu_vram_gb', 4)
            if vram >= 16:
                return 85
            elif vram >= 12:
                return 75
            elif vram >= 8:
                return 65
            elif vram >= 6:
                return 55
            elif vram >= 4:
                return 45
            else:
                return 35
        
        return self.gpu_hierarchy.get(gpu_key, 50)
    
    def _calculate_cpu_score(self, hardware_specs: Dict, game_requirements: Dict) -> int:
        """Calculate CPU performance score"""
        cpu_name = getattr(hardware_specs, 'cpu_model', '')
        cpu_key = self._find_cpu_key(cpu_name)
        
        if not cpu_key:
            # Fallback: estimate based on cores
            cores = getattr(hardware_specs, 'cpu_cores', 4)
            if cores >= 16:
                return 85
            elif cores >= 8:
                return 75
            elif cores >= 6:
                return 65
            elif cores >= 4:
                return 55
            else:
                return 45
        
        return self.cpu_hierarchy.get(cpu_key, 60)
    
    def _calculate_ram_score(self, hardware_specs: Dict, game_requirements: Dict) -> int:
        """Calculate RAM performance score"""
        ram_gb = getattr(hardware_specs, 'ram_total_gb', 8)
        
        # Score based on amount
        if ram_gb >= 32:
            score = 100
        elif ram_gb >= 16:
            score = 90
        elif ram_gb >= 12:
            score = 75
        elif ram_gb >= 8:
            score = 60
        elif ram_gb >= 6:
            score = 45
        else:
            score = 30
        
        # Bonus for fast RAM
        ram_speed = getattr(hardware_specs, 'ram_speed_mhz', 2400)
        if ram_speed >= 4800:
            score += 5
        elif ram_speed >= 3600:
            score += 3
        elif ram_speed >= 3200:
            score += 2
        
        return min(100, score)
    
    def _determine_tier(self, score: int) -> PerformanceTier:
        """Determine performance tier based on score"""
        for tier in PerformanceTier:
            min_score, max_score, _ = tier.value
            if min_score <= score <= max_score:
                return tier
        return PerformanceTier.F
    
    def _calculate_expected_fps(self, tier: PerformanceTier, gpu_score: int, cpu_score: int) -> int:
        """Calculate expected FPS based on tier and component scores"""
        base_fps = {
            PerformanceTier.S: 90,
            PerformanceTier.A: 75,
            PerformanceTier.B: 60,
            PerformanceTier.C: 40,
            PerformanceTier.D: 30,
            PerformanceTier.F: 20
        }
        
        fps = base_fps.get(tier, 30)
        
        # Adjust based on GPU performance
        if gpu_score >= 90:
            fps += 20
        elif gpu_score >= 80:
            fps += 10
        elif gpu_score <= 50:
            fps -= 10
        
        # Slight CPU adjustment
        if cpu_score >= 90:
            fps += 5
        elif cpu_score <= 50:
            fps -= 5
        
        return max(15, fps)
    
    def _determine_recommendations(self, tier: PerformanceTier, score: int) -> Tuple[str, str]:
        """Determine recommended settings and resolution"""
        if tier == PerformanceTier.S:
            return "Ultra/Maximum", "4K (3840x2160)"
        elif tier == PerformanceTier.A:
            return "High", "1440p (2560x1440)"
        elif tier == PerformanceTier.B:
            return "High", "1080p (1920x1080)"
        elif tier == PerformanceTier.C:
            return "Medium", "1080p (1920x1080)"
        elif tier == PerformanceTier.D:
            return "Low", "720p (1280x720)"
        else:
            return "Minimum", "720p (1280x720)"
    
    def _identify_bottlenecks(self, cpu_score: int, gpu_score: int, ram_score: int) -> list:
        """Identify system bottlenecks"""
        bottlenecks = []
        scores = {'CPU': cpu_score, 'GPU': gpu_score, 'RAM': ram_score}
        
        # Find the lowest scoring component(s)
        min_score = min(scores.values())
        avg_score = sum(scores.values()) / len(scores)
        
        for component, score in scores.items():
            if score <= min_score + 5 and score < avg_score - 10:
                bottlenecks.append(component)
        
        return bottlenecks
    
    def _generate_upgrade_suggestions(self, hardware_specs: Dict, cpu_score: int, 
                                    gpu_score: int, ram_score: int, tier: PerformanceTier) -> list:
        """Generate hardware upgrade suggestions"""
        suggestions = []
        
        # GPU upgrades
        if gpu_score < 70:
            if tier == PerformanceTier.F or tier == PerformanceTier.D:
                suggestions.append("GPU upgrade essential - Consider RTX 3060 or better")
            elif tier == PerformanceTier.C:
                suggestions.append("GPU upgrade recommended - Consider RTX 3070 or RX 6700 XT")
        
        # CPU upgrades
        if cpu_score < 65:
            suggestions.append("CPU upgrade recommended for better performance")
        
        # RAM upgrades
        ram_gb = getattr(hardware_specs, 'ram_total_gb', 8)
        
        # Account for system-reserved memory (30+ GB typically means 32GB installed)
        effective_ram = ram_gb
        if ram_gb >= 30:
            effective_ram = 32
        elif ram_gb >= 14:
            effective_ram = 16
        elif ram_gb >= 6:
            effective_ram = 8
        
        if effective_ram < 16:
            suggestions.append("Upgrade to 16GB+ RAM for optimal performance")
        elif effective_ram < 32 and tier == PerformanceTier.S:
            suggestions.append("Consider 32GB RAM for maximum performance")
        
        # Storage suggestions
        storage_type = getattr(hardware_specs, 'storage_type', 'Unknown')
        if 'HDD' in storage_type:
            suggestions.append("Upgrade to SSD/NVMe for faster loading times")
        
        return suggestions
    
    def _find_cpu_key(self, cpu_string: str) -> Optional[str]:
        """Find matching CPU key from hierarchy"""
        cpu_upper = cpu_string.upper()
        
        # Direct matches first
        for key in self.cpu_hierarchy:
            if key.upper() in cpu_upper:
                return key
        
        # Pattern matching for common CPU naming
        for key in self.cpu_hierarchy:
            key_parts = key.upper().replace('-', ' ').split()
            if all(part in cpu_upper for part in key_parts):
                return key
        
        return None
    
    def _find_gpu_key(self, gpu_string: str) -> Optional[str]:
        """Find matching GPU key from hierarchy"""
        gpu_upper = gpu_string.upper()
        
        # Direct matches first
        for key in self.gpu_hierarchy:
            if key.upper() in gpu_upper:
                return key
        
        # Pattern matching for GPU naming
        for key in self.gpu_hierarchy:
            key_clean = key.upper().replace(' ', '')
            gpu_clean = gpu_upper.replace(' ', '')
            if key_clean in gpu_clean:
                return key
        
        return None