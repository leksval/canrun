"""
Advanced Performance Assessment System for CanRun
Implements comprehensive tiered scoring with hardware hierarchies and weighted algorithms.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import re
import logging


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
    """Complete performance assessment result"""
    score: int
    tier: PerformanceTier
    expected_fps: int
    recommended_settings: str
    recommended_resolution: str
    bottlenecks: list
    upgrade_suggestions: list


class AdvancedPerformanceAssessor:
    """
    Advanced performance assessment using tiered scoring system
    with comprehensive hardware hierarchies and weighted algorithms.
    """
    
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
            
            # Older GPUs
            "GTX 980 Ti": 42, "GTX 980": 38, "GTX 970": 35,
            "RX 580": 46, "RX 570": 42, "RX 560": 35
        }
        
        # Comprehensive CPU hierarchy
        self.cpu_hierarchy = {
            # Intel 13th Gen
            "i9-13900K": 100, "i7-13700K": 92, "i5-13600K": 85,
            "i5-13400": 78, "i3-13100": 65,
            
            # Intel 12th Gen
            "i9-12900K": 95, "i7-12700K": 88, "i5-12600K": 82,
            "i5-12400": 75, "i3-12100": 62,
            
            # Intel 11th Gen
            "i9-11900K": 85, "i7-11700K": 80, "i5-11600K": 72,
            "i5-11400": 68, "i3-11100": 55,
            
            # AMD Ryzen 7000
            "Ryzen 9 7950X": 98, "Ryzen 9 7900X": 94, "Ryzen 7 7700X": 90,
            "Ryzen 5 7600X": 84, "Ryzen 5 7600": 80,
            
            # AMD Ryzen 5000
            "Ryzen 9 5950X": 93, "Ryzen 9 5900X": 90, "Ryzen 7 5800X3D": 92,
            "Ryzen 7 5800X": 86, "Ryzen 5 5600X": 80, "Ryzen 5 5600": 76,
            
            # AMD Ryzen 3000
            "Ryzen 9 3900X": 78, "Ryzen 7 3700X": 72, "Ryzen 5 3600X": 68,
            "Ryzen 5 3600": 65,
            
            # Older Intel
            "i7-10700K": 75, "i5-10600K": 68, "i7-9700K": 70,
            "i5-9600K": 62, "i7-8700K": 65, "i5-8600K": 58
        }
        
        # Performance tier details
        self.tier_details = {
            PerformanceTier.S: {
                "resolution": "4K (3840x2160)",
                "settings": "Ultra/Maximum",
                "fps_range": "60-120+ FPS",
                "features": ["Ray Tracing", "DLSS/FSR", "HDR", "All effects enabled"]
            },
            PerformanceTier.A: {
                "resolution": "1440p (2560x1440)",
                "settings": "High/Very High",
                "fps_range": "60-90 FPS",
                "features": ["Most effects enabled", "Some Ray Tracing", "DLSS/FSR available"]
            },
            PerformanceTier.B: {
                "resolution": "1080p (1920x1080)",
                "settings": "High",
                "fps_range": "60-75 FPS",
                "features": ["Good visual quality", "Stable performance", "Most features enabled"]
            },
            PerformanceTier.C: {
                "resolution": "1080p (1920x1080)",
                "settings": "Medium",
                "fps_range": "30-45 FPS",
                "features": ["Acceptable visuals", "Some compromises", "Core features enabled"]
            },
            PerformanceTier.D: {
                "resolution": "720p-1080p",
                "settings": "Low/Medium",
                "fps_range": "30 FPS",
                "features": ["Basic visuals", "Playable experience", "Many features disabled"]
            },
            PerformanceTier.F: {
                "resolution": "Below 720p",
                "settings": "Lowest",
                "fps_range": "Below 30 FPS",
                "features": ["Severe compromises", "Poor experience", "Not recommended"]
            }
        }
    
    def assess_performance(self, hardware_specs: Dict, game_requirements: Dict) -> PerformanceAssessment:
        """
        Perform comprehensive performance assessment
        
        Args:
            hardware_specs: User's hardware specifications
            game_requirements: Game's system requirements
            
        Returns:
            PerformanceAssessment with detailed results
        """
        # Calculate component scores
        cpu_score = self._calculate_cpu_score(hardware_specs, game_requirements)
        gpu_score = self._calculate_gpu_score(hardware_specs, game_requirements)
        ram_score = self._calculate_ram_score(hardware_specs, game_requirements)
        
        # Weighted average (GPU is most important for gaming)
        total_score = int(
            gpu_score * 0.6 +  # GPU weight: 60%
            cpu_score * 0.25 + # CPU weight: 25%
            ram_score * 0.15   # RAM weight: 15%
        )
        
        # Determine tier
        tier = self._determine_tier(total_score)
        
        # Calculate expected FPS
        expected_fps = self._calculate_expected_fps(tier, gpu_score, cpu_score)
        
        # Get performance details
        details = self.tier_details[tier]
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(cpu_score, gpu_score, ram_score)
        
        # Generate upgrade suggestions
        upgrade_suggestions = self._generate_upgrade_suggestions(
            hardware_specs, cpu_score, gpu_score, ram_score, tier
        )
        
        self.logger.info(f"Performance assessment: Score {total_score}, Tier {tier.name}")
        
        return PerformanceAssessment(
            score=total_score,
            tier=tier,
            expected_fps=expected_fps,
            recommended_settings=details["settings"],
            recommended_resolution=details["resolution"],
            bottlenecks=bottlenecks,
            upgrade_suggestions=upgrade_suggestions
        )
    
    def _calculate_cpu_score(self, hardware_specs: Dict, game_requirements: Dict) -> int:
        """Calculate CPU performance score"""
        cpu_name = hardware_specs.get('cpu_model', '')
        cpu_key = self._find_cpu_key(cpu_name)
        
        if not cpu_key:
            # Fallback: estimate based on cores/threads
            cores = hardware_specs.get('cpu_cores', 4)
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
        
        base_score = self.cpu_hierarchy.get(cpu_key, 60)
        
        # Adjust based on game requirements if available
        # For now, return base score (can be enhanced with requirement parsing)
        return min(100, base_score)
    
    def _calculate_gpu_score(self, hardware_specs: Dict, game_requirements: Dict) -> int:
        """Calculate GPU performance score"""
        gpu_name = hardware_specs.get('gpu_model', '')
        gpu_key = self._find_gpu_key(gpu_name)
        
        if not gpu_key:
            # Fallback: estimate based on VRAM
            vram = hardware_specs.get('gpu_vram_gb', 4)
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
        
        base_score = self.gpu_hierarchy.get(gpu_key, 50)
        
        # Adjust for VRAM if needed
        vram = hardware_specs.get('gpu_vram_gb', 8)
        if vram < 6 and base_score > 70:
            base_score -= 10  # Penalize high-end GPUs with low VRAM
        
        return min(100, base_score)
    
    def _calculate_ram_score(self, hardware_specs: Dict, game_requirements: Dict) -> int:
        """Calculate RAM performance score"""
        ram_gb = hardware_specs.get('ram_total_gb', 8)
        
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
        ram_speed = hardware_specs.get('ram_speed_mhz', 2400)
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
        ram_gb = hardware_specs.get('ram_total_gb', 8)
        
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
        storage_type = hardware_specs.get('storage_type', 'Unknown')
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
        
        # Pattern matching for GPU naming variations
        for key in self.gpu_hierarchy:
            key_clean = key.upper().replace(' ', '')
            gpu_clean = gpu_upper.replace(' ', '')
            if key_clean in gpu_clean:
                return key
        
        return None
    
    def get_tier_description(self, tier: PerformanceTier) -> Dict:
        """Get detailed description for a performance tier"""
        return self.tier_details.get(tier, {})