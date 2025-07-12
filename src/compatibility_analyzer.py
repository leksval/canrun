"""
Compatibility Analysis Engine for CanRun
Compatibility analysis for RTX/GTX gaming systems with G-Assist integration.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from privacy_aware_hardware_detector import PrivacyAwareHardwareSpecs
from game_requirements_fetcher import GameRequirements


class CompatibilityLevel(Enum):
    """Compatibility levels for RTX/GTX gaming systems."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    ADEQUATE = "Adequate"
    POOR = "Poor"
    INCOMPATIBLE = "Incompatible"


class ComponentType(Enum):
    """Hardware component types for RTX/GTX gaming analysis."""
    GPU = "GPU"
    CPU = "CPU"
    RAM = "RAM"
    STORAGE = "Storage"
    OS = "OS"
    DIRECTX = "DirectX"


@dataclass
class ComponentAnalysis:
    """Analysis result for a single hardware component."""
    component: ComponentType
    meets_minimum: bool
    meets_recommended: bool
    score: float  # 0-1 scale
    bottleneck_factor: float  # 0-1 scale (1 = major bottleneck)
    details: str
    upgrade_suggestion: Optional[str] = None

    def __post_init__(self):
        """Validate component analysis after initialization."""
        assert 0.0 <= self.score <= 1.0, "Score must be between 0 and 1"
        assert 0.0 <= self.bottleneck_factor <= 1.0, "Bottleneck factor must be between 0 and 1"
        assert self.details.strip(), "Details cannot be empty"


@dataclass
class CompatibilityAnalysis:
    """Complete RTX/GTX gaming compatibility analysis result."""
    game_name: str
    overall_compatibility: CompatibilityLevel
    can_run_minimum: bool
    can_run_recommended: bool
    component_analyses: List[ComponentAnalysis]
    bottlenecks: List[ComponentType]
    overall_score: float
    summary: str
    recommendations: List[str]

    def __post_init__(self):
        """Validate compatibility analysis after initialization."""
        assert self.game_name.strip(), "Game name cannot be empty"
        assert 0.0 <= self.overall_score <= 1.0, "Overall score must be between 0 and 1"
        assert self.component_analyses, "Component analyses cannot be empty"
        assert self.summary.strip(), "Summary cannot be empty"


class CompatibilityAnalyzer:
    """Compatibility analyzer for RTX/GTX gaming systems."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # RTX/GTX-focused component weights for gaming performance
        self.component_weights = {
            ComponentType.GPU: 0.45,    # Higher weight for GPU in gaming
            ComponentType.CPU: 0.30,    # Important for modern games
            ComponentType.RAM: 0.15,    # Memory requirements
            ComponentType.STORAGE: 0.05, # Less critical for analysis
            ComponentType.OS: 0.03,     # Usually compatible
            ComponentType.DIRECTX: 0.02  # DirectX support
        }
        
        # RTX/GTX GPU performance tiers
        self.nvidia_gpu_tiers = {
            # RTX 40 Series
            'rtx 4090': 100, 'rtx 4080': 90, 'rtx 4070 ti': 80, 'rtx 4070': 75,
            'rtx 4060 ti': 65, 'rtx 4060': 60,
            # RTX 30 Series
            'rtx 3090': 95, 'rtx 3080': 85, 'rtx 3070': 70, 'rtx 3060 ti': 60,
            'rtx 3060': 55, 'rtx 3050': 45,
            # RTX 20 Series
            'rtx 2080 ti': 80, 'rtx 2080': 70, 'rtx 2070': 60, 'rtx 2060': 50,
            # GTX 16 Series
            'gtx 1660 ti': 45, 'gtx 1660': 40, 'gtx 1650': 30,
            # GTX 10 Series
            'gtx 1080 ti': 65, 'gtx 1080': 55, 'gtx 1070': 45, 'gtx 1060': 35,
            'gtx 1050': 25
        }
        
        self.logger.info("RTX/GTX compatibility analyzer initialized")
    
    def analyze_compatibility(self, game_name: str, hardware: PrivacyAwareHardwareSpecs,
                            requirements: GameRequirements) -> CompatibilityAnalysis:
        """Perform complete RTX/GTX gaming compatibility analysis."""
        # Validate inputs
        assert game_name and game_name.strip(), "Game name must be provided"
        assert hardware.is_nvidia_gpu, "RTX/GTX GPU required for G-Assist compatibility"
        assert requirements.game_name.strip(), "Game requirements must be valid"
        
        # Analyze each component with RTX/GTX focus
        component_analyses = [
            self._analyze_nvidia_gpu(hardware, requirements),
            self._analyze_cpu(hardware, requirements),
            self._analyze_ram(hardware, requirements),
            self._analyze_storage(hardware, requirements),
            self._analyze_os(hardware, requirements),
            self._analyze_directx(hardware, requirements)
        ]
        
        # Calculate overall compatibility
        overall_score = self._calculate_overall_score(component_analyses)
        overall_compatibility = self._determine_compatibility_level(overall_score)
        
        # Determine run capabilities
        can_run_minimum = all(c.meets_minimum for c in component_analyses)
        can_run_recommended = all(c.meets_recommended for c in component_analyses)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(component_analyses)
        
        # Generate summary and recommendations
        summary = self._generate_summary(overall_compatibility, can_run_minimum, 
                                       can_run_recommended, bottlenecks)
        recommendations = self._generate_recommendations(component_analyses, bottlenecks, hardware)
        
        return CompatibilityAnalysis(
            game_name=requirements.game_name,
            overall_compatibility=overall_compatibility,
            can_run_minimum=can_run_minimum,
            can_run_recommended=can_run_recommended,
            component_analyses=component_analyses,
            bottlenecks=bottlenecks,
            overall_score=overall_score,
            summary=summary,
            recommendations=recommendations
        )
    
    def _analyze_nvidia_gpu(self, hardware: PrivacyAwareHardwareSpecs,
                          requirements: GameRequirements) -> ComponentAnalysis:
        """Analyze RTX/GTX GPU compatibility."""
        assert hardware.is_nvidia_gpu, "RTX/GTX GPU required"
        
        # Get GPU performance score
        gpu_score = self._get_nvidia_gpu_score(hardware.gpu_name)
        
        # Estimate required scores from requirements
        min_gpu_text = requirements.minimum.get('graphics', '').lower()
        rec_gpu_text = requirements.recommended.get('graphics', '').lower()
        
        min_score = self._estimate_required_gpu_score(min_gpu_text)
        rec_score = self._estimate_required_gpu_score(rec_gpu_text)
        
        # Check compatibility
        meets_minimum = gpu_score >= min_score
        meets_recommended = gpu_score >= rec_score
        
        # Calculate performance metrics
        score = min(1.0, gpu_score / max(rec_score, 1))
        bottleneck_factor = max(0.0, (min_score - gpu_score) / max(min_score, 1))
        
        # Generate details with RTX/DLSS features
        rtx_features = []
        if hardware.supports_rtx:
            rtx_features.append("RTX Ray Tracing")
        if hardware.supports_dlss:
            rtx_features.append("DLSS")
        
        details = f"NVIDIA {hardware.gpu_name} ({hardware.vram_gb}GB VRAM"
        if rtx_features:
            details += f", {', '.join(rtx_features)}"
        details += ")"
        
        if meets_recommended:
            details += " - Exceeds recommended requirements"
        elif meets_minimum:
            details += " - Meets minimum requirements"
        else:
            details += " - Below minimum requirements"
        
        # Generate upgrade suggestion
        upgrade_suggestion = None
        if not meets_minimum:
            upgrade_suggestion = "Consider upgrading to a more powerful RTX GPU"
        elif not meets_recommended:
            upgrade_suggestion = "RTX upgrade recommended for better performance and ray tracing"
        
        return ComponentAnalysis(
            component=ComponentType.GPU,
            meets_minimum=meets_minimum,
            meets_recommended=meets_recommended,
            score=score,
            bottleneck_factor=bottleneck_factor,
            details=details,
            upgrade_suggestion=upgrade_suggestion
        )
    
    def _analyze_cpu(self, hardware: PrivacyAwareHardwareSpecs,
                    requirements: GameRequirements) -> ComponentAnalysis:
        """Analyze CPU compatibility for RTX/GTX gaming."""
        assert hardware.cpu_cores > 0, "CPU cores must be greater than 0"
        assert hardware.cpu_threads > 0, "CPU threads must be greater than 0"
        
        # Estimate CPU performance
        cpu_score = self._estimate_cpu_performance(hardware.cpu_name, hardware.cpu_cores, hardware.cpu_threads)
        
        # Get required scores
        min_cpu_text = requirements.minimum.get('processor', '').lower()
        rec_cpu_text = requirements.recommended.get('processor', '').lower()
        
        min_score = self._estimate_required_cpu_score(min_cpu_text)
        rec_score = self._estimate_required_cpu_score(rec_cpu_text)
        
        # Check compatibility
        meets_minimum = cpu_score >= min_score
        meets_recommended = cpu_score >= rec_score
        
        # Calculate metrics
        score = min(1.0, cpu_score / max(rec_score, 1))
        bottleneck_factor = max(0.0, (min_score - cpu_score) / max(min_score, 1))
        
        # Generate details
        details = f"CPU: {hardware.cpu_name} ({hardware.cpu_cores}C/{hardware.cpu_threads}T)"
        
        if meets_recommended:
            details += " - Exceeds recommended requirements"
        elif meets_minimum:
            details += " - Meets minimum requirements"
        else:
            details += " - Below minimum requirements"
        
        # Generate upgrade suggestion
        upgrade_suggestion = None
        if not meets_minimum:
            upgrade_suggestion = "Consider upgrading to a faster CPU"
        elif not meets_recommended:
            upgrade_suggestion = "CPU upgrade recommended for optimal NVIDIA gaming performance"
        
        return ComponentAnalysis(
            component=ComponentType.CPU,
            meets_minimum=meets_minimum,
            meets_recommended=meets_recommended,
            score=score,
            bottleneck_factor=bottleneck_factor,
            details=details,
            upgrade_suggestion=upgrade_suggestion
        )
    
    def _analyze_ram(self, hardware: PrivacyAwareHardwareSpecs, 
                    requirements: GameRequirements) -> ComponentAnalysis:
        """Analyze RAM compatibility."""
        assert hardware.ram_gb > 0, "RAM must be greater than 0"
        
        # Extract required RAM amounts
        min_ram = self._extract_ram_amount(requirements.minimum.get('memory', ''))
        rec_ram = self._extract_ram_amount(requirements.recommended.get('memory', ''))
        
        # Check compatibility
        meets_minimum = hardware.ram_gb >= min_ram
        meets_recommended = hardware.ram_gb >= rec_ram
        
        # Calculate metrics
        score = min(1.0, hardware.ram_gb / max(rec_ram, 1))
        bottleneck_factor = max(0.0, (min_ram - hardware.ram_gb) / max(min_ram, 1))
        
        # Generate details
        details = f"RAM: {hardware.ram_gb}GB"
        
        if meets_recommended:
            details += " - Sufficient for recommended settings"
        elif meets_minimum:
            details += " - Meets minimum requirements"
        else:
            details += " - Insufficient RAM"
        
        # Generate upgrade suggestion
        upgrade_suggestion = None
        if not meets_minimum:
            upgrade_suggestion = f"Add more RAM (need at least {min_ram}GB)"
        elif not meets_recommended:
            upgrade_suggestion = f"Consider upgrading to {rec_ram}GB for better performance"
        
        return ComponentAnalysis(
            component=ComponentType.RAM,
            meets_minimum=meets_minimum,
            meets_recommended=meets_recommended,
            score=score,
            bottleneck_factor=bottleneck_factor,
            details=details,
            upgrade_suggestion=upgrade_suggestion
        )
    
    def _analyze_storage(self, hardware: PrivacyAwareHardwareSpecs, 
                        requirements: GameRequirements) -> ComponentAnalysis:
        """Analyze storage compatibility."""
        # Extract required storage amounts
        min_storage = self._extract_storage_amount(requirements.minimum.get('storage', ''))
        rec_storage = self._extract_storage_amount(requirements.recommended.get('storage', ''))
        
        # For this analysis, assume adequate storage is available
        # In production, this would check actual disk space
        meets_minimum = True
        meets_recommended = True
        score = 1.0
        bottleneck_factor = 0.0
        
        details = f"Storage: {min_storage}GB required"
        if rec_storage > min_storage:
            details += f" ({rec_storage}GB recommended)"
        
        return ComponentAnalysis(
            component=ComponentType.STORAGE,
            meets_minimum=meets_minimum,
            meets_recommended=meets_recommended,
            score=score,
            bottleneck_factor=bottleneck_factor,
            details=details
        )
    
    def _analyze_os(self, hardware: PrivacyAwareHardwareSpecs, 
                   requirements: GameRequirements) -> ComponentAnalysis:
        """Analyze OS compatibility for NVIDIA gaming."""
        assert hardware.os_name.strip(), "OS name cannot be empty"
        
        # Check OS compatibility
        min_os = requirements.minimum.get('os', '').lower()
        rec_os = requirements.recommended.get('os', '').lower()
        
        is_windows = 'windows' in hardware.os_name.lower()
        meets_minimum = is_windows and ('windows' in min_os or not min_os)
        meets_recommended = is_windows and ('windows' in rec_os or not rec_os)
        
        score = 1.0 if meets_minimum else 0.0
        bottleneck_factor = 0.0 if meets_minimum else 1.0
        
        details = f"OS: {hardware.os_name}"
        if meets_minimum:
            details += " - Compatible with G-Assist"
        else:
            details += " - May not be compatible with G-Assist"
        
        upgrade_suggestion = None
        if not meets_minimum:
            upgrade_suggestion = "Windows OS recommended for full G-Assist compatibility"
        
        return ComponentAnalysis(
            component=ComponentType.OS,
            meets_minimum=meets_minimum,
            meets_recommended=meets_recommended,
            score=score,
            bottleneck_factor=bottleneck_factor,
            details=details,
            upgrade_suggestion=upgrade_suggestion
        )
    
    def _analyze_directx(self, hardware: PrivacyAwareHardwareSpecs, 
                        requirements: GameRequirements) -> ComponentAnalysis:
        """Analyze DirectX compatibility."""
        assert hardware.directx_version.strip(), "DirectX version cannot be empty"
        
        # Extract version numbers
        hardware_dx_version = self._extract_directx_version(hardware.directx_version)
        min_dx_version = self._extract_directx_version(requirements.minimum.get('directx', ''))
        rec_dx_version = self._extract_directx_version(requirements.recommended.get('directx', ''))
        
        meets_minimum = hardware_dx_version >= min_dx_version
        meets_recommended = hardware_dx_version >= rec_dx_version
        
        score = 1.0 if meets_minimum else 0.0
        bottleneck_factor = 0.0 if meets_minimum else 0.5
        
        details = f"DirectX: {hardware.directx_version}"
        if meets_recommended:
            details += " - Fully supported"
        elif meets_minimum:
            details += " - Minimum version supported"
        else:
            details += " - Version may be insufficient"
        
        upgrade_suggestion = None
        if not meets_minimum:
            upgrade_suggestion = "Update DirectX to the latest version"
        
        return ComponentAnalysis(
            component=ComponentType.DIRECTX,
            meets_minimum=meets_minimum,
            meets_recommended=meets_recommended,
            score=score,
            bottleneck_factor=bottleneck_factor,
            details=details,
            upgrade_suggestion=upgrade_suggestion
        )
    
    def _get_nvidia_gpu_score(self, gpu_name: str) -> int:
        """Get NVIDIA GPU performance score."""
        assert gpu_name.strip(), "GPU name cannot be empty"
        
        gpu_lower = gpu_name.lower()
        
        # Check against known NVIDIA GPU tiers
        for gpu_key, score in self.nvidia_gpu_tiers.items():
            if gpu_key in gpu_lower:
                return score
        
        # Fallback estimation based on GPU name patterns
        if 'rtx 40' in gpu_lower:
            return 70  # Average RTX 40 series
        elif 'rtx 30' in gpu_lower:
            return 60  # Average RTX 30 series
        elif 'rtx 20' in gpu_lower:
            return 50  # Average RTX 20 series
        elif 'gtx 16' in gpu_lower:
            return 40  # Average GTX 16 series
        elif 'gtx 10' in gpu_lower:
            return 35  # Average GTX 10 series
        else:
            return 30  # Conservative estimate for unknown NVIDIA GPUs
    
    def _estimate_cpu_performance(self, cpu_name: str, cores: int, threads: int) -> int:
        """Estimate CPU performance score."""
        assert cpu_name.strip(), "CPU name cannot be empty"
        assert cores > 0, "CPU cores must be greater than 0"
        assert threads > 0, "CPU threads must be greater than 0"
        
        cpu_lower = cpu_name.lower()
        base_score = 50  # Default score
        
        # Intel processors
        if 'intel' in cpu_lower:
            if 'i9' in cpu_lower:
                base_score = 90
            elif 'i7' in cpu_lower:
                base_score = 80
            elif 'i5' in cpu_lower:
                base_score = 70
            elif 'i3' in cpu_lower:
                base_score = 60
        
        # AMD processors
        elif 'amd' in cpu_lower:
            if 'ryzen 9' in cpu_lower:
                base_score = 90
            elif 'ryzen 7' in cpu_lower:
                base_score = 80
            elif 'ryzen 5' in cpu_lower:
                base_score = 70
            elif 'ryzen 3' in cpu_lower:
                base_score = 60
        
        # Adjust for core count
        core_multiplier = min(1.5, cores / 4)  # Cap at 1.5x for 4+ cores
        thread_multiplier = min(1.2, threads / cores)  # Hyperthreading bonus
        
        return int(base_score * core_multiplier * thread_multiplier)
    
    def _calculate_overall_score(self, component_analyses: List[ComponentAnalysis]) -> float:
        """Calculate weighted overall performance score."""
        assert component_analyses, "Component analyses cannot be empty"
        
        total_score = 0.0
        total_weight = 0.0
        
        for analysis in component_analyses:
            weight = self.component_weights.get(analysis.component, 0.1)
            total_score += analysis.score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_compatibility_level(self, score: float) -> CompatibilityLevel:
        """Determine compatibility level based on score."""
        assert 0.0 <= score <= 1.0, "Score must be between 0 and 1"
        
        if score >= 0.9:
            return CompatibilityLevel.EXCELLENT
        elif score >= 0.7:
            return CompatibilityLevel.GOOD
        elif score >= 0.5:
            return CompatibilityLevel.ADEQUATE
        elif score >= 0.3:
            return CompatibilityLevel.POOR
        else:
            return CompatibilityLevel.INCOMPATIBLE
    
    def _identify_bottlenecks(self, component_analyses: List[ComponentAnalysis]) -> List[ComponentType]:
        """Identify component bottlenecks."""
        assert component_analyses, "Component analyses cannot be empty"
        
        bottlenecks = []
        for analysis in component_analyses:
            if analysis.bottleneck_factor > 0.3:  # Bottleneck threshold
                bottlenecks.append(analysis.component)
        
        return bottlenecks
    
    def _generate_summary(self, compatibility: CompatibilityLevel, can_run_min: bool, 
                         can_run_rec: bool, bottlenecks: List[ComponentType]) -> str:
        """Generate NVIDIA gaming compatibility summary."""
        if compatibility == CompatibilityLevel.EXCELLENT:
            return "Your NVIDIA RTX/GTX system exceeds recommended requirements and will run this game excellently with full G-Assist support."
        elif compatibility == CompatibilityLevel.GOOD:
            return "Your NVIDIA RTX/GTX system meets recommended requirements and will run this game well with G-Assist features."
        elif compatibility == CompatibilityLevel.ADEQUATE:
            return "Your NVIDIA RTX/GTX system meets minimum requirements but may need setting adjustments for optimal performance."
        elif compatibility == CompatibilityLevel.POOR:
            return "Your NVIDIA RTX/GTX system barely meets requirements and may experience performance issues."
        else:
            return "Your NVIDIA RTX/GTX system does not meet minimum requirements for this game."
    
    def _generate_recommendations(self, component_analyses: List[ComponentAnalysis], 
                                bottlenecks: List[ComponentType], 
                                hardware: PrivacyAwareHardwareSpecs) -> List[str]:
        """Generate NVIDIA gaming recommendations."""
        recommendations = []
        
        # Add component-specific recommendations
        for analysis in component_analyses:
            if analysis.upgrade_suggestion:
                recommendations.append(analysis.upgrade_suggestion)
        
        # Add NVIDIA-specific recommendations
        if ComponentType.GPU in bottlenecks:
            recommendations.append("Consider upgrading to a newer NVIDIA RTX GPU for better ray tracing and DLSS performance")
        
        # Add RTX-specific features
        if hardware.supports_rtx and ComponentType.GPU not in bottlenecks:
            recommendations.append("Enable ray tracing if supported by the game for enhanced visual quality")
        
        if hardware.supports_dlss and ComponentType.GPU not in bottlenecks:
            recommendations.append("Enable DLSS if supported by the game for improved performance")
        
        return recommendations
    
    # Helper methods for parsing game requirements
    def _extract_ram_amount(self, ram_text: str) -> int:
        """Extract RAM amount in GB from text."""
        if not ram_text:
            return 8  # Default assumption
        
        # Look for GB values
        match = re.search(r'(\d+)\s*GB', ram_text.upper())
        if match:
            return int(match.group(1))
        
        # Look for MB values and convert
        match = re.search(r'(\d+)\s*MB', ram_text.upper())
        if match:
            return max(1, int(match.group(1)) // 1024)
        
        return 8  # Default fallback
    
    def _extract_storage_amount(self, storage_text: str) -> int:
        """Extract storage amount in GB from text."""
        if not storage_text:
            return 50  # Default assumption
        
        # Look for GB values
        match = re.search(r'(\d+)\s*GB', storage_text.upper())
        if match:
            return int(match.group(1))
        
        return 50  # Default fallback
    
    def _extract_directx_version(self, dx_text: str) -> float:
        """Extract DirectX version number."""
        if not dx_text:
            return 12.0  # Default to DirectX 12
        
        # Look for version numbers
        match = re.search(r'(\d+)\.?(\d*)', dx_text.upper())
        if match:
            major = int(match.group(1))
            minor = int(match.group(2)) if match.group(2) else 0
            return major + (minor / 10)
        
        return 12.0  # Default fallback
    
    def _estimate_required_gpu_score(self, gpu_text: str) -> int:
        """Estimate required GPU score from game requirements text."""
        if not gpu_text:
            return 30  # Default minimum
        
        gpu_lower = gpu_text.lower()
        
        # Check for specific GPU mentions
        for gpu_key, score in self.nvidia_gpu_tiers.items():
            if gpu_key in gpu_lower:
                return score
        
        # Fallback patterns
        if 'rtx' in gpu_lower:
            return 50  # RTX requirement
        elif 'gtx' in gpu_lower:
            return 40  # GTX requirement
        elif 'nvidia' in gpu_lower:
            return 35  # General NVIDIA requirement
        
        return 30  # Conservative fallback
    
    def _estimate_required_cpu_score(self, cpu_text: str) -> int:
        """Estimate required CPU score from game requirements text."""
        if not cpu_text:
            return 50  # Default minimum
        
        cpu_lower = cpu_text.lower()
        
        # Intel patterns
        if 'i9' in cpu_lower:
            return 80
        elif 'i7' in cpu_lower:
            return 70
        elif 'i5' in cpu_lower:
            return 60
        elif 'i3' in cpu_lower:
            return 50
        
        # AMD patterns
        elif 'ryzen 9' in cpu_lower:
            return 80
        elif 'ryzen 7' in cpu_lower:
            return 70
        elif 'ryzen 5' in cpu_lower:
            return 60
        elif 'ryzen 3' in cpu_lower:
            return 50
        
        return 50  # Conservative fallback