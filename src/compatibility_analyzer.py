"""
Compatibility Analysis Engine for CanRun
Analyzes hardware compatibility with game requirements and identifies bottlenecks.
NVIDIA G-Assist compatible - focuses on RTX/GTX GPUs.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .hardware_detector import HardwareSpecs
from .requirements_fetcher import GameRequirements


class CompatibilityLevel(Enum):
    """Compatibility levels for games."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    ADEQUATE = "Adequate"
    POOR = "Poor"
    INCOMPATIBLE = "Incompatible"


class ComponentType(Enum):
    """Hardware component types."""
    GPU = "GPU"
    CPU = "CPU"
    RAM = "RAM"
    STORAGE = "Storage"
    OS = "OS"
    DIRECTX = "DirectX"


@dataclass
class ComponentAnalysis:
    """Analysis result for a single component."""
    component: ComponentType
    meets_minimum: bool
    meets_recommended: bool
    score: float  # 0-1 scale
    bottleneck_factor: float  # 0-1 scale (1 = major bottleneck)
    details: str
    upgrade_suggestion: Optional[str] = None


@dataclass
class CompatibilityAnalysis:
    """Complete compatibility analysis result."""
    game_name: str
    overall_compatibility: CompatibilityLevel
    can_run_minimum: bool
    can_run_recommended: bool
    component_analyses: List[ComponentAnalysis]
    bottlenecks: List[ComponentType]
    overall_score: float
    summary: str
    recommendations: List[str]


class CompatibilityAnalyzer:
    """Analyzes hardware compatibility with game requirements."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Performance weight factors for overall score calculation
        self.component_weights = {
            ComponentType.GPU: 0.4,
            ComponentType.CPU: 0.3,
            ComponentType.RAM: 0.15,
            ComponentType.STORAGE: 0.05,
            ComponentType.OS: 0.05,
            ComponentType.DIRECTX: 0.05
        }
    
    def analyze_compatibility(self, hardware: HardwareSpecs, 
                            requirements: GameRequirements) -> CompatibilityAnalysis:
        """Perform complete compatibility analysis."""
        try:
            # Check if using NVIDIA GPU (G-Assist requirement)
            is_nvidia_gpu = self._is_nvidia_gpu(hardware.gpu_name)
            
            # Analyze each component
            component_analyses = []
            
            # GPU Analysis
            gpu_analysis = self._analyze_gpu(hardware, requirements, is_nvidia_gpu)
            component_analyses.append(gpu_analysis)
            
            # CPU Analysis
            cpu_analysis = self._analyze_cpu(hardware, requirements)
            component_analyses.append(cpu_analysis)
            
            # RAM Analysis
            ram_analysis = self._analyze_ram(hardware, requirements)
            component_analyses.append(ram_analysis)
            
            # Storage Analysis
            storage_analysis = self._analyze_storage(hardware, requirements)
            component_analyses.append(storage_analysis)
            
            # OS Analysis
            os_analysis = self._analyze_os(hardware, requirements)
            component_analyses.append(os_analysis)
            
            # DirectX Analysis
            directx_analysis = self._analyze_directx(hardware, requirements)
            component_analyses.append(directx_analysis)
            
            # Calculate overall compatibility
            overall_score = self._calculate_overall_score(component_analyses)
            overall_compatibility = self._determine_compatibility_level(overall_score)
            
            # Determine if can run at minimum/recommended
            can_run_minimum = all(c.meets_minimum for c in component_analyses)
            can_run_recommended = all(c.meets_recommended for c in component_analyses)
            
            # Identify bottlenecks
            bottlenecks = self._identify_bottlenecks(component_analyses)
            
            # Generate summary and recommendations
            summary = self._generate_summary(overall_compatibility, can_run_minimum, 
                                           can_run_recommended, bottlenecks, is_nvidia_gpu)
            recommendations = self._generate_recommendations(component_analyses, bottlenecks, is_nvidia_gpu)
            
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
            
        except Exception as e:
            self.logger.error(f"Compatibility analysis failed: {e}")
            return self._create_fallback_analysis(requirements.game_name)
    
    def _is_nvidia_gpu(self, gpu_name: str) -> bool:
        """Check if GPU is NVIDIA (required for G-Assist)."""
        gpu_lower = gpu_name.lower()
        return 'nvidia' in gpu_lower or 'geforce' in gpu_lower or 'rtx' in gpu_lower or 'gtx' in gpu_lower
    
    def _analyze_gpu(self, hardware: HardwareSpecs, requirements: GameRequirements, 
                    is_nvidia: bool) -> ComponentAnalysis:
        """Analyze GPU compatibility."""
        try:
            min_gpu = requirements.minimum.get('graphics', '').lower()
            rec_gpu = requirements.recommended.get('graphics', '').lower()
            
            # Get GPU performance score from hardware detector
            from .hardware_detector import HardwareDetector
            detector = HardwareDetector()
            gpu_score = detector.get_gpu_performance_score(hardware.gpu_name)
            
            if gpu_score is None:
                # Fallback: basic name matching for NVIDIA GPUs
                gpu_score = self._estimate_nvidia_gpu_performance(hardware.gpu_name)
            
            # Estimate minimum and recommended GPU scores
            min_score = self._estimate_required_gpu_score(min_gpu)
            rec_score = self._estimate_required_gpu_score(rec_gpu)
            
            meets_minimum = gpu_score >= min_score and is_nvidia
            meets_recommended = gpu_score >= rec_score and is_nvidia
            
            # Calculate relative performance score (0-1)
            score = min(1.0, gpu_score / max(rec_score, 1)) if is_nvidia else 0.0
            
            # Calculate bottleneck factor
            bottleneck_factor = max(0, (min_score - gpu_score) / max(min_score, 1))
            if not is_nvidia:
                bottleneck_factor = 1.0  # Non-NVIDIA is a major bottleneck for G-Assist
            
            # Generate details
            details = f"GPU: {hardware.gpu_name} ({hardware.gpu_memory}MB VRAM)"
            if not is_nvidia:
                details += " - Non-NVIDIA GPU detected (G-Assist requires RTX/GTX)"
            elif meets_recommended:
                details += " - Exceeds recommended requirements"
            elif meets_minimum:
                details += " - Meets minimum requirements"
            else:
                details += " - Below minimum requirements"
            
            # Upgrade suggestion
            upgrade_suggestion = None
            if not is_nvidia:
                upgrade_suggestion = "G-Assist requires an NVIDIA RTX or GTX GPU"
            elif not meets_minimum:
                upgrade_suggestion = "Consider upgrading to a more powerful NVIDIA GPU"
            elif not meets_recommended:
                upgrade_suggestion = "NVIDIA GPU upgrade recommended for better performance"
            
            return ComponentAnalysis(
                component=ComponentType.GPU,
                meets_minimum=meets_minimum,
                meets_recommended=meets_recommended,
                score=score,
                bottleneck_factor=bottleneck_factor,
                details=details,
                upgrade_suggestion=upgrade_suggestion
            )
            
        except Exception as e:
            self.logger.error(f"GPU analysis failed: {e}")
            return self._create_fallback_component_analysis(ComponentType.GPU)
    
    def _analyze_cpu(self, hardware: HardwareSpecs, requirements: GameRequirements) -> ComponentAnalysis:
        """Analyze CPU compatibility."""
        try:
            min_cpu = requirements.minimum.get('processor', '').lower()
            rec_cpu = requirements.recommended.get('processor', '').lower()
            
            # Extract CPU performance indicators
            cpu_score = self._estimate_cpu_performance(hardware.cpu_name, hardware.cpu_cores, hardware.cpu_freq)
            min_score = self._estimate_required_cpu_score(min_cpu)
            rec_score = self._estimate_required_cpu_score(rec_cpu)
            
            meets_minimum = cpu_score >= min_score
            meets_recommended = cpu_score >= rec_score
            
            score = min(1.0, cpu_score / max(rec_score, 1))
            bottleneck_factor = max(0, (min_score - cpu_score) / max(min_score, 1))
            
            details = f"CPU: {hardware.cpu_name} ({hardware.cpu_cores} cores, {hardware.cpu_freq:.1f}GHz)"
            if meets_recommended:
                details += " - Exceeds recommended requirements"
            elif meets_minimum:
                details += " - Meets minimum requirements"
            else:
                details += " - Below minimum requirements"
            
            upgrade_suggestion = None
            if not meets_minimum:
                upgrade_suggestion = "Consider upgrading to a faster CPU"
            elif not meets_recommended:
                upgrade_suggestion = "CPU upgrade recommended for optimal performance"
            
            return ComponentAnalysis(
                component=ComponentType.CPU,
                meets_minimum=meets_minimum,
                meets_recommended=meets_recommended,
                score=score,
                bottleneck_factor=bottleneck_factor,
                details=details,
                upgrade_suggestion=upgrade_suggestion
            )
            
        except Exception as e:
            self.logger.error(f"CPU analysis failed: {e}")
            return self._create_fallback_component_analysis(ComponentType.CPU)
    
    def _analyze_ram(self, hardware: HardwareSpecs, requirements: GameRequirements) -> ComponentAnalysis:
        """Analyze RAM compatibility."""
        try:
            min_ram = self._extract_ram_amount(requirements.minimum.get('memory', ''))
            rec_ram = self._extract_ram_amount(requirements.recommended.get('memory', ''))
            
            hardware_ram_gb = hardware.ram_total / 1024  # Convert MB to GB
            
            meets_minimum = hardware_ram_gb >= min_ram
            meets_recommended = hardware_ram_gb >= rec_ram
            
            score = min(1.0, hardware_ram_gb / max(rec_ram, 1))
            bottleneck_factor = max(0, (min_ram - hardware_ram_gb) / max(min_ram, 1))
            
            details = f"RAM: {hardware_ram_gb:.1f}GB ({hardware.ram_available/1024:.1f}GB available)"
            if meets_recommended:
                details += " - Sufficient for recommended settings"
            elif meets_minimum:
                details += " - Meets minimum requirements"
            else:
                details += " - Insufficient RAM"
            
            upgrade_suggestion = None
            if not meets_minimum:
                upgrade_suggestion = f"Add more RAM (need at least {min_ram:.0f}GB)"
            elif not meets_recommended:
                upgrade_suggestion = f"Consider upgrading to {rec_ram:.0f}GB for better performance"
            
            return ComponentAnalysis(
                component=ComponentType.RAM,
                meets_minimum=meets_minimum,
                meets_recommended=meets_recommended,
                score=score,
                bottleneck_factor=bottleneck_factor,
                details=details,
                upgrade_suggestion=upgrade_suggestion
            )
            
        except Exception as e:
            self.logger.error(f"RAM analysis failed: {e}")
            return self._create_fallback_component_analysis(ComponentType.RAM)
    
    def _analyze_storage(self, hardware: HardwareSpecs, requirements: GameRequirements) -> ComponentAnalysis:
        """Analyze storage compatibility."""
        try:
            min_storage = self._extract_storage_amount(requirements.minimum.get('storage', ''))
            rec_storage = self._extract_storage_amount(requirements.recommended.get('storage', ''))
            
            # For this analysis, we assume adequate storage is available
            # In a real implementation, you'd check actual disk space
            meets_minimum = True
            meets_recommended = True
            score = 1.0
            bottleneck_factor = 0.0
            
            details = f"Storage: {min_storage:.0f}GB required"
            if rec_storage > min_storage:
                details += f" ({rec_storage:.0f}GB recommended)"
            
            return ComponentAnalysis(
                component=ComponentType.STORAGE,
                meets_minimum=meets_minimum,
                meets_recommended=meets_recommended,
                score=score,
                bottleneck_factor=bottleneck_factor,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"Storage analysis failed: {e}")
            return self._create_fallback_component_analysis(ComponentType.STORAGE)
    
    def _analyze_os(self, hardware: HardwareSpecs, requirements: GameRequirements) -> ComponentAnalysis:
        """Analyze OS compatibility."""
        try:
            min_os = requirements.minimum.get('os', '').lower()
            rec_os = requirements.recommended.get('os', '').lower()
            
            # Simple OS compatibility check
            is_windows = 'windows' in hardware.os_version.lower()
            meets_minimum = is_windows and ('windows' in min_os or not min_os)
            meets_recommended = is_windows and ('windows' in rec_os or not rec_os)
            
            score = 1.0 if meets_minimum else 0.0
            bottleneck_factor = 0.0 if meets_minimum else 1.0
            
            details = f"OS: {hardware.os_version}"
            if meets_minimum:
                details += " - Compatible"
            else:
                details += " - May not be compatible"
            
            upgrade_suggestion = None
            if not meets_minimum:
                upgrade_suggestion = "Ensure you have a compatible operating system"
            
            return ComponentAnalysis(
                component=ComponentType.OS,
                meets_minimum=meets_minimum,
                meets_recommended=meets_recommended,
                score=score,
                bottleneck_factor=bottleneck_factor,
                details=details,
                upgrade_suggestion=upgrade_suggestion
            )
            
        except Exception as e:
            self.logger.error(f"OS analysis failed: {e}")
            return self._create_fallback_component_analysis(ComponentType.OS)
    
    def _analyze_directx(self, hardware: HardwareSpecs, requirements: GameRequirements) -> ComponentAnalysis:
        """Analyze DirectX compatibility."""
        try:
            min_dx = requirements.minimum.get('directx', '').lower()
            rec_dx = requirements.recommended.get('directx', '').lower()
            
            # Extract version numbers
            hardware_dx_version = self._extract_directx_version(hardware.directx_version)
            min_dx_version = self._extract_directx_version(min_dx)
            rec_dx_version = self._extract_directx_version(rec_dx)
            
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
            
        except Exception as e:
            self.logger.error(f"DirectX analysis failed: {e}")
            return self._create_fallback_component_analysis(ComponentType.DIRECTX)
    
    def _calculate_overall_score(self, component_analyses: List[ComponentAnalysis]) -> float:
        """Calculate weighted overall performance score."""
        total_score = 0.0
        total_weight = 0.0
        
        for analysis in component_analyses:
            weight = self.component_weights.get(analysis.component, 0.1)
            total_score += analysis.score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_compatibility_level(self, score: float) -> CompatibilityLevel:
        """Determine compatibility level based on score."""
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
        bottlenecks = []
        for analysis in component_analyses:
            if analysis.bottleneck_factor > 0.3:  # Threshold for bottleneck
                bottlenecks.append(analysis.component)
        return bottlenecks
    
    def _generate_summary(self, compatibility: CompatibilityLevel, can_run_min: bool, 
                         can_run_rec: bool, bottlenecks: List[ComponentType], is_nvidia: bool) -> str:
        """Generate compatibility summary."""
        if not is_nvidia:
            return "G-Assist requires an NVIDIA RTX or GTX GPU. Non-NVIDIA GPUs are not supported."
        
        if compatibility == CompatibilityLevel.EXCELLENT:
            return "Your RTX/GTX system exceeds the recommended requirements and will run this game excellently."
        elif compatibility == CompatibilityLevel.GOOD:
            return "Your RTX/GTX system meets the recommended requirements and will run this game well."
        elif compatibility == CompatibilityLevel.ADEQUATE:
            return "Your RTX/GTX system meets the minimum requirements but may struggle with higher settings."
        elif compatibility == CompatibilityLevel.POOR:
            return "Your RTX/GTX system barely meets the requirements and may have poor performance."
        else:
            return "Your RTX/GTX system does not meet the minimum requirements for this game."
    
    def _generate_recommendations(self, component_analyses: List[ComponentAnalysis], 
                                bottlenecks: List[ComponentType], is_nvidia: bool) -> List[str]:
        """Generate upgrade recommendations."""
        recommendations = []
        
        if not is_nvidia:
            recommendations.append("G-Assist requires an NVIDIA RTX or GTX GPU")
            return recommendations
        
        for analysis in component_analyses:
            if analysis.upgrade_suggestion:
                recommendations.append(analysis.upgrade_suggestion)
        
        # Add general recommendations based on bottlenecks
        if ComponentType.GPU in bottlenecks:
            recommendations.append("NVIDIA GPU is the primary bottleneck - consider RTX upgrade for DLSS benefits")
        if ComponentType.CPU in bottlenecks:
            recommendations.append("CPU upgrade would improve overall performance")
        
        # Add RTX-specific recommendations
        if is_nvidia and ComponentType.GPU not in bottlenecks:
            recommendations.append("Consider enabling DLSS if supported for better performance")
        
        return recommendations
    
    # Helper methods for parsing requirements
    def _extract_ram_amount(self, ram_text: str) -> float:
        """Extract RAM amount in GB from text."""
        if not ram_text:
            return 8.0  # Default assumption
        
        # Look for patterns like "8 GB", "16GB", etc.
        match = re.search(r'(\d+)\s*(GB|gb)', ram_text)
        if match:
            return float(match.group(1))
        
        # Look for MB values
        match = re.search(r'(\d+)\s*(MB|mb)', ram_text)
        if match:
            return float(match.group(1)) / 1024
        
        return 8.0  # Default
    
    def _extract_storage_amount(self, storage_text: str) -> float:
        """Extract storage amount in GB from text."""
        if not storage_text:
            return 50.0  # Default assumption
        
        match = re.search(r'(\d+)\s*(GB|gb)', storage_text)
        if match:
            return float(match.group(1))
        
        return 50.0  # Default
    
    def _extract_directx_version(self, dx_text: str) -> float:
        """Extract DirectX version number."""
        if not dx_text:
            return 11.0  # Default
        
        match = re.search(r'(\d+(?:\.\d+)?)', dx_text)
        if match:
            return float(match.group(1))
        
        return 11.0  # Default
    
    def _estimate_nvidia_gpu_performance(self, gpu_name: str) -> int:
        """Estimate NVIDIA GPU performance score based on name."""
        gpu_lower = gpu_name.lower()
        
        # RTX 40 series
        if 'rtx 4090' in gpu_lower:
            return 1000
        elif 'rtx 4080' in gpu_lower:
            return 850
        elif 'rtx 4070 ti' in gpu_lower:
            return 750
        elif 'rtx 4070' in gpu_lower:
            return 650
        elif 'rtx 4060' in gpu_lower:
            return 500
        
        # RTX 30 series
        elif 'rtx 3090' in gpu_lower:
            return 800
        elif 'rtx 3080' in gpu_lower:
            return 720
        elif 'rtx 3070' in gpu_lower:
            return 600
        elif 'rtx 3060' in gpu_lower:
            return 450
        
        # RTX 20 series
        elif 'rtx 2080' in gpu_lower:
            return 550
        elif 'rtx 2070' in gpu_lower:
            return 500
        elif 'rtx 2060' in gpu_lower:
            return 400
        
        # GTX 16 series
        elif 'gtx 1660' in gpu_lower:
            return 350
        elif 'gtx 1650' in gpu_lower:
            return 300
        
        # GTX 10 series
        elif 'gtx 1080' in gpu_lower:
            return 400
        elif 'gtx 1070' in gpu_lower:
            return 350
        elif 'gtx 1060' in gpu_lower:
            return 250
        
        # Older GTX cards
        elif 'gtx 970' in gpu_lower:
            return 200
        elif 'gtx 960' in gpu_lower:
            return 150
        
        # Non-NVIDIA cards get very low score
        elif 'nvidia' not in gpu_lower and 'geforce' not in gpu_lower:
            return 0
        
        return 200  # Default for unrecognized NVIDIA cards
    
    def _estimate_required_gpu_score(self, gpu_text: str) -> int:
        """Estimate required GPU performance score from requirement text."""
        gpu_lower = gpu_text.lower()
        
        # RTX series requirements
        if 'rtx 3080' in gpu_lower:
            return 720
        elif 'rtx 3070' in gpu_lower:
            return 600
        elif 'rtx 3060' in gpu_lower:
            return 450
        elif 'rtx 2060' in gpu_lower:
            return 400
        
        # GTX series requirements
        elif 'gtx 1660' in gpu_lower:
            return 350
        elif 'gtx 1060' in gpu_lower:
            return 250
        elif 'gtx 970' in gpu_lower:
            return 200
        elif 'gtx 960' in gpu_lower:
            return 150
        
        return 250  # Default minimum estimate
    
    def _estimate_cpu_performance(self, cpu_name: str, cores: int, freq: float) -> int:
        """Estimate CPU performance score with comprehensive Intel and AMD support."""
        base_score = cores * freq * 10  # Base heuristic
        
        cpu_lower = cpu_name.lower()
        
        # Intel CPU families with generation-specific multipliers
        if 'i9' in cpu_lower:
            if any(gen in cpu_lower for gen in ['13900', '12900', '11900']):  # 13th, 12th, 11th gen
                base_score *= 1.5
            elif any(gen in cpu_lower for gen in ['10900', '9900']):  # 10th, 9th gen
                base_score *= 1.4
            else:
                base_score *= 1.3  # Older i9s
        elif 'i7' in cpu_lower:
            if any(gen in cpu_lower for gen in ['13700', '12700', '11700']):  # 13th, 12th, 11th gen
                base_score *= 1.35
            elif any(gen in cpu_lower for gen in ['10700', '9700', '8700']):  # 10th, 9th, 8th gen
                base_score *= 1.25
            elif any(gen in cpu_lower for gen in ['7700', '6700', '4790']):  # 7th, 6th, 4th gen
                base_score *= 1.15
            else:
                base_score *= 1.2  # Other i7s
        elif 'i5' in cpu_lower:
            if any(gen in cpu_lower for gen in ['13600', '12600', '11600']):  # 13th, 12th, 11th gen
                base_score *= 1.25
            elif any(gen in cpu_lower for gen in ['10600', '9600', '8600']):  # 10th, 9th, 8th gen
                base_score *= 1.15
            elif any(gen in cpu_lower for gen in ['7600', '6600', '4690']):  # 7th, 6th, 4th gen
                base_score *= 1.05
            else:
                base_score *= 1.1  # Other i5s
        elif 'i3' in cpu_lower:
            if any(gen in cpu_lower for gen in ['13100', '12100', '11100']):  # 13th, 12th, 11th gen
                base_score *= 1.0
            elif any(gen in cpu_lower for gen in ['10100', '9100', '8100']):  # 10th, 9th, 8th gen
                base_score *= 0.9
            else:
                base_score *= 0.8  # Older i3s
        
        # AMD Ryzen CPU families with generation-specific multipliers
        elif 'ryzen 9' in cpu_lower:
            if any(gen in cpu_lower for gen in ['7950', '7900', '5950', '5900']):  # Ryzen 7000, 5000 series
                base_score *= 1.5
            elif any(gen in cpu_lower for gen in ['3950', '3900']):  # Ryzen 3000 series
                base_score *= 1.4
            else:
                base_score *= 1.3  # Other Ryzen 9s
        elif 'ryzen 7' in cpu_lower:
            if any(gen in cpu_lower for gen in ['7800', '7700', '5800', '5700']):  # Ryzen 7000, 5000 series
                base_score *= 1.35
            elif any(gen in cpu_lower for gen in ['3800', '3700']):  # Ryzen 3000 series
                base_score *= 1.25
            elif any(gen in cpu_lower for gen in ['2700', '1700']):  # Ryzen 2000, 1000 series
                base_score *= 1.15
            else:
                base_score *= 1.2  # Other Ryzen 7s
        elif 'ryzen 5' in cpu_lower:
            if any(gen in cpu_lower for gen in ['7600', '5600']):  # Ryzen 7000, 5000 series
                base_score *= 1.25
            elif any(gen in cpu_lower for gen in ['3600']):  # Ryzen 3000 series
                base_score *= 1.15
            elif any(gen in cpu_lower for gen in ['2600', '1600']):  # Ryzen 2000, 1000 series
                base_score *= 1.05
            else:
                base_score *= 1.1  # Other Ryzen 5s
        elif 'ryzen 3' in cpu_lower:
            if any(gen in cpu_lower for gen in ['5300', '3300']):  # Newer Ryzen 3s
                base_score *= 1.0
            else:
                base_score *= 0.9  # Older Ryzen 3s
        
        # AMD FX series (older architecture)
        elif 'fx-' in cpu_lower:
            if any(model in cpu_lower for model in ['9590', '9370', '8370', '8350']):  # Higher-end FX
                base_score *= 0.8
            else:
                base_score *= 0.7  # Lower-end FX
        
        # AMD A-series APUs
        elif 'a10' in cpu_lower or 'a8' in cpu_lower:
            base_score *= 0.6
        elif 'a6' in cpu_lower or 'a4' in cpu_lower:
            base_score *= 0.5
        
        return int(base_score)
    
    def _estimate_required_cpu_score(self, cpu_text: str) -> int:
        """Estimate required CPU performance score from requirement text with comprehensive support."""
        cpu_lower = cpu_text.lower()
        
        # Intel CPU requirements with generation-specific scores
        if 'i9' in cpu_lower:
            if any(gen in cpu_lower for gen in ['13900', '12900', '11900']):  # Latest gen i9
                return 450
            elif any(gen in cpu_lower for gen in ['10900', '9900']):  # Recent i9
                return 400
            else:
                return 350  # Older i9s
        elif 'i7' in cpu_lower:
            if any(gen in cpu_lower for gen in ['13700', '12700', '11700']):  # Latest gen i7
                return 380
            elif any(gen in cpu_lower for gen in ['10700', '9700', '8700']):  # Recent i7
                return 320
            elif any(gen in cpu_lower for gen in ['7700', '6700', '4790']):  # Older i7
                return 280
            else:
                return 300  # Default i7
        elif 'i5' in cpu_lower:
            if any(gen in cpu_lower for gen in ['13600', '12600', '11600']):  # Latest gen i5
                return 300
            elif any(gen in cpu_lower for gen in ['10600', '9600', '8600']):  # Recent i5
                return 260
            elif any(gen in cpu_lower for gen in ['7600', '6600', '4690']):  # Older i5
                return 220
            else:
                return 250  # Default i5
        elif 'i3' in cpu_lower:
            if any(gen in cpu_lower for gen in ['13100', '12100', '11100']):  # Latest gen i3
                return 200
            else:
                return 180  # Older i3s
        
        # AMD Ryzen CPU requirements with generation-specific scores
        elif 'ryzen 9' in cpu_lower:
            if any(gen in cpu_lower for gen in ['7950', '7900', '5950', '5900']):  # Latest Ryzen 9
                return 450
            elif any(gen in cpu_lower for gen in ['3950', '3900']):  # Ryzen 3000 series
                return 400
            else:
                return 350  # Other Ryzen 9s
        elif 'ryzen 7' in cpu_lower:
            if any(gen in cpu_lower for gen in ['7800', '7700', '5800', '5700']):  # Latest Ryzen 7
                return 380
            elif any(gen in cpu_lower for gen in ['3800', '3700']):  # Ryzen 3000 series
                return 320
            elif any(gen in cpu_lower for gen in ['2700', '1700']):  # Ryzen 2000, 1000 series
                return 280
            else:
                return 300  # Default Ryzen 7
        elif 'ryzen 5' in cpu_lower:
            if any(gen in cpu_lower for gen in ['7600', '5600']):  # Latest Ryzen 5
                return 300
            elif any(gen in cpu_lower for gen in ['3600']):  # Ryzen 3000 series
                return 260
            elif any(gen in cpu_lower for gen in ['2600', '1600']):  # Ryzen 2000, 1000 series
                return 220
            else:
                return 250  # Default Ryzen 5
        elif 'ryzen 3' in cpu_lower:
            if any(gen in cpu_lower for gen in ['5300', '3300']):  # Newer Ryzen 3s
                return 200
            else:
                return 180  # Older Ryzen 3s
        
        # AMD FX series requirements
        elif 'fx-' in cpu_lower:
            if any(model in cpu_lower for model in ['9590', '9370', '8370', '8350']):  # Higher-end FX
                return 180
            else:
                return 150  # Lower-end FX
        
        # AMD A-series APU requirements
        elif 'a10' in cpu_lower or 'a8' in cpu_lower:
            return 120
        elif 'a6' in cpu_lower or 'a4' in cpu_lower:
            return 100
        
        # Generic CPU requirement patterns
        elif 'quad core' in cpu_lower or '4 core' in cpu_lower:
            return 220
        elif 'dual core' in cpu_lower or '2 core' in cpu_lower:
            return 150
        elif 'core' in cpu_lower and ('2.' in cpu_lower or '3.' in cpu_lower):  # Frequency hints
            if '3.' in cpu_lower:
                return 240
            else:
                return 200
        
        return 200  # Default minimum requirement
    
    def _create_fallback_analysis(self, game_name: str) -> CompatibilityAnalysis:
        """Create fallback analysis when main analysis fails."""
        return CompatibilityAnalysis(
            game_name=game_name,
            overall_compatibility=CompatibilityLevel.ADEQUATE,
            can_run_minimum=True,
            can_run_recommended=False,
            component_analyses=[],
            bottlenecks=[],
            overall_score=0.5,
            summary="Analysis could not be completed. Basic compatibility assumed.",
            recommendations=["Unable to provide detailed recommendations"]
        )
    
    def _create_fallback_component_analysis(self, component: ComponentType) -> ComponentAnalysis:
        """Create fallback component analysis."""
        return ComponentAnalysis(
            component=component,
            meets_minimum=True,
            meets_recommended=False,
            score=0.5,
            bottleneck_factor=0.0,
            details=f"{component.value}: Analysis unavailable",
            upgrade_suggestion=None
        )


def main():
    """Test the compatibility analyzer."""
    # This would require actual hardware specs and game requirements
    # For testing purposes, we'll create mock data
    print("NVIDIA-focused compatibility analyzer module loaded successfully")


if __name__ == "__main__":
    main()