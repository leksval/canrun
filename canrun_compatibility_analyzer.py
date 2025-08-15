"""
CanRun Compatibility Analyzer - Standalone module for NVIDIA G-Assist Plugin
Compatibility analysis using ML predictions and JSON data from other modules.
Flattened structure with all dependencies bundled for PyInstaller compatibility.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


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

    def get_minimum_requirements_status(self) -> Dict[str, Any]:
        """Get clear status about minimum requirements compliance."""
        failing_components = []
        meeting_components = []
        
        for analysis in self.component_analyses:
            if analysis.meets_minimum:
                meeting_components.append({
                    'component': analysis.component.value,
                    'status': 'MEETS_MINIMUM',
                    'details': analysis.details
                })
            else:
                failing_components.append({
                    'component': analysis.component.value,
                    'status': 'BELOW_MINIMUM',
                    'details': analysis.details,
                    'upgrade_suggestion': analysis.upgrade_suggestion
                })
        
        return {
            'can_run_game': self.can_run_minimum,
            'overall_status': 'MEETS_MINIMUM_REQUIREMENTS' if self.can_run_minimum else 'BELOW_MINIMUM_REQUIREMENTS',
            'meeting_components': meeting_components,
            'failing_components': failing_components,
            'summary_message': self._get_minimum_requirements_message()
        }
    
    def _get_minimum_requirements_message(self) -> str:
        """Generate clear message about minimum requirements status."""
        if self.can_run_minimum:
            if self.can_run_recommended:
                return f"CANRUN: {self.game_name} will run EXCELLENTLY - System exceeds recommended requirements!"
            else:
                return f"CANRUN: {self.game_name} will run - System meets minimum requirements!"
        else:
            failing_components = [c.component.value for c in self.component_analyses if not c.meets_minimum]
            return f"CANNOT RUN: {self.game_name} requires upgrades - Failing components: {', '.join(failing_components)}"

    def get_runnable_status(self) -> str:
        """Get simple runnable status message."""
        return self._get_minimum_requirements_message()


class CanRunCompatibilityAnalyzer:
    """
    Compatibility analyzer that uses ML predictions and JSON data from other modules.
    No hardcoded data - relies on existing ML predictor and hardware detector.
    """
    
    def __init__(self, ml_predictor=None):
        self.logger = logging.getLogger(__name__)
        self.ml_predictor = ml_predictor  # Use existing ML predictor for performance scores
        
        # Load static analysis configuration once (following user's pattern)
        self._load_static_config()
        
        self.logger.info("CanRun compatibility analyzer initialized")
    
    def _load_static_config(self):
        """Load static configuration data once (following user's pattern)."""
        
        # Component weights for gaming performance (loaded once)
        self.component_weights = {
            ComponentType.GPU: 0.45,    # Higher weight for GPU in gaming
            ComponentType.CPU: 0.30,    # Important for modern games
            ComponentType.RAM: 0.15,    # Memory requirements
            ComponentType.STORAGE: 0.05, # Less critical for analysis
            ComponentType.OS: 0.03,     # Usually compatible
            ComponentType.DIRECTX: 0.02  # DirectX support
        }
        
        # Compatibility thresholds (loaded once)
        self.compatibility_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'adequate': 0.5,
            'poor': 0.3
        }
        
        # Basic tier mappings for fallback when ML unavailable (loaded once)
        self.basic_tier_fallbacks = {
            'rtx': 70, 'gtx': 40, 'intel': 60, 'amd': 65
        }

    def analyze_compatibility(self, game_name: str, hardware_specs: Dict[str, Any],
                            game_requirements: Dict[str, Any]) -> CompatibilityAnalysis:
        """
        Perform compatibility analysis using ML predictions and JSON data.
        Leverages existing ML predictor instead of hardcoded values.
        """
        # Validate inputs
        assert game_name and game_name.strip(), "Game name must be provided"
        assert hardware_specs.get('is_nvidia_gpu', False), "RTX/GTX GPU required for G-Assist compatibility"
        assert game_requirements.get('game_name', '').strip(), "Game requirements must be valid"
        
        # Get ML performance assessment if available
        ml_assessment = None
        if self.ml_predictor:
            try:
                ml_assessment = self.ml_predictor.assess_performance(hardware_specs, game_requirements)
                self.logger.info(f"Using ML assessment for {game_name}: {ml_assessment.tier.name} tier")
            except Exception as e:
                self.logger.warning(f"ML assessment failed, using fallback analysis: {e}")
        
        # Analyze each component using ML data when available
        component_analyses = [
            self._analyze_gpu_with_ml(hardware_specs, game_requirements, ml_assessment),
            self._analyze_cpu_with_ml(hardware_specs, game_requirements, ml_assessment),
            self._analyze_ram(hardware_specs, game_requirements),
            self._analyze_storage(hardware_specs, game_requirements),
            self._analyze_os(hardware_specs, game_requirements),
            self._analyze_directx(hardware_specs, game_requirements)
        ]
        
        # Calculate overall compatibility using pre-loaded weights
        overall_score = self._calculate_overall_score(component_analyses)
        overall_compatibility = self._determine_compatibility_level(overall_score)
        
        # Determine run capabilities
        can_run_minimum = all(c.meets_minimum for c in component_analyses)
        can_run_recommended = all(c.meets_recommended for c in component_analyses)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(component_analyses)
        
        # Generate summary and recommendations
        summary = self._generate_summary(overall_compatibility, can_run_minimum, 
                                       can_run_recommended, bottlenecks, ml_assessment)
        recommendations = self._generate_recommendations(component_analyses, bottlenecks, 
                                                       hardware_specs, ml_assessment)
        
        return CompatibilityAnalysis(
            game_name=game_requirements.get('game_name', game_name),
            overall_compatibility=overall_compatibility,
            can_run_minimum=can_run_minimum,
            can_run_recommended=can_run_recommended,
            component_analyses=component_analyses,
            bottlenecks=bottlenecks,
            overall_score=overall_score,
            summary=summary,
            recommendations=recommendations
        )

    def _analyze_gpu_with_ml(self, hardware: Dict[str, Any], requirements: Dict[str, Any], 
                           ml_assessment=None) -> ComponentAnalysis:
        """Analyze GPU compatibility using ML predictions when available."""
        assert hardware.get('is_nvidia_gpu', False), "RTX/GTX GPU required"
        
        # Use ML assessment if available, otherwise basic comparison
        if ml_assessment:
            # Extract GPU-specific data from ML assessment
            gpu_score = ml_assessment.score / 100.0  # Convert to 0-1 scale
            meets_minimum = ml_assessment.tier.value[0] >= 50  # D tier or better
            meets_recommended = ml_assessment.tier.value[0] >= 70  # B tier or better
            bottleneck_factor = 1.0 - gpu_score if gpu_score < 0.7 else 0.0
        else:
            # Fallback analysis without ML
            gpu_score = 0.6  # Conservative estimate
            meets_minimum = True  # Assume NVIDIA GPU meets minimum
            meets_recommended = False  # Conservative for recommended
            bottleneck_factor = 0.3
        
        # Generate details with RTX/DLSS features
        rtx_features = []
        if hardware.get('supports_rtx', False):
            rtx_features.append("RTX Ray Tracing")
        if hardware.get('supports_dlss', False):
            rtx_features.append("DLSS")
        
        gpu_model = hardware.get('gpu_model', 'Unknown')
        gpu_vram = hardware.get('gpu_vram_gb', 8)
        details = f"NVIDIA {gpu_model} ({gpu_vram}GB VRAM"
        if rtx_features:
            details += f", {', '.join(rtx_features)}"
        details += ")"
        
        if ml_assessment:
            details += f" - {ml_assessment.tier.name} tier performance"
        elif meets_recommended:
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
            score=gpu_score,
            bottleneck_factor=bottleneck_factor,
            details=details,
            upgrade_suggestion=upgrade_suggestion
        )

    def _analyze_cpu_with_ml(self, hardware: Dict[str, Any], requirements: Dict[str, Any], 
                           ml_assessment=None) -> ComponentAnalysis:
        """Analyze CPU compatibility using ML data when available."""
        cpu_cores = hardware.get('cpu_cores', 4)
        cpu_threads = hardware.get('cpu_threads', 4)
        cpu_model = hardware.get('cpu_model', '')
        
        # Use ML assessment insights if available
        if ml_assessment:
            # Estimate CPU contribution to overall score
            cpu_score = min(1.0, (ml_assessment.score / 100.0) * 1.2)  # Slight boost for CPU
            meets_minimum = ml_assessment.tier.value[0] >= 50
            meets_recommended = ml_assessment.tier.value[0] >= 70
            bottleneck_factor = max(0.0, (50 - ml_assessment.score) / 50.0) if ml_assessment.score < 50 else 0.0
        else:
            # Basic analysis based on cores and model
            cpu_score = min(1.0, (cpu_cores * cpu_threads) / 16.0)  # Normalize to 16 threads
            meets_minimum = cpu_cores >= 4
            meets_recommended = cpu_cores >= 6 and cpu_threads >= 12
            bottleneck_factor = max(0.0, (4 - cpu_cores) / 4.0) if cpu_cores < 4 else 0.0
        
        # Generate details
        details = f"CPU: {cpu_model} ({cpu_cores}C/{cpu_threads}T)"
        
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
            score=cpu_score,
            bottleneck_factor=bottleneck_factor,
            details=details,
            upgrade_suggestion=upgrade_suggestion
        )

    def _analyze_ram(self, hardware: Dict[str, Any], requirements: Dict[str, Any]) -> ComponentAnalysis:
        """Analyze RAM compatibility."""
        ram_total = hardware.get('ram_total_gb', 16)
        min_ram = requirements.get('minimum_ram_gb', 8)
        rec_ram = requirements.get('recommended_ram_gb', 16)
        
        # Apply tolerance for RAM comparison
        min_ram_with_tolerance = min_ram * 0.95
        rec_ram_with_tolerance = rec_ram * 0.95
        
        meets_minimum = ram_total >= min_ram_with_tolerance
        meets_recommended = ram_total >= rec_ram_with_tolerance
        score = min(1.0, ram_total / max(rec_ram, 1))
        bottleneck_factor = max(0.0, (min_ram - ram_total) / max(min_ram, 1))
        
        details = f"RAM: {ram_total}GB"
        if meets_recommended:
            details += " - Sufficient for recommended settings"
        elif meets_minimum:
            details += " - Meets minimum requirements"
        else:
            details += " - Insufficient RAM"
        
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

    def _analyze_storage(self, hardware: Dict[str, Any], requirements: Dict[str, Any]) -> ComponentAnalysis:
        """Analyze storage compatibility."""
        min_storage = requirements.get('minimum_storage_gb', 50)
        rec_storage = requirements.get('recommended_storage_gb', min_storage)
        
        # Assume adequate storage for analysis purposes
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

    def _analyze_os(self, hardware: Dict[str, Any], requirements: Dict[str, Any]) -> ComponentAnalysis:
        """Analyze OS compatibility for NVIDIA gaming."""
        os_version = hardware.get('os_version', '')
        is_windows = 'windows' in os_version.lower()
        
        meets_minimum = is_windows
        meets_recommended = is_windows
        score = 1.0 if is_windows else 0.0
        bottleneck_factor = 0.0 if is_windows else 1.0
        
        details = f"OS: {os_version}"
        if is_windows:
            details += " - Compatible with G-Assist"
        else:
            details += " - May not be compatible with G-Assist"
        
        upgrade_suggestion = None
        if not is_windows:
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

    def _analyze_directx(self, hardware: Dict[str, Any], requirements: Dict[str, Any]) -> ComponentAnalysis:
        """Analyze DirectX compatibility."""
        directx_version = hardware.get('directx_version', 'DirectX 12')
        
        # Simple DirectX compatibility check
        meets_minimum = True  # Assume modern systems have adequate DirectX
        meets_recommended = True
        score = 1.0
        bottleneck_factor = 0.0
        
        details = f"DirectX: {directx_version} - Fully supported"
        
        return ComponentAnalysis(
            component=ComponentType.DIRECTX,
            meets_minimum=meets_minimum,
            meets_recommended=meets_recommended,
            score=score,
            bottleneck_factor=bottleneck_factor,
            details=details
        )

    def _calculate_overall_score(self, component_analyses: List[ComponentAnalysis]) -> float:
        """Calculate weighted overall performance score using pre-loaded weights."""
        total_score = 0.0
        total_weight = 0.0
        
        for analysis in component_analyses:
            weight = self.component_weights.get(analysis.component, 0.1)
            total_score += analysis.score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

    def _determine_compatibility_level(self, score: float) -> CompatibilityLevel:
        """Determine compatibility level using pre-loaded thresholds."""
        if score >= self.compatibility_thresholds['excellent']:
            return CompatibilityLevel.EXCELLENT
        elif score >= self.compatibility_thresholds['good']:
            return CompatibilityLevel.GOOD
        elif score >= self.compatibility_thresholds['adequate']:
            return CompatibilityLevel.ADEQUATE
        elif score >= self.compatibility_thresholds['poor']:
            return CompatibilityLevel.POOR
        else:
            return CompatibilityLevel.INCOMPATIBLE

    def _identify_bottlenecks(self, component_analyses: List[ComponentAnalysis]) -> List[ComponentType]:
        """Identify component bottlenecks."""
        bottlenecks = []
        for analysis in component_analyses:
            if analysis.bottleneck_factor > 0.3:
                bottlenecks.append(analysis.component)
        return bottlenecks

    def _generate_summary(self, compatibility: CompatibilityLevel, can_run_min: bool, 
                         can_run_rec: bool, bottlenecks: List[ComponentType], ml_assessment=None) -> str:
        """Generate NVIDIA gaming compatibility summary."""
        if ml_assessment:
            return f"ML Analysis: {ml_assessment.tier_description}. Your NVIDIA RTX/GTX system shows {compatibility.value.lower()} compatibility."
        
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
                                hardware: Dict[str, Any], ml_assessment=None) -> List[str]:
        """Generate NVIDIA gaming recommendations."""
        recommendations = []
        
        # Add ML-based recommendations if available
        if ml_assessment and hasattr(ml_assessment, 'upgrade_suggestions'):
            recommendations.extend(ml_assessment.upgrade_suggestions)
        
        # Add component-specific recommendations
        for analysis in component_analyses:
            if analysis.upgrade_suggestion:
                recommendations.append(analysis.upgrade_suggestion)
        
        # Add NVIDIA-specific recommendations
        if ComponentType.GPU in bottlenecks:
            recommendations.append("Consider upgrading to a newer NVIDIA RTX GPU for better ray tracing and DLSS performance")
        
        # Add RTX-specific features
        if hardware.get('supports_rtx', False) and ComponentType.GPU not in bottlenecks:
            recommendations.append("Enable ray tracing if supported by the game for enhanced visual quality")
        
        if hardware.get('supports_dlss', False) and ComponentType.GPU not in bottlenecks:
            recommendations.append("Enable DLSS if supported by the game for improved performance")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations

    def quick_compatibility_check(self, hardware_specs: Dict[str, Any], 
                                 game_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Quick compatibility check for G-Assist responses."""
        try:
            is_nvidia = hardware_specs.get('is_nvidia_gpu', False)
            ram_gb = hardware_specs.get('ram_total_gb', 16)
            min_ram = game_requirements.get('minimum_ram_gb', 8)
            
            # Use ML predictor for quick assessment if available
            if self.ml_predictor:
                try:
                    ml_result = self.ml_predictor.assess_performance(hardware_specs, game_requirements)
                    return {
                        'can_run': ml_result.tier.value[0] >= 50,
                        'confidence': 'high',
                        'tier': ml_result.tier.name,
                        'expected_fps': ml_result.expected_fps,
                        'source': 'ml_prediction'
                    }
                except Exception:
                    pass
            
            # Fallback basic check
            can_run = is_nvidia and ram_gb >= min_ram * 0.95
            
            return {
                'can_run': can_run,
                'confidence': 'medium',
                'tier': 'ADEQUATE' if can_run else 'POOR',
                'source': 'basic_check'
            }
            
        except Exception as e:
            self.logger.error(f"Quick compatibility check failed: {e}")
            return {
                'can_run': False,
                'confidence': 'low',
                'error': str(e),
                'source': 'error_fallback'
            }