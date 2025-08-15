"""
CanRun ML Predictor - Standalone module for NVIDIA G-Assist Plugin
ML-powered performance prediction for RTX/GTX gaming systems.
Flattened structure with all dependencies bundled for PyInstaller compatibility.
"""

import logging
import json
import os
import re
import platform
import psutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Try to import ML model loader
try:
    from canrun_model_loader import LightweightMLPredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import GPUtil
except ImportError:
    GPUtil = None

try:
    import wmi  # Windows only
except ImportError:
    wmi = None


class PerformanceTier(Enum):
    """Performance tier classifications for S-A-B-C-D-F system."""
    S = (90, 100, "Exceptional - Ultra settings, 4K@60fps+")
    A = (80, 89, "Excellent - High settings, 1440p@60fps")
    B = (70, 79, "Good - High settings, 1080p@60fps")
    C = (60, 69, "Adequate - Medium settings, 1080p@30fps")
    D = (50, 59, "Minimum - Low settings, 720p@30fps")
    F = (0, 49, "Below Minimum - Unable to run acceptably")


@dataclass
class PerformanceAssessment:
    """Complete performance assessment result for G-Assist integration."""
    score: int
    tier: PerformanceTier
    tier_description: str
    expected_fps: int
    recommended_settings: str
    recommended_resolution: str
    bottlenecks: List[str]
    upgrade_suggestions: List[str]
    fps_variance_range: int = 0
    fps_min: int = 0
    fps_max: int = 0
    variance_description: str = ""


class CanRunMLPredictor:
    """ML-powered performance predictor for RTX/GTX gaming systems."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Component weights for gaming performance
        self.weights = {'gpu': 0.60, 'cpu': 0.25, 'ram': 0.15}
        
        # Load hardware databases from JSON files
        self.gpu_database = self._load_gpu_database()
        self.cpu_database = self._load_cpu_database()
        
        # Initialize ML predictor if available
        self.ml_predictor = None
        if ML_AVAILABLE:
            try:
                self.ml_predictor = LightweightMLPredictor()
                if self.ml_predictor.is_loaded:
                    self.logger.info("ML FPS predictor loaded successfully")
                else:
                    self.ml_predictor = None
                    self.logger.warning("ML model not loaded - using formula-based predictions")
            except Exception as e:
                self.logger.warning(f"ML predictor initialization failed: {e}")
                self.ml_predictor = None
        else:
            self.ml_predictor = None
        
        # Resolution scaling factors from research data
        self.resolution_scaling = {
            "4K": 0.40,      # 60% performance loss at 4K
            "1440p": 0.65,   # 35% performance loss at 1440p
            "1080p": 1.0,    # Baseline
            "720p": 1.4      # 40% performance gain
        }
        
        # Quality adjustment factors per GPU generation
        self.quality_factors = {
            "RTX_50": 1.2,   # Best performance with latest optimizations
            "RTX_40": 1.1,   # Excellent performance
            "RTX_30": 1.0,   # High settings (baseline)
            "RTX_20": 0.95,  # Slight penalty for older architecture
            "GTX_16": 0.9,   # Medium-High settings
            "GTX_10": 0.85,  # Medium settings
            "OLD": 0.7       # Low settings for very old GPUs
        }
        
        self.logger.info("CanRun ML predictor initialized")
    
    def _load_gpu_database(self) -> Dict[str, Any]:
        """Load GPU database from JSON file."""
        try:
            possible_paths = [
                "data/gpu_hierarchy.json",
                "canrun/data/gpu_hierarchy.json",
                "gpu_hierarchy.json"
            ]
            
            for json_path in possible_paths:
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        nvidia_data = data.get('nvidia', {})
                        self.logger.info(f"Loaded GPU database with {len(nvidia_data)} entries")
                        return nvidia_data
            
            self.logger.warning("GPU database not found")
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load GPU database: {e}")
            return {}
    
    def _load_cpu_database(self) -> Dict[str, Any]:
        """Load CPU database from JSON file."""
        try:
            possible_paths = [
                "data/cpu_benchmarks.json",
                "canrun/data/cpu_benchmarks.json",
                "cpu_benchmarks.json"
            ]
            
            for json_path in possible_paths:
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        cpu_lookup = {}
                        if 'cpu_patterns' in data:
                            for group, patterns in data['cpu_patterns'].items():
                                cpu_lookup.update(patterns)
                        self.logger.info(f"Loaded CPU database with {len(cpu_lookup)} patterns")
                        return cpu_lookup
            
            self.logger.warning("CPU database not found")
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load CPU database: {e}")
            return {}
    
    def assess_performance(self, hardware_specs: Dict[str, Any], game_requirements: Dict[str, Any] = None) -> PerformanceAssessment:
        """
        Main performance assessment using ML prediction when available.
        
        Args:
            hardware_specs: Hardware specifications from detector
            game_requirements: Game requirements from fetcher
            
        Returns:
            PerformanceAssessment with tier, score, FPS, and recommendations
        """
        self.logger.info("Starting ML performance assessment")
        
        # Get hardware data in normalized format
        hardware = self._normalize_hardware_data(hardware_specs)
        
        # Store current context for variance calculation
        self._current_gpu_model = hardware['gpu']['model']
        self._current_game_name = game_requirements.get('game_name', 'General Gaming') if game_requirements else 'General Gaming'
        
        # Get component scores using database lookup
        gpu_score, gpu_passmark = self._get_gpu_score(hardware['gpu'])
        cpu_score, cpu_passmark = self._get_cpu_score(hardware['cpu'])
        ram_score = self._get_ram_score(hardware['ram'], game_requirements)
        
        # Calculate weighted performance score
        total_score = int(
            gpu_score * self.weights['gpu'] +
            cpu_score * self.weights['cpu'] +
            ram_score * self.weights['ram']
        )
        
        # Get performance tier
        tier = self._score_to_tier(total_score)
        
        # Calculate FPS using ML model or fallback
        expected_fps = self._calculate_fps_with_ml(
            gpu_passmark=gpu_passmark,
            cpu_passmark=cpu_passmark,
            resolution=hardware['display']['resolution_class'],
            gpu_model=hardware['gpu']['model'],
            game_requirements=game_requirements
        )
        
        # Generate recommendations
        settings, resolution = self._get_recommendations(tier)
        bottlenecks = self._identify_bottlenecks({'gpu': gpu_score, 'cpu': cpu_score, 'ram': ram_score})
        upgrades = self._get_upgrade_suggestions(tier, {'gpu': gpu_score, 'cpu': cpu_score, 'ram': ram_score})
        
        # Calculate FPS variance using actual resolution scaling factors
        variance_range, variance_min, variance_max, variance_desc = self._calculate_fps_variance(
            expected_fps, hardware['display']['resolution_class'], gpu_passmark
        )
        
        assessment = PerformanceAssessment(
            score=total_score,
            tier=tier,
            tier_description=tier.value[2],
            expected_fps=expected_fps,
            recommended_settings=settings,
            recommended_resolution=resolution,
            bottlenecks=bottlenecks,
            upgrade_suggestions=upgrades,
            fps_variance_range=variance_range,
            fps_min=variance_min,
            fps_max=variance_max,
            variance_description=variance_desc
        )
        
        self.logger.info(f"Assessment complete: {tier.name} tier, {expected_fps} FPS")
        return assessment
    
    def _normalize_hardware_data(self, hardware_specs: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize hardware data from various input formats."""
        # Handle both flat and nested formats
        if 'gpu_model' in hardware_specs:
            # Flat format from hardware detector
            gpu_model = hardware_specs.get('gpu_model', '')
            gpu_vram = hardware_specs.get('gpu_vram_gb', 8)
            cpu_model = hardware_specs.get('cpu_model', '')
            cpu_cores = hardware_specs.get('cpu_cores', 4)
            ram_total = hardware_specs.get('ram_total_gb', 16)
            width = hardware_specs.get('display_width', 1920)
            height = hardware_specs.get('display_height', 1080)
        else:
            # Nested format
            gpu_model = hardware_specs.get('gpu', {}).get('model', '')
            gpu_vram = hardware_specs.get('gpu', {}).get('vram', 8)
            cpu_model = hardware_specs.get('cpu', {}).get('model', '')
            cpu_cores = hardware_specs.get('cpu', {}).get('cores', 4)
            ram_total = hardware_specs.get('ram', {}).get('total', 16)
            width = hardware_specs.get('display', {}).get('width', 1920)
            height = hardware_specs.get('display', {}).get('height', 1080)
        
        return {
            'gpu': {
                'model': gpu_model,
                'vram': gpu_vram
            },
            'cpu': {
                'model': cpu_model,
                'cores': cpu_cores,
                'frequency': 3000  # Default
            },
            'ram': {
                'total': ram_total,
                'available': ram_total * 0.7  # Estimate
            },
            'display': {
                'resolution_class': self._classify_resolution(width, height)
            }
        }
    
    def _get_gpu_score(self, gpu_info: Dict[str, Any]) -> Tuple[int, int]:
        """Get GPU score and PassMark from pre-loaded database (loaded once in constructor)."""
        gpu_model = gpu_info.get('model', '').lower()
        
        # Search in pre-loaded database (self.gpu_database loaded once in __init__)
        for gpu_key, gpu_data in self.gpu_database.items():
            if gpu_key.lower() in gpu_model or gpu_model in gpu_key.lower():
                passmark = gpu_data.get('passmark_g3d', gpu_data.get('score', 500) * 45)
                score = min(100, passmark / 400)
                self.logger.debug(f"GPU '{gpu_model}' found: PassMark {passmark}, Score {score}")
                return int(score), int(passmark)
        
        # If not found in database, log and use minimal fallback
        self.logger.warning(f"GPU data not found for: {gpu_model}")
        return 30, 4000  # Minimal fallback
    
    def _get_cpu_score(self, cpu_info: Dict[str, Any]) -> Tuple[int, int]:
        """Get CPU score and PassMark from pre-loaded database (loaded once in constructor)."""
        cpu_model = cpu_info.get('model', '')
        
        # Handle special case for default/unknown CPUs
        if cpu_model in ['Default CPU', 'Unknown CPU', ''] or not cpu_model:
            cores = cpu_info.get('cores', 8)  # Use reasonable default
            freq = cpu_info.get('frequency', 3500)  # Modern CPU frequency
            passmark = cores * freq * 2.5
            score = min(100, passmark / 300)
            self.logger.debug(f"Using estimated values for unknown CPU: PassMark {passmark}, Score {score}")
            return int(score), int(passmark)
        
        # Search in pre-loaded database using regex patterns (self.cpu_database loaded once in __init__)
        for pattern, passmark in self.cpu_database.items():
            if re.search(pattern, cpu_model, re.IGNORECASE):
                score = min(100, passmark / 300)
                self.logger.debug(f"CPU '{cpu_model}' matched: PassMark {passmark}, Score {score}")
                return int(score), int(passmark)
        
        # If not found, use basic estimation
        cores = cpu_info.get('cores', 4)
        freq = cpu_info.get('frequency', 3000)
        passmark = cores * freq * 2.5
        score = min(100, passmark / 300)
        self.logger.debug(f"CPU data not found for: {cpu_model}, using estimation")
        return int(score), int(passmark)
    
    def _get_ram_score(self, ram_info: Dict[str, Any], game_requirements: Dict[str, Any] = None) -> int:
        """Get RAM score based on available memory."""
        available = ram_info.get('available', ram_info.get('total', 8) * 0.7)
        required = 8  # Default requirement
        
        if game_requirements:
            required = game_requirements.get('recommended_ram_gb', 
                      game_requirements.get('minimum_ram_gb', 8))
        
        score = min(100, (available / required) * 100)
        return int(score)
    
    def _calculate_fps_with_ml(self, gpu_passmark: int, cpu_passmark: int, resolution: str,
                               gpu_model: str, game_requirements: Dict[str, Any] = None) -> int:
        """Calculate FPS using ML model when available, otherwise use formula."""
        if self.ml_predictor and self.ml_predictor.is_loaded:
            try:
                game_name = "General Gaming"
                if game_requirements:
                    game_name = game_requirements.get('game_name', 'General Gaming')
                
                ml_fps = self.ml_predictor.predict_fps(
                    gpu_name=gpu_model,
                    passmark_score=gpu_passmark,
                    resolution=resolution,
                    game_name=game_name
                )
                
                if ml_fps is not None and ml_fps > 0:
                    self.logger.debug(f"ML prediction: {gpu_model} @ {resolution} = {ml_fps} FPS")
                    return int(ml_fps)
            except Exception as e:
                self.logger.error(f"ML prediction error: {e}")
        
        # Fallback to formula-based calculation
        return self._calculate_fps_formula(gpu_passmark, cpu_passmark, resolution, gpu_model, game_requirements)
    
    def _calculate_fps_formula(self, gpu_passmark: int, cpu_passmark: int, resolution: str,
                               gpu_model: str, game_requirements: Dict[str, Any] = None) -> int:
        """Formula-based FPS calculation when ML is not available."""
        
        # Game-specific base calculations
        if game_requirements:
            game_name = game_requirements.get('game_name', '').lower()
            
            if any(game in game_name for game in ['valorant', 'counter-strike', 'cs2']):
                base_fps = int(gpu_passmark / 60)  # Esports games
            elif any(game in game_name for game in ['fortnite', 'apex']):
                base_fps = int(gpu_passmark / 75)  # Battle royale
            else:
                base_fps = int(gpu_passmark / 200)  # Demanding games
        else:
            base_fps = int(gpu_passmark / 160)  # Default
        
        # Apply resolution scaling
        resolution_factor = self.resolution_scaling.get(resolution, 1.0)
        
        # GPU generation quality factor
        gpu_gen = self._get_gpu_generation(gpu_model)
        quality_factor = self.quality_factors.get(gpu_gen, 1.0)
        
        # CPU bottleneck factor
        cpu_factor = 1.0
        if cpu_passmark < 8000:
            cpu_factor = 0.8
        elif cpu_passmark < 15000:
            cpu_factor = 0.9
        
        # Calculate final FPS
        predicted_fps = int(base_fps * resolution_factor * quality_factor * cpu_factor)
        
        # Apply reasonable bounds
        bounds = {"720p": 400, "1080p": 300, "1440p": 200, "4K": 120}
        max_fps = bounds.get(resolution, 200)
        final_fps = max(15, min(predicted_fps, max_fps))
        
        self.logger.debug(f"Formula FPS: {final_fps} for {gpu_model} @ {resolution}")
        return final_fps
    
    def _get_gpu_generation(self, gpu_model: str) -> str:
        """Classify GPU generation for quality settings."""
        gpu_lower = gpu_model.lower()
        
        if 'rtx 50' in gpu_lower or any(model in gpu_lower for model in ['5090', '5080', '5070']):
            return "RTX_50"
        elif 'rtx 40' in gpu_lower or any(model in gpu_lower for model in ['4090', '4080', '4070']):
            return "RTX_40"
        elif 'rtx 30' in gpu_lower:
            return "RTX_30"
        elif 'rtx 20' in gpu_lower:
            return "RTX_20"
        elif 'gtx 16' in gpu_lower:
            return "GTX_16"
        elif 'gtx 10' in gpu_lower:
            return "GTX_10"
        else:
            return "OLD"
    
    def _classify_resolution(self, width: int, height: int) -> str:
        """Classify resolution for performance impact."""
        total_pixels = width * height
        
        if total_pixels >= 3840 * 2160 * 0.9:  # True 4K
            return "4K"
        elif total_pixels >= 2560 * 1440 * 0.9:  # 1440p
            return "1440p"
        elif total_pixels >= 1920 * 1080 * 0.9:  # 1080p
            return "1080p"
        else:
            return "720p"
    
    def _score_to_tier(self, score: int) -> PerformanceTier:
        """Convert score to performance tier."""
        for tier in PerformanceTier:
            min_score, max_score, _ = tier.value
            if min_score <= score <= max_score:
                return tier
        return PerformanceTier.F
    
    def _get_recommendations(self, tier: PerformanceTier) -> Tuple[str, str]:
        """Get settings and resolution recommendations."""
        recommendations = {
            PerformanceTier.S: ("Ultra/Maximum", "4K (3840x2160)"),
            PerformanceTier.A: ("High", "1440p (2560x1440)"),
            PerformanceTier.B: ("High", "1080p (1920x1080)"),
            PerformanceTier.C: ("Medium", "1080p (1920x1080)"),
            PerformanceTier.D: ("Low", "720p (1280x720)"),
            PerformanceTier.F: ("Minimum", "720p (1280x720)")
        }
        return recommendations.get(tier, ("Low", "720p"))
    
    def _identify_bottlenecks(self, scores: Dict[str, int]) -> List[str]:
        """Identify system bottlenecks."""
        bottlenecks = []
        min_score = min(scores.values())
        avg_score = sum(scores.values()) / len(scores)
        
        for component, score in scores.items():
            if score <= min_score + 5 and score < avg_score - 10:
                bottlenecks.append(component.upper())
        
        return bottlenecks
    
    def _get_upgrade_suggestions(self, tier: PerformanceTier, scores: Dict[str, int]) -> List[str]:
        """Get upgrade suggestions based on tier and scores."""
        suggestions = []
        
        if tier in [PerformanceTier.S, PerformanceTier.A]:
            suggestions.append("System performing excellently")
            if scores['gpu'] < 90:
                suggestions.append("Consider DLSS/RTX features for enhanced visuals")
        elif tier == PerformanceTier.B:
            if scores['gpu'] < 75:
                suggestions.append("GPU upgrade recommended - Consider RTX 4060 Ti or RTX 4070")
            if scores['cpu'] < 75:
                suggestions.append("CPU upgrade could improve performance")
        elif tier in [PerformanceTier.C, PerformanceTier.D]:
            suggestions.append("GPU upgrade needed - Consider RTX 4060 or newer")
            if scores['ram'] < 65:
                suggestions.append("Upgrade to 16GB+ RAM recommended")
        else:  # F tier
            suggestions.append("Major system upgrade required")
            suggestions.append("Consider complete system rebuild for modern gaming")
        
        return suggestions
    
    def _calculate_fps_variance(self, base_fps: int, current_resolution: str, gpu_passmark: int) -> Tuple[int, int, int, str]:
        """
        Calculate FPS variance using ML model predictions across different resolutions.
        Uses actual ML predictions when available for more accurate variance calculation.
        
        Args:
            base_fps: FPS at current resolution
            current_resolution: Current resolution tier (720p, 1080p, 1440p, 4K)
            gpu_passmark: GPU PassMark score for fallback calculations
            
        Returns:
            Tuple of (variance_range, min_fps, max_fps, description)
        """
        try:
            fps_predictions = {}
            
            # Try to use ML model for variance calculation if available
            if self.ml_predictor and self.ml_predictor.is_loaded:
                try:
                    # Get GPU model from the current assessment context (stored during assessment)
                    gpu_model = getattr(self, '_current_gpu_model', 'RTX 4060')
                    game_name = getattr(self, '_current_game_name', 'General Gaming')
                    
                    # Get ML predictions for all resolutions
                    for resolution in ["720p", "1080p", "1440p", "4K"]:
                        ml_fps = self.ml_predictor.predict_fps(
                            gpu_name=gpu_model,
                            passmark_score=gpu_passmark,
                            resolution=resolution,
                            game_name=game_name
                        )
                        if ml_fps and ml_fps > 0:
                            fps_predictions[resolution] = int(ml_fps)
                    
                    # If we got ML predictions for multiple resolutions, use them
                    if len(fps_predictions) >= 3:
                        fps_values = list(fps_predictions.values())
                        min_fps = min(fps_values)
                        max_fps = max(fps_values)
                        variance_range = max_fps - min_fps
                        
                        # Create descriptive message with ML confidence
                        fps_info = ", ".join([f"{res}: {fps}" for res, fps in fps_predictions.items()])
                        description = f"ML-Predicted FPS Variance: {variance_range} ({fps_info})"
                        
                        self.logger.debug(f"ML-based variance calculated: {description}")
                        return variance_range, min_fps, max_fps, description
                        
                except Exception as e:
                    self.logger.warning(f"ML variance calculation failed: {e}")
            
            # Fallback to resolution scaling factors if ML not available
            current_scaling = self.resolution_scaling.get(current_resolution, 1.0)
            normalized_1080p_fps = int(base_fps / current_scaling)
            
            # Calculate FPS at all resolutions using scaling factors
            fps_720p = int(normalized_1080p_fps * self.resolution_scaling["720p"])
            fps_1080p = normalized_1080p_fps
            fps_1440p = int(normalized_1080p_fps * self.resolution_scaling["1440p"])
            fps_4k = int(normalized_1080p_fps * self.resolution_scaling["4K"])
            
            # Calculate variance metrics
            fps_values = [fps_720p, fps_1080p, fps_1440p, fps_4k]
            min_fps = min(fps_values)
            max_fps = max(fps_values)
            variance_range = max_fps - min_fps
            
            # Create descriptive message
            description = f"Resolution Scaling Variance: {variance_range} FPS (720p: {fps_720p}, 1080p: {fps_1080p}, 1440p: {fps_1440p}, 4K: {fps_4k})"
            
            self.logger.debug(f"Scaling-based variance calculated: {description}")
            return variance_range, min_fps, max_fps, description
            
        except Exception as e:
            self.logger.error(f"FPS variance calculation failed: {e}")
            return 0, base_fps, base_fps, "FPS variance calculation unavailable"
    
    def predict_fps(self, gpu_name: str, game_name: str, resolution: str = "1080p", quality: str = "Ultra") -> int:
        """
        Simple FPS prediction interface for testing and G-Assist integration.
        
        Args:
            gpu_name: GPU model name
            game_name: Game name
            resolution: Target resolution (1080p, 1440p, 4K)
            quality: Quality settings (ignored for now)
            
        Returns:
            Predicted FPS as integer
        """
        try:
            # Create minimal hardware specs for prediction
            hardware_specs = {
                'gpu_model': gpu_name,
                'gpu_vram_gb': 8,  # Default
                'cpu_model': 'Default CPU',
                'cpu_cores': 8,
                'ram_total_gb': 16,
                'display_width': {"720p": 1280, "1080p": 1920, "1440p": 2560, "4K": 3840}.get(resolution, 1920),
                'display_height': {"720p": 720, "1080p": 1080, "1440p": 1440, "4K": 2160}.get(resolution, 1080)
            }
            
            # Create minimal game requirements
            game_requirements = {
                'game_name': game_name,
                'recommended_ram_gb': 16,
                'minimum_ram_gb': 8
            }
            
            # Get performance assessment
            assessment = self.assess_performance(hardware_specs, game_requirements)
            
            return assessment.expected_fps
            
        except Exception as e:
            self.logger.error(f"FPS prediction failed: {e}")
            return 60  # Safe fallback