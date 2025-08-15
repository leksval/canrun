"""
CanRun Engine - Standalone module for NVIDIA G-Assist Plugin
Privacy-focused game compatibility analysis for NVIDIA RTX/GTX systems.
Flattened structure with all dependencies bundled for PyInstaller compatibility.
"""

import logging
import asyncio
import json
import os
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

# Import local modules from canrun directory
from canrun_hardware_detector import CanRunHardwareDetector, HardwareSpecs
from canrun_game_fetcher import CanRunGameFetcher, GameRequirements
from canrun_game_matcher import CanRunGameMatcher
from canrun_compatibility_analyzer import CanRunCompatibilityAnalyzer, CompatibilityAnalysis, ComponentAnalysis, ComponentType, CompatibilityLevel
from canrun_ml_predictor import CanRunMLPredictor, PerformanceAssessment


@dataclass
class CanRunResult:
    """Complete CanRun analysis result for G-Assist integration."""
    game_name: str
    timestamp: str
    hardware_specs: HardwareSpecs
    game_requirements: GameRequirements
    compatibility_analysis: CompatibilityAnalysis
    performance_prediction: PerformanceAssessment
    detected_resolution_class: str
    cache_used: bool
    analysis_time_ms: int
    
    def get_minimum_requirements_status(self) -> Dict[str, Any]:
        """Get clear status about minimum requirements compliance."""
        return self.compatibility_analysis.get_minimum_requirements_status()
    
    def get_runnable_status_message(self) -> str:
        """Get simple runnable status message for CANRUN."""
        return self.compatibility_analysis.get_runnable_status()
    
    def can_run_game(self) -> bool:
        """Check if the game can run on minimum requirements."""
        return self.compatibility_analysis.can_run_minimum
    
    def exceeds_recommended_requirements(self) -> bool:
        """Check if system exceeds recommended requirements."""
        return self.compatibility_analysis.can_run_recommended


class CanRunEngine:
    """Main CanRun engine for NVIDIA G-Assist plugin compatibility."""
    
    def __init__(self, cache_dir: str = "cache", enable_llm: bool = True):
        """Initialize CanRun engine with all components."""
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(minutes=15)
        
        # Initialize components
        self.hardware_detector = CanRunHardwareDetector()
        self.game_fetcher = CanRunGameFetcher()
        self.game_matcher = CanRunGameMatcher()
        self.compatibility_analyzer = CanRunCompatibilityAnalyzer()
        self.ml_predictor = CanRunMLPredictor()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Session hardware cache
        self._hardware_cache: Optional[HardwareSpecs] = None
        
        self.logger.info("CanRun engine initialized for G-Assist plugin")
    
    async def check_game_compatibility(self, game_name: str, use_cache: bool = True) -> CanRunResult:
        """
        Main entry point for game compatibility checking.
        
        Args:
            game_name: Name of the game to check
            use_cache: Whether to use cached results
            
        Returns:
            Complete CanRun analysis result
        """
        assert game_name and isinstance(game_name, str), "Game name must be a non-empty string"
        assert isinstance(use_cache, bool), "Cache flag must be boolean"
        
        game_name = game_name.strip()
        start_time = datetime.now()
        
        self.logger.info(f"Starting G-Assist compatibility check for: {game_name}")

        # Check cache first
        if use_cache:
            normalized_name = self.game_matcher.normalize_game_name(game_name)
            cache_file = os.path.join(self.cache_dir, f"{normalized_name}.json")
            cached_result = self._load_cache_file(cache_file)
            if cached_result:
                self.logger.info(f"Returning cached result for '{game_name}'")
                return cached_result

        # Fetch game requirements from Steam API
        game_requirements = await self._fetch_game_requirements(game_name)
        if game_requirements is None:
            raise ValueError(f"Game requirements not found for '{game_name}'")

        # Get hardware specifications
        hardware_specs = await self._get_hardware_specs()
        
        # Analyze compatibility
        compatibility_analysis = await self._analyze_compatibility(
            game_name, hardware_specs, game_requirements
        )
        
        # Predict performance using ML model
        display_resolution = self._get_display_resolution(hardware_specs)
        detected_resolution_class = self._classify_resolution(display_resolution)
        
        hardware_dict = {
            "gpu_model": hardware_specs.gpu_model,
            "gpu_vram_gb": hardware_specs.gpu_vram_gb,
            "cpu_model": hardware_specs.cpu_model,
            "ram_total_gb": hardware_specs.ram_total_gb,
            "supports_rtx": hardware_specs.supports_rtx,
            "supports_dlss": hardware_specs.supports_dlss,
            "display_width": display_resolution.get('width', 1920),
            "display_height": display_resolution.get('height', 1080)
        }
        
        game_requirements_dict = {
            "game_name": game_requirements.game_name,
            "minimum_gpu": game_requirements.minimum_gpu,
            "recommended_gpu": game_requirements.recommended_gpu,
            "minimum_cpu": game_requirements.minimum_cpu,
            "recommended_cpu": game_requirements.recommended_cpu,
            "minimum_ram_gb": game_requirements.minimum_ram_gb,
            "recommended_ram_gb": game_requirements.recommended_ram_gb
        }
        
        performance_prediction = await asyncio.get_event_loop().run_in_executor(
            None, self.ml_predictor.assess_performance, hardware_dict, game_requirements_dict
        )
        
        # Calculate analysis time
        analysis_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Create result
        result = CanRunResult(
            game_name=game_name,
            timestamp=datetime.now().isoformat(),
            hardware_specs=hardware_specs,
            game_requirements=game_requirements,
            compatibility_analysis=compatibility_analysis,
            performance_prediction=performance_prediction,
            detected_resolution_class=detected_resolution_class,
            cache_used=False,
            analysis_time_ms=analysis_time
        )
        
        # Cache result
        if use_cache:
            self._save_cached_result(game_name, result)
        
        self.logger.info(f"G-Assist analysis completed for {game_name} in {analysis_time}ms")
        return result
    
    async def get_hardware_info(self) -> HardwareSpecs:
        """Get current hardware specifications."""
        return await self._get_hardware_specs()
    
    async def batch_check_games(self, game_names: List[str], use_cache: bool = True) -> List[CanRunResult]:
        """Check compatibility for multiple games."""
        assert isinstance(game_names, list), "Game names must be a list"
        assert all(isinstance(name, str) for name in game_names), "All game names must be strings"
        
        results = []
        for game_name in game_names:
            try:
                result = await self.check_game_compatibility(game_name, use_cache)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch check failed for {game_name}: {e}")
                results.append(self._create_error_result(game_name, str(e)))
        
        return results
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
        for cache_file in cache_files:
            os.remove(os.path.join(self.cache_dir, cache_file))
        self.logger.info(f"Cleared {len(cache_files)} cache files")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
        total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in cache_files)
        
        return {
            'total_files': len(cache_files),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }
    
    async def _get_hardware_specs(self) -> HardwareSpecs:
        """Get hardware specifications with session caching."""
        if self._hardware_cache is None:
            self._hardware_cache = await self.hardware_detector.get_hardware_specs()
        return self._hardware_cache
    
    async def _fetch_game_requirements(self, game_name: str) -> GameRequirements:
        """Fetch game requirements from Steam API."""
        requirements = await self.game_fetcher.fetch_requirements(game_name)
        if requirements is None:
            raise ValueError(f"Requirements not found for {game_name}")
        return requirements
    
    async def _analyze_compatibility(self, game_name: str,
                                   hardware_specs: HardwareSpecs,
                                   game_requirements: GameRequirements) -> CompatibilityAnalysis:
        """Analyze hardware compatibility with game requirements."""
        # Convert dataclasses to dictionaries for compatibility analyzer
        hardware_dict = {
            "gpu_model": hardware_specs.gpu_model,
            "gpu_vram_gb": hardware_specs.gpu_vram_gb,
            "gpu_vendor": hardware_specs.gpu_vendor,
            "cpu_model": hardware_specs.cpu_model,
            "cpu_cores": hardware_specs.cpu_cores,
            "cpu_threads": hardware_specs.cpu_threads,
            "ram_total_gb": hardware_specs.ram_total_gb,
            "ram_speed_mhz": hardware_specs.ram_speed_mhz,
            "storage_type": hardware_specs.storage_type,
            "os_version": hardware_specs.os_version,
            "directx_version": hardware_specs.directx_version,
            "is_nvidia_gpu": hardware_specs.is_nvidia_gpu,
            "supports_rtx": hardware_specs.supports_rtx,
            "supports_dlss": hardware_specs.supports_dlss,
            "nvidia_driver_version": hardware_specs.nvidia_driver_version
        }
        
        game_requirements_dict = {
            "game_name": game_requirements.game_name,
            "minimum_gpu": game_requirements.minimum_gpu,
            "recommended_gpu": game_requirements.recommended_gpu,
            "minimum_cpu": game_requirements.minimum_cpu,
            "recommended_cpu": game_requirements.recommended_cpu,
            "minimum_ram_gb": game_requirements.minimum_ram_gb,
            "recommended_ram_gb": game_requirements.recommended_ram_gb,
            "minimum_vram_gb": game_requirements.minimum_vram_gb,
            "recommended_vram_gb": game_requirements.recommended_vram_gb,
            "minimum_storage_gb": game_requirements.minimum_storage_gb,
            "recommended_storage_gb": game_requirements.recommended_storage_gb,
            "minimum_directx": game_requirements.minimum_directx,
            "recommended_directx": game_requirements.recommended_directx,
            "minimum_os": game_requirements.minimum_os,
            "recommended_os": game_requirements.recommended_os
        }
        
        analysis = await asyncio.get_event_loop().run_in_executor(
            None, self.compatibility_analyzer.analyze_compatibility,
            game_name, hardware_dict, game_requirements_dict
        )
        return analysis
    
    def _get_display_resolution(self, hardware_specs: HardwareSpecs) -> Dict[str, int]:
        """Get display resolution from hardware specs or detect."""
        if hasattr(hardware_specs, 'display_resolution'):
            return hardware_specs.display_resolution
        
        # Fallback to default
        return {'width': 1920, 'height': 1080}
    
    def _classify_resolution(self, display_resolution: Dict[str, int]) -> str:
        """Classify resolution for performance prediction."""
        width = display_resolution.get('width', 1920)
        height = display_resolution.get('height', 1080)
        total_pixels = width * height
        
        if total_pixels >= 3840 * 2160 * 0.9:
            return "4K"
        elif total_pixels >= 2560 * 1440 * 0.9:
            return "1440p"
        elif total_pixels >= 1920 * 1080 * 0.9:
            return "1080p"
        else:
            return "720p"
    
    def _load_cache_file(self, cache_file: str) -> Optional[CanRunResult]:
        """Load and validate a single cache file."""
        if not os.path.isfile(cache_file):
            return None
        
        try:
            mtime = os.path.getmtime(cache_file)
            if (datetime.now().timestamp() - mtime) > self.cache_duration.total_seconds():
                return None

            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            return self._reconstruct_canrun_result(data)
        except Exception as e:
            self.logger.warning(f"Failed to load cache file {cache_file}: {e}")
            return None
    
    def _reconstruct_canrun_result(self, data: Dict[str, Any]) -> CanRunResult:
        """Reconstruct CanRunResult from dictionary data."""
        # Reconstruct nested dataclasses
        hardware_specs = HardwareSpecs(**data['hardware_specs'])
        game_requirements = GameRequirements(**data['game_requirements'])
        
        # Properly reconstruct CompatibilityAnalysis with ComponentAnalysis objects
        compat_data = data['compatibility_analysis']
        
        # Convert component_analyses from dicts to ComponentAnalysis objects
        component_analyses = []
        if 'component_analyses' in compat_data:
            for comp_dict in compat_data['component_analyses']:
                # Convert component enum from string back to ComponentType
                if isinstance(comp_dict.get('component'), str):
                    comp_dict['component'] = ComponentType(comp_dict['component'])
                component_analyses.append(ComponentAnalysis(**comp_dict))
        
        # Convert bottlenecks from strings back to ComponentType enums
        bottlenecks = []
        if 'bottlenecks' in compat_data:
            for bottleneck in compat_data['bottlenecks']:
                if isinstance(bottleneck, str):
                    bottlenecks.append(ComponentType(bottleneck))
                else:
                    bottlenecks.append(bottleneck)
        
        # Convert overall_compatibility from string back to CompatibilityLevel enum
        overall_compatibility = compat_data['overall_compatibility']
        if isinstance(overall_compatibility, str):
            overall_compatibility = CompatibilityLevel(overall_compatibility)
        
        compatibility_analysis = CompatibilityAnalysis(
            game_name=compat_data['game_name'],
            overall_compatibility=overall_compatibility,
            can_run_minimum=compat_data['can_run_minimum'],
            can_run_recommended=compat_data['can_run_recommended'],
            component_analyses=component_analyses,
            bottlenecks=bottlenecks,
            overall_score=compat_data['overall_score'],
            summary=compat_data['summary'],
            recommendations=compat_data['recommendations']
        )
        
        performance_prediction = PerformanceAssessment(**data['performance_prediction'])
        
        return CanRunResult(
            game_name=data['game_name'],
            timestamp=data['timestamp'],
            hardware_specs=hardware_specs,
            game_requirements=game_requirements,
            compatibility_analysis=compatibility_analysis,
            performance_prediction=performance_prediction,
            detected_resolution_class=data.get('detected_resolution_class', '1080p'),
            cache_used=data.get('cache_used', True),
            analysis_time_ms=data.get('analysis_time_ms', 0)
        )
    
    def _save_cached_result(self, game_name: str, result: CanRunResult) -> None:
        """Save analysis result to cache."""
        normalized_name = self.game_matcher.normalize_game_name(game_name)
        cache_file = os.path.join(self.cache_dir, f"{normalized_name}.json")
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        try:
            result_dict = asdict(result)
            result_dict['game_name'] = normalized_name
            
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, default=str)
                
            self.logger.debug(f"Cached result for '{game_name}' as '{normalized_name}'")
        except Exception as e:
            self.logger.warning(f"Failed to save cache for {game_name}: {e}")
    
    def _create_error_result(self, game_name: str, error_message: str) -> CanRunResult:
        """Create an error result for failed analysis."""
        # Create minimal error components
        error_hardware = HardwareSpecs(
            gpu_model="Unknown",
            gpu_vram_gb=0,
            cpu_model="Unknown", 
            cpu_cores=0,
            cpu_threads=0,
            ram_total_gb=0,
            ram_speed_mhz=0,
            storage_type="Unknown",
            primary_monitor_refresh_hz=0,
            primary_monitor_resolution="Unknown",
            os_version="Unknown",
            directx_version="Unknown"
        )
        
        error_requirements = GameRequirements(
            game_name=game_name,
            minimum_cpu="Unknown",
            minimum_gpu="Unknown", 
            minimum_ram_gb=0,
            minimum_vram_gb=0,
            minimum_storage_gb=0
        )
        
        error_compatibility = CompatibilityAnalysis(
            game_name=game_name,
            overall_compatibility="INCOMPATIBLE",
            can_run_minimum=False,
            can_run_recommended=False,
            component_analyses=[],
            bottlenecks=[],
            overall_score=0,
            summary=f"Error: {error_message}",
            recommendations=["Please retry the analysis"]
        )
        
        error_performance = PerformanceAssessment(
            score=0,
            tier="F",
            tier_description="Error occurred during analysis",
            expected_fps=0,
            recommended_settings="Unable to determine",
            recommended_resolution="Unknown",
            bottlenecks=[],
            upgrade_suggestions=["Please retry the analysis"]
        )
        
        return CanRunResult(
            game_name=game_name,
            timestamp=datetime.now().isoformat(),
            hardware_specs=error_hardware,
            game_requirements=error_requirements,
            compatibility_analysis=error_compatibility,
            performance_prediction=error_performance,
            detected_resolution_class="Unknown",
            cache_used=False,
            analysis_time_ms=0
        )
    
    def check_compatibility(self, game_name: str) -> bool:
        """
        Simple synchronous compatibility check for testing and basic integration.
        Uses quick assessment without async complexity.
        
        Args:
            game_name: Name of the game to check
            
        Returns:
            True if game can run on minimum requirements, False otherwise
        """
        try:
            # Quick compatibility assessment using existing components
            hardware_dict = {
                'is_nvidia_gpu': True,  # Assume NVIDIA for G-Assist
                'ram_total_gb': 16,     # Reasonable default
                'gpu_model': 'RTX 4070',  # Reasonable default
                'cpu_cores': 8
            }
            
            game_requirements = {
                'game_name': game_name,
                'minimum_ram_gb': 8,    # Conservative estimate
                'minimum_gpu': 'GTX 1060'
            }
            
            # Use quick compatibility check from analyzer
            result = self.compatibility_analyzer.quick_compatibility_check(
                hardware_dict, game_requirements
            )
            
            return result.get('can_run', False)
            
        except Exception as e:
            self.logger.debug(f"Quick compatibility check failed for {game_name}: {e}")
            return True  # Default to optimistic for testing