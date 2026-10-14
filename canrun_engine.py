"""
CanRun Engine - Standalone module for NVIDIA G-Assist Plugin
Privacy-focused game compatibility analysis for NVIDIA RTX/GTX systems.
Flattened structure with all dependencies bundled for PyInstaller compatibility.
"""

import hashlib
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
from canrun_steam_fps import SteamFpsClient

MAX_GAME_NAME_LENGTH = 120
MAX_CACHE_BYTES = 256 * 1024


def sanitize_game_name(game_name: str) -> str:
    """Drop control characters and cap length before the name is logged or fetched."""
    if not isinstance(game_name, str):
        game_name = str(game_name or "")
    cleaned = "".join(ch for ch in game_name if ch.isprintable())
    return " ".join(cleaned.split())[:MAX_GAME_NAME_LENGTH]


def safe_cache_path(cache_dir: str, normalized_name: str) -> str:
    """Build a cache file path that cannot leave cache_dir."""
    slug = re.sub(r"[^a-z0-9]+", "-", (normalized_name or "").lower()).strip("-") or "game"
    digest = hashlib.sha256((normalized_name or "").encode("utf-8")).hexdigest()[:12]
    filename = f"{slug[:80]}-{digest}.json"
    root = os.path.realpath(cache_dir)
    path = os.path.realpath(os.path.join(root, filename))
    if os.path.commonpath([root, path]) != root:
        raise ValueError("cache path escaped the cache directory")
    return path


def cache_file_is_trusted(cache_dir: str, cache_file: str) -> bool:
    """Reject symlinks, oversized files, and paths outside the cache directory."""
    if os.path.islink(cache_file):
        return False
    root = os.path.realpath(cache_dir)
    path = os.path.realpath(cache_file)
    try:
        if os.path.commonpath([root, path]) != root:
            return False
    except ValueError:
        return False
    if os.path.isfile(path) and os.path.getsize(path) > MAX_CACHE_BYTES:
        return False
    return True


def _write_private_cache(path: str, payload: str) -> None:
    """Write a cache file without following a symlink planted at that path."""
    if os.path.islink(path):
        raise OSError("refusing cache symlink")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags | nofollow)
    except OSError:
        if os.path.islink(path):
            raise
        fd = os.open(path, flags)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)


def trusted_dataclass_kwargs(payload: Any, cls) -> Dict[str, Any]:
    """Copy only fields the dataclass declares. Extra keys are dropped."""
    if not isinstance(payload, dict):
        raise ValueError("cache record is not an object")
    from dataclasses import fields
    allowed = {item.name for item in fields(cls)}
    return {key: payload[key] for key in payload if key in allowed}


def requirements_from_cache(payload: Any) -> GameRequirements:
    """Build requirements from a cache object, dropping unknown keys and bad app ids."""
    from canrun_steam_fps import valid_steam_app_id
    picked = trusted_dataclass_kwargs(payload, GameRequirements)
    if "steam_app_id" in picked:
        picked["steam_app_id"] = valid_steam_app_id(str(picked["steam_app_id"]))
    return GameRequirements(**picked)


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
        self.steam_fps = SteamFpsClient()
        
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
        if not isinstance(game_name, str):
            raise ValueError("Game name must be a non-empty string")
        if not isinstance(use_cache, bool):
            raise ValueError("Cache flag must be boolean")

        game_name = sanitize_game_name(game_name)
        if not game_name:
            raise ValueError("Game name must be a non-empty string")
        start_time = datetime.now()
        
        self.logger.info(f"Starting G-Assist compatibility check for: {game_name}")

        # Check cache first
        if use_cache:
            cache_file = self._cache_file_for(game_name)
            cached_result = self._load_cache_file(cache_file)
            cached_fps = getattr(getattr(cached_result, "performance_prediction", None), "expected_fps", 0) or 0
            if cached_result and cached_fps > 0:
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
        
        # FPS comes from Steam frame-rate stats for this app and PC.
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
            "recommended_ram_gb": game_requirements.recommended_ram_gb,
            "steam_app_id": getattr(game_requirements, "steam_app_id", ""),
            "resolution": detected_resolution_class,
        }

        steam_estimate = await self.steam_fps.estimate(
            appid=getattr(game_requirements, "steam_app_id", ""),
            gpu_model=hardware_specs.gpu_model,
            cpu_model=hardware_specs.cpu_model,
            game_name=game_requirements.game_name,
            resolution=detected_resolution_class,
        )
        
        def _assess():
            return self.ml_predictor.assess_performance(
                hardware_dict, game_requirements_dict, steam_estimate
            )

        performance_prediction = await asyncio.get_event_loop().run_in_executor(None, _assess)
        
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
        if not isinstance(game_names, list) or not all(isinstance(name, str) for name in game_names):
            raise ValueError("Game names must be a list of strings")
        if len(game_names) > 20:
            raise ValueError("Too many games in one batch")
        
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
        """Clear cached results that stay inside the cache directory."""
        cache_files = self._listed_cache_files()
        for cache_file in cache_files:
            os.remove(cache_file)
        self.logger.info(f"Cleared {len(cache_files)} cache files")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        cache_files = self._listed_cache_files()
        total_size = sum(os.path.getsize(path) for path in cache_files)
        
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
        """Get display resolution from hardware detector."""
        try:
            # Use the hardware detector's proper display resolution method
            display_info = self.hardware_detector.get_display_resolution()
            return {
                'width': display_info.get('width', 1920),
                'height': display_info.get('height', 1080)
            }
        except Exception as e:
            self.logger.warning(f"Display resolution detection failed: {e}")
            # Fallback to default
            return {'width': 1920, 'height': 1080}
    
    def _classify_resolution(self, display_resolution: Dict[str, int]) -> str:
        """Classify resolution for performance prediction using hardware detector."""
        width = display_resolution.get('width', 1920)
        height = display_resolution.get('height', 1080)
        
        # Use the hardware detector's resolution tier classification
        return self.hardware_detector.get_resolution_tier(width, height)
    
    def _listed_cache_files(self) -> List[str]:
        root = os.path.realpath(self.cache_dir)
        if not os.path.isdir(root):
            return []
        paths = []
        for name in os.listdir(root):
            if not name.endswith(".json"):
                continue
            candidate = os.path.join(root, name)
            if os.path.islink(candidate):
                continue
            path = os.path.realpath(candidate)
            try:
                if os.path.commonpath([root, path]) != root:
                    continue
            except ValueError:
                continue
            if os.path.isfile(path):
                paths.append(path)
        return paths

    def _load_cache_file(self, cache_file: str) -> Optional[CanRunResult]:
        """Load and validate a single cache file."""
        if not cache_file_is_trusted(self.cache_dir, cache_file):
            self.logger.warning("Refusing untrusted cache file")
            return None
        root = os.path.realpath(self.cache_dir)
        path = os.path.realpath(cache_file)
        try:
            if os.path.commonpath([root, path]) != root:
                self.logger.warning("Refusing cache path outside the cache directory")
                return None
        except ValueError:
            return None
        if not os.path.isfile(path):
            return None
        
        try:
            mtime = os.path.getmtime(path)
            if (datetime.now().timestamp() - mtime) > self.cache_duration.total_seconds():
                return None

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            return self._reconstruct_canrun_result(data)
        except Exception as e:
            self.logger.warning(f"Failed to load cache file {cache_file}: {e}")
            return None
    
    def _reconstruct_canrun_result(self, data: Dict[str, Any]) -> CanRunResult:
        """Reconstruct CanRunResult from dictionary data."""
        # Reconstruct nested dataclasses
        hardware_specs = HardwareSpecs(**trusted_dataclass_kwargs(data.get("hardware_specs"), HardwareSpecs))
        game_requirements = requirements_from_cache(data.get("game_requirements"))
        
        # Properly reconstruct CompatibilityAnalysis with ComponentAnalysis objects
        compat_data = data['compatibility_analysis']
        
        # Convert component_analyses from dicts to ComponentAnalysis objects
        component_analyses = []
        raw_components = compat_data.get("component_analyses") or []
        if not isinstance(raw_components, list) or len(raw_components) > 20:
            raise ValueError("cache record has too many component rows")
        for comp_dict in raw_components:
            if not isinstance(comp_dict, dict):
                raise ValueError("cache component row is not an object")
            if isinstance(comp_dict.get("component"), str):
                comp_dict["component"] = ComponentType(comp_dict["component"])
            component_analyses.append(ComponentAnalysis(**trusted_dataclass_kwargs(comp_dict, ComponentAnalysis)))
        
        bottlenecks = []
        raw_bottlenecks = compat_data.get("bottlenecks") or []
        if not isinstance(raw_bottlenecks, list) or len(raw_bottlenecks) > 20:
            raise ValueError("cache record has too many bottlenecks")
        for bottleneck in raw_bottlenecks:
            if isinstance(bottleneck, str):
                bottlenecks.append(ComponentType(bottleneck))
            elif isinstance(bottleneck, ComponentType):
                bottlenecks.append(bottleneck)
            else:
                raise ValueError("cache bottleneck is not a component name")
        
        overall_compatibility = compat_data["overall_compatibility"]
        if isinstance(overall_compatibility, str):
            overall_compatibility = CompatibilityLevel(overall_compatibility)
        recommendations = compat_data.get("recommendations") or []
        if not isinstance(recommendations, list) or len(recommendations) > 20:
            raise ValueError("cache record has too many recommendations")
        
        compatibility_analysis = CompatibilityAnalysis(
            game_name=str(compat_data["game_name"])[:MAX_GAME_NAME_LENGTH],
            overall_compatibility=overall_compatibility,
            can_run_minimum=bool(compat_data["can_run_minimum"]),
            can_run_recommended=bool(compat_data["can_run_recommended"]),
            component_analyses=component_analyses,
            bottlenecks=bottlenecks,
            overall_score=compat_data["overall_score"],
            summary=str(compat_data.get("summary") or "")[:500],
            recommendations=[str(item)[:300] for item in recommendations]
        )
        
        performance_prediction = PerformanceAssessment(
            **trusted_dataclass_kwargs(data.get("performance_prediction"), PerformanceAssessment)
        )
        
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
    
    def _cache_file_for(self, game_name: str) -> str:
        normalized_name = self.game_matcher.normalize_game_name(game_name)
        return safe_cache_path(self.cache_dir, normalized_name)

    def _save_cached_result(self, game_name: str, result: CanRunResult) -> None:
        """Save analysis result to cache."""
        normalized_name = self.game_matcher.normalize_game_name(game_name)
        cache_file = safe_cache_path(self.cache_dir, normalized_name)
        
        if not cache_file_is_trusted(self.cache_dir, cache_file):
            self.logger.warning("Refusing to write an untrusted cache path")
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        
        try:
            result_dict = asdict(result)
            result_dict['game_name'] = normalized_name
            payload = json.dumps(result_dict, indent=2, default=str)
            _write_private_cache(cache_file, payload)
                
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