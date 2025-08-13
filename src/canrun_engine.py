r"""
CanRun Engine - Main orchestration module for Universal Game Compatibility Checker
Privacy-focused game compatibility analysis for NVIDIA RTX/GTX systems.
"""

import logging
import asyncio
import json
import os
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from canrun.src.privacy_aware_hardware_detector import PrivacyAwareHardwareDetector, PrivacyAwareHardwareSpecs
from canrun.src.game_requirements_fetcher import GameRequirementsFetcher, GameRequirements
from canrun.src.optimized_game_fuzzy_matcher import OptimizedGameFuzzyMatcher
from canrun.src.compatibility_analyzer import CompatibilityAnalyzer, CompatibilityAnalysis, ComponentAnalysis, ComponentType, CompatibilityLevel
from canrun.src.dynamic_performance_predictor import DynamicPerformancePredictor, PerformanceAssessment, PerformanceTier
from canrun.src.rtx_llm_analyzer import GAssistLLMAnalyzer, LLMAnalysisResult


@dataclass
class CanRunResult:
    """Complete CanRun analysis result."""
    game_name: str
    timestamp: str
    hardware_specs: PrivacyAwareHardwareSpecs
    game_requirements: GameRequirements
    compatibility_analysis: CompatibilityAnalysis
    performance_prediction: PerformanceAssessment
    llm_analysis: Optional[Dict[str, LLMAnalysisResult]]
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
    """Main CanRun engine for privacy-aware game compatibility checking."""
    
    def __init__(self, cache_dir: str = "cache", enable_llm: bool = True):
        """Initialize CanRun engine with all components."""
        assert isinstance(cache_dir, str), "Cache directory must be a string"
        assert isinstance(enable_llm, bool), "LLM enable flag must be boolean"
        
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(minutes=15)
        self.enable_llm = enable_llm
        
        # Initialize G-Assist LLM analyzer if enabled
        self.llm_analyzer = None
        if enable_llm:
            try:
                self.llm_analyzer = GAssistLLMAnalyzer()
                self.logger.info("G-Assist LLM analyzer initialized")
            except Exception as e:
                self.logger.warning(f"LLM analyzer initialization failed: {e}")
        
        # Initialize components with LLM analyzer
        self.hardware_detector = PrivacyAwareHardwareDetector()
        self.requirements_fetcher = GameRequirementsFetcher(self.llm_analyzer)
        self.fuzzy_matcher = OptimizedGameFuzzyMatcher()
        self.compatibility_analyzer = CompatibilityAnalyzer()
        self.performance_predictor = DynamicPerformancePredictor()
        
        # Create cache directory and validate
        os.makedirs(cache_dir, exist_ok=True)
        assert os.path.isdir(cache_dir), f"Cache directory creation failed: {cache_dir}"
        
        # Session hardware cache
        self._hardware_cache: Optional[PrivacyAwareHardwareSpecs] = None
        
        self.logger.info("CanRun engine initialized successfully")
    
    async def check_game_compatibility(self, game_name: str, use_cache: bool = True) -> CanRunResult:
        """
        Main entry point for game compatibility checking.
        
        Args:
            game_name: Name of the game to check
            use_cache: Whether to use cached results
            
        Returns:
            Complete CanRun analysis result
        """
        # Validate inputs
        assert game_name and isinstance(game_name, str), "Game name must be a non-empty string"
        assert isinstance(use_cache, bool), "Cache flag must be boolean"
        
        game_name = game_name.strip()
        assert len(game_name) > 0, "Game name cannot be empty after strip"
        
        start_time = datetime.now()
        self.logger.info(f"Starting compatibility check for: {game_name}")

        # Step 1: Skip local cache matching - go directly to Steam API
        self.logger.info(f"Bypassing local cache matching, using Steam API directly for: {game_name}")
        
        # Step 2: Check Cache with Original Name (only for exact matches)
        if use_cache:
            normalized_name = self.fuzzy_matcher.normalize_game_name(game_name)
            cache_file = os.path.join(self.cache_dir, f"{normalized_name}.json")
            cached_result = self._load_cache_file(cache_file)
            if cached_result:
                self.logger.info(f"Returning cached result for '{game_name}'")
                return cached_result

        # Step 3: Fetch Requirements Directly from Steam API
        game_requirements = await self._fetch_game_requirements(game_name)
        if game_requirements is None:
            raise ValueError(f"Game requirements not found for '{corrected_game_name}'.")

        # Step 4: Get Hardware Specifications
        hardware_specs = await self._get_hardware_specs()
        assert hardware_specs is not None, "Hardware detection failed"
        
        # Step 3: Analyze compatibility
        compatibility_analysis = await self._analyze_compatibility(
            game_name, hardware_specs, game_requirements
        )
        assert compatibility_analysis is not None, "Compatibility analysis failed"
        
        # Step 4: Predict performance using S-A-B-C-D-F tier system
        hardware_dict = {
            "gpu_model": hardware_specs.gpu_model,
            "gpu_vram_gb": hardware_specs.gpu_vram_gb,
            "cpu_model": hardware_specs.cpu_model,
            "ram_total_gb": hardware_specs.ram_total_gb,
            "supports_rtx": hardware_specs.supports_rtx,
            "supports_dlss": hardware_specs.supports_dlss
        }
        
        game_requirements_dict = {
            "minimum_gpu": game_requirements.minimum_gpu,
            "recommended_gpu": game_requirements.recommended_gpu,
            "minimum_cpu": game_requirements.minimum_cpu,
            "recommended_cpu": game_requirements.recommended_cpu,
            "minimum_ram_gb": game_requirements.minimum_ram_gb,
            "recommended_ram_gb": game_requirements.recommended_ram_gb
        }
        
        performance_prediction = await asyncio.get_event_loop().run_in_executor(
            None, self.performance_predictor.assess_performance, hardware_dict, game_requirements_dict
        )
        assert performance_prediction is not None, "Performance assessment failed"
        
        # Step 5: Perform LLM analysis if enabled
        llm_analysis = None
        if self.llm_analyzer:
            llm_analysis = await self._perform_llm_analysis(
                compatibility_analysis, performance_prediction, hardware_specs
            )
        
        # Calculate analysis time
        analysis_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Create result
        result = CanRunResult(
            game_name=game_name,  # Use original game name, not corrected
            timestamp=datetime.now().isoformat(),
            hardware_specs=hardware_specs,
            game_requirements=game_requirements,
            compatibility_analysis=compatibility_analysis,
            performance_prediction=performance_prediction,
            llm_analysis=llm_analysis,
            cache_used=False,
            analysis_time_ms=analysis_time
        )
        
        # Cache result
        if use_cache:
            self._save_cached_result(game_name, result)  # Use original game name
        
        self.logger.info(f"Analysis completed for {game_name} in {analysis_time}ms")
        return result
    
    async def get_hardware_info(self) -> PrivacyAwareHardwareSpecs:
        """Get current hardware specifications."""
        return await self._get_hardware_specs()
    
    async def batch_check_games(self, game_names: List[str], use_cache: bool = True) -> List[CanRunResult]:
        """Check compatibility for multiple games."""
        assert isinstance(game_names, list), "Game names must be a list"
        assert all(isinstance(name, str) for name in game_names), "All game names must be strings"
        assert len(game_names) > 0, "Game names list cannot be empty"
        
        self.logger.info(f"Starting batch check for {len(game_names)} games")
        
        results = []
        for game_name in game_names:
            try:
                result = await self.check_game_compatibility(game_name, use_cache)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch check failed for {game_name}: {e}")
                results.append(self._create_error_result(game_name, str(e)))
        
        self.logger.info(f"Batch check completed for {len(game_names)} games")
        return results
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        assert os.path.isdir(self.cache_dir), "Cache directory does not exist"
        
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
        for cache_file in cache_files:
            os.remove(os.path.join(self.cache_dir, cache_file))
        
        self.logger.info(f"Cleared {len(cache_files)} cache files")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        assert os.path.isdir(self.cache_dir), "Cache directory does not exist"
        
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
        total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in cache_files)
        
        return {
            'total_files': len(cache_files),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }
    
    async def _get_hardware_specs(self) -> PrivacyAwareHardwareSpecs:
        """Get hardware specifications with session caching."""
        if self._hardware_cache is None:
            # Since get_hardware_specs is now async, we await it directly
            self._hardware_cache = await self.hardware_detector.get_hardware_specs()
            assert self._hardware_cache is not None, "Hardware detection returned None"
        
        return self._hardware_cache
    
    async def _fetch_game_requirements(self, game_name: str) -> GameRequirements:
        """Fetch game requirements from available sources."""
        assert game_name and isinstance(game_name, str), "Game name must be a non-empty string"
        
        requirements = await self.requirements_fetcher.fetch_requirements(game_name)
        assert requirements is not None, f"Requirements not found for {game_name}"
        
        return requirements
    
    async def _analyze_compatibility(self, game_name: str, 
                                   hardware_specs: PrivacyAwareHardwareSpecs,
                                   game_requirements: GameRequirements) -> CompatibilityAnalysis:
        """Analyze hardware compatibility with game requirements."""
        assert all([game_name, hardware_specs, game_requirements]), "All parameters are required"
        
        analysis = await asyncio.get_event_loop().run_in_executor(
            None, self.compatibility_analyzer.analyze_compatibility,
            game_name, hardware_specs, game_requirements
        )
        assert analysis is not None, "Compatibility analysis returned None"
        
        return analysis


    async def _predict_advanced_performance(self, hardware_specs: Dict, game_requirements: Dict = None) -> Dict:
        """
        Predict game performance using the advanced tiered assessment system.
        
        Args:
            hardware_specs: Hardware specifications from the detector
            game_requirements: Optional game requirements
            
        Returns:
            Dict containing advanced performance assessment with tier information
        """
        loop = asyncio.get_event_loop()
        assessment = await loop.run_in_executor(
            None,
            self.performance_predictor.predict_advanced_performance,
            hardware_specs,
            game_requirements
        )
        
        # Convert assessment to dict for compatibility
        return {
            'tier': assessment.tier.name,
            'tier_description': assessment.tier_description,
            'score': assessment.score,
            'expected_fps': assessment.expected_fps,
            'recommended_settings': assessment.recommended_settings,
            'recommended_resolution': assessment.recommended_resolution,
            'bottlenecks': assessment.bottlenecks,
            'upgrade_suggestions': assessment.upgrade_suggestions
        }

    def _get_cached_result(self, game_name: str) -> Optional[CanRunResult]:
        """DEPRECATED: This method is no longer the primary way to get cached results.
        It is kept for potential direct cache inspection but should not be used in the main workflow.
        The main workflow now fetches requirements first, then checks the cache with the corrected name.
        """
        normalized_name = self.fuzzy_matcher.normalize_game_name(game_name)
        cache_file = os.path.join(self.cache_dir, f"{normalized_name}.json")
        return self._load_cache_file(cache_file)
    
    def _load_cache_file(self, cache_file: str) -> Optional[CanRunResult]:
        """Load and validate a single cache file."""
        
        if not os.path.isfile(cache_file):
            return None
            
        try:
            mtime = os.path.getmtime(cache_file)
            if (datetime.now().timestamp() - mtime) > self.cache_duration.total_seconds():
                # Cache expired
                return None

            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Convert dictionary data back to proper dataclass objects
            return self._reconstruct_canrun_result(data)
        except Exception as e:
            self.logger.warning(f"Failed to load cache file {cache_file}: {e}")
            return None

    def _reconstruct_canrun_result(self, data: Dict[str, Any]) -> CanRunResult:
        """Reconstruct CanRunResult from dictionary data."""
        # Reconstruct nested dataclasses
        hardware_specs = PrivacyAwareHardwareSpecs(**data['hardware_specs'])
        game_requirements = GameRequirements(**data['game_requirements'])
        
        # Reconstruct compatibility analysis with proper ComponentAnalysis objects
        compat_data = data['compatibility_analysis'].copy()
        if 'component_analyses' in compat_data:
            component_analyses = []
            for comp_data in compat_data['component_analyses']:
                if isinstance(comp_data, dict):
                    # Handle enum serialization - extract value from string representation
                    component_value = comp_data['component']
                    if isinstance(component_value, str):
                        # Handle both "ComponentType.GPU" and "GPU" formats
                        if '.' in component_value:
                            component_value = component_value.split('.')[-1]  # Extract "GPU" from "ComponentType.GPU"
                        try:
                            component_type = ComponentType[component_value]
                        except KeyError:
                            component_type = ComponentType(component_value)
                    else:
                        component_type = component_value
                    
                    # Convert dictionary back to ComponentAnalysis
                    component_analyses.append(ComponentAnalysis(
                        component=component_type,
                        meets_minimum=comp_data['meets_minimum'],
                        meets_recommended=comp_data['meets_recommended'],
                        score=comp_data['score'],
                        bottleneck_factor=comp_data['bottleneck_factor'],
                        details=comp_data['details'],
                        upgrade_suggestion=comp_data.get('upgrade_suggestion')
                    ))
                else:
                    # Already a ComponentAnalysis object
                    component_analyses.append(comp_data)
            compat_data['component_analyses'] = component_analyses
        
        # Convert CompatibilityLevel from string if needed
        if isinstance(compat_data.get('overall_compatibility'), str):
            compat_data['overall_compatibility'] = CompatibilityLevel(compat_data['overall_compatibility'])
        
        # Convert bottlenecks from strings to ComponentType if needed
        if 'bottlenecks' in compat_data:
            bottlenecks = []
            for bottleneck in compat_data['bottlenecks']:
                if isinstance(bottleneck, str):
                    # Handle both "ComponentType.GPU" and "GPU" formats
                    if '.' in bottleneck:
                        bottleneck = bottleneck.split('.')[-1]  # Extract "GPU" from "ComponentType.GPU"
                    try:
                        bottlenecks.append(ComponentType[bottleneck])
                    except KeyError:
                        bottlenecks.append(ComponentType(bottleneck))
                else:
                    bottlenecks.append(bottleneck)
            compat_data['bottlenecks'] = bottlenecks
        
        compatibility_analysis = CompatibilityAnalysis(**compat_data)
        performance_prediction = PerformanceAssessment(**data['performance_prediction'])
        
        # Handle LLM analysis if present
        llm_analysis = None
        if data.get('llm_analysis'):
            llm_analysis = {}
            for key, value in data['llm_analysis'].items():
                llm_analysis[key] = LLMAnalysisResult(**value)
        
        return CanRunResult(
            game_name=data['game_name'],
            timestamp=data['timestamp'],
            hardware_specs=hardware_specs,
            game_requirements=game_requirements,
            compatibility_analysis=compatibility_analysis,
            performance_prediction=performance_prediction,
            llm_analysis=llm_analysis,
            cache_used=data.get('cache_used', True),
            analysis_time_ms=data.get('analysis_time_ms', 0)
        )

    def _save_cached_result(self, game_name: str, result: CanRunResult) -> None:
        """Save analysis result to cache using normalized game name."""

        # Normalize game name for consistent caching
        # This ensures "Diablo 4" and "Diablo IV" use the same cache file
        normalized_name = self.fuzzy_matcher.normalize_game_name(game_name)
        cache_file = os.path.join(self.cache_dir, f"{normalized_name}.json")
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        try:
            # Convert dataclass to dict recursively, handling nested dataclasses
            result_dict = asdict(result)
            
            # Update the result to use the normalized name for consistency
            result_dict['game_name'] = normalized_name
            
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, default=str)
                
            self.logger.debug(f"Cached result for '{game_name}' as '{normalized_name}'")
        except Exception as e:
            self.logger.warning(f"Failed to save cache for {game_name}: {e}")

    async def _perform_llm_analysis(self, compatibility_analysis: CompatibilityAnalysis,
                                   performance_prediction: PerformanceAssessment,
                                   hardware_specs: PrivacyAwareHardwareSpecs) -> Optional[Dict[str, LLMAnalysisResult]]:
        """Perform LLM analysis if G-Assist is available."""
        if not self.llm_analyzer:
            return None
        
        try:
            # Create analysis context for LLM
            context = {
                'compatibility': compatibility_analysis,
                'performance': performance_prediction,
                'hardware': hardware_specs
            }
            
            # Perform LLM analysis
            llm_result = await self.llm_analyzer.analyze_bottlenecks(context)
            
            return {'analysis': llm_result} if llm_result else None
            
        except Exception as e:
            self.logger.warning(f"LLM analysis failed: {e}")
            return None

    def _create_error_result(self, game_name: str, error_message: str) -> CanRunResult:
        """Create an error result for failed analysis."""
        from datetime import datetime
        
        # Create minimal error hardware specs
        error_hardware = PrivacyAwareHardwareSpecs(
            gpu_model="Unknown",
            gpu_vram_gb=0,
            cpu_name="Unknown", 
            cpu_cores=0,
            cpu_threads=0,
            ram_gb=0,
            is_nvidia_gpu=False,
            supports_rtx=False,
            supports_dlss=False,
            nvidia_driver_version="Unknown",
            os_name="Unknown",
            directx_version="Unknown"
        )
        
        # Create minimal error requirements
        error_requirements = GameRequirements(
            game_name=game_name,
            minimum_cpu="Unknown",
            minimum_gpu="Unknown", 
            minimum_ram_gb=0,
            minimum_vram_gb=0,
            minimum_storage_gb=0,
            recommended_cpu="Unknown",
            recommended_gpu="Unknown",
            recommended_ram_gb=0,
            recommended_vram_gb=0,
            recommended_storage_gb=0,
            supports_rtx=False,
            supports_dlss=False,
            directx_version="Unknown"
        )
        
        # Create error compatibility analysis
        error_compatibility = CompatibilityAnalysis(
            game_name=game_name,
            overall_compatibility="incompatible",
            cpu_compatibility="error",
            gpu_compatibility="error", 
            ram_compatibility="error",
            vram_compatibility="error",
            storage_compatibility="error",
            overall_score=0,
            bottlenecks=[f"Error: {error_message}"],
            recommendations=[]
        )
        
        # Create error performance assessment
        error_performance = PerformanceAssessment(
            score=0,
            tier=PerformanceTier.F,
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
            llm_analysis=None,
            cache_used=False,
            analysis_time_ms=0
        )
    
    def _parse_ram_value(self, ram_str: str) -> int:
        """Parse RAM value from string to integer GB."""
        if not ram_str or ram_str == "Unknown":
            return 0
        
        # Extract number from strings like "8 GB", "16GB", "8192 MB", etc.
        ram_str = str(ram_str).upper()
        
        # Match number followed by optional space and unit
        match = re.search(r'(\d+)\s*(GB|MB|G|M)?', ram_str)
        if match:
            value = int(match.group(1))
            unit = match.group(2) or 'GB'
            
            # Convert MB to GB
            if unit in ['MB', 'M']:
                value = max(1, value // 1024)  # Convert MB to GB, minimum 1GB
            
            return value
        
        return 0

    async def analyze_multiple_games(self, game_names: List[str], use_cache: bool = True) -> Dict[str, Optional[CanRunResult]]:
        """Analyze multiple games and convert the results to a dictionary format expected by tests.
        
        Args:
            game_names: List of game names to analyze
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary containing compatibility and performance analysis in the format expected by tests
        """
        results = {}
        
        for game_name in game_names:
            try:
                result = await self.check_game_compatibility(game_name, use_cache)
                results[game_name] = result
            except Exception as e:
                self.logger.error(f"Failed to analyze {game_name}: {e}")
                results[game_name] = None
                
        # Return the dictionary of results
        return results

    async def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        hardware_specs = await self._get_hardware_specs()
        
        return {
            'cpu': {
                'name': hardware_specs.cpu_name,
                'cores': hardware_specs.cpu_cores,
                'threads': hardware_specs.cpu_threads
            },
            'gpu': {
                'name': hardware_specs.gpu_model,
                'vram_gb': hardware_specs.gpu_vram_gb,
                'supports_rtx': hardware_specs.supports_rtx,
                'supports_dlss': hardware_specs.supports_dlss,
                'driver_version': hardware_specs.nvidia_driver_version
            },
            'memory': {
                'total': hardware_specs.ram_gb
            },
            'system': {
                'os': hardware_specs.os_name,
                'directx': hardware_specs.directx_version
            }
        }

    async def get_optimization_suggestions(self, game_name: str, settings: str, resolution: str) -> List[Dict[str, str]]:
        """Get optimization suggestions for specific game and settings."""
        try:
            # Get game requirements and hardware specs
            hardware_specs = await self._get_hardware_specs()
            game_requirements = await self._fetch_game_requirements(game_name)
            
            if not game_requirements:
                return [{'type': 'error', 'description': f'Game requirements not found for {game_name}'}]
            
            # Analyze compatibility to get recommendations
            compatibility_analysis = await self._analyze_compatibility(
                game_name, hardware_specs, game_requirements
            )
            
            # Convert recommendations to optimization format
            optimizations = []
            for rec in compatibility_analysis.recommendations:
                optimizations.append({
                    'type': 'settings',
                    'description': rec
                })
            
            # Add resolution-specific optimizations
            if resolution == '4K':
                optimizations.append({
                    'type': 'resolution',
                    'description': 'Consider using DLSS Quality mode for better 4K performance'
                })
            elif resolution == '1440p':
                optimizations.append({
                    'type': 'resolution', 
                    'description': 'DLSS Balanced mode recommended for optimal 1440p experience'
                })
            
            # Add RTX-specific optimizations
            if hardware_specs.supports_rtx and game_requirements.supports_rtx:
                optimizations.append({
                    'type': 'rtx',
                    'description': 'Enable RTX ray tracing for enhanced visual quality'
                })
            
            return optimizations
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization suggestions: {e}")
            return [{'type': 'error', 'description': str(e)}]

    async def analyze_game_compatibility(self, game_name: str, settings: str = "Medium", resolution: str = "System Default") -> Optional[Dict[str, Any]]:
        """Legacy method for backward compatibility with tests."""
        try:
            result = await self.check_game_compatibility(game_name)
            
            if not result:
                return None
            
            # Check if result is already a dictionary (from cache) or CanRunResult object
            if isinstance(result, dict):
                # Result is already in dictionary format from cache
                return result
            
            # Use LLM to estimate missing values intelligently
            llm_estimates = {}
            if self.llm_analyzer:
                try:
                    # Get LLM estimates for component scores and performance metrics
                    llm_estimates = await self.llm_analyzer.estimate_compatibility_metrics(
                        game_name,
                        result.hardware_specs,
                        result.compatibility_analysis,
                        result.performance_prediction
                    )
                except Exception as e:
                    self.logger.warning(f"LLM estimation failed, using fallback: {e}")
            
            # Convert CanRunResult to dictionary format with LLM estimates
            return {
                'compatibility': {
                    'compatibility_level': result.compatibility_analysis.overall_compatibility,
                    'overall_score': result.compatibility_analysis.overall_score,
                    'bottlenecks': result.compatibility_analysis.bottlenecks,
                    'component_analysis': {
                        'cpu': {
                            'status': next(('Excellent' if comp.meets_recommended else 'Good' if comp.meets_minimum else 'Poor'
                            for comp in result.compatibility_analysis.component_analyses
                            if comp.component.name.lower() == 'cpu'), 'Unknown'),
                            'score': llm_estimates.get('cpu_score', next((int(comp.score * 100)
                            for comp in result.compatibility_analysis.component_analyses
                            if comp.component.name.lower() == 'cpu'), 75))
                        },
                        'gpu': {
                            'status': next(('Excellent' if comp.meets_recommended else 'Good' if comp.meets_minimum else 'Poor'
                            for comp in result.compatibility_analysis.component_analyses
                            if comp.component.name.lower() == 'gpu'), 'Unknown'),
                            'score': llm_estimates.get('gpu_score', 80)
                        },
                        'memory': {
                            'status': next(('Excellent' if comp.meets_recommended else 'Good' if comp.meets_minimum else 'Poor'
                            for comp in result.compatibility_analysis.component_analyses
                            if comp.component.name.lower() == 'ram'), 'Unknown'),
                            'score': llm_estimates.get('memory_score', 85)
                        },
                        'storage': {
                            'status': next(('Excellent' if comp.meets_recommended else 'Good' if comp.meets_minimum else 'Poor'
                            for comp in result.compatibility_analysis.component_analyses
                            if comp.component.name.lower() == 'storage'), 'Unknown'),
                            'score': llm_estimates.get('storage_score', 90)
                        }
                    }
                },
                'performance': {
                    'fps': result.performance_prediction.expected_fps if hasattr(result.performance_prediction, 'expected_fps') else 0,
                    'performance_level': result.performance_prediction.tier.value if hasattr(result.performance_prediction, 'tier') else 'Unknown',
                    'stability': llm_estimates.get('stability', 'stable'),
                    'optimization_suggestions': result.performance_prediction.upgrade_suggestions if hasattr(result.performance_prediction, 'upgrade_suggestions') else []
                },
                'optimization_suggestions': result.performance_prediction.upgrade_suggestions if hasattr(result.performance_prediction, 'upgrade_suggestions') else [],
                'hardware_analysis': {
                    'gpu_tier': llm_estimates.get('gpu_tier', 'high-end'),
                    'bottleneck_analysis': result.compatibility_analysis.bottlenecks
                }
            }
            
        except Exception as e:
            self.logger.error(f"Legacy compatibility analysis failed: {e}")
            return None

    def _parse_ram_value(self, ram_str: str) -> int:
        """Parse RAM value from string to integer GB with proper unit handling."""
        if not ram_str or ram_str == "Unknown":
            return 0
        
        # Convert to uppercase for consistency
        ram_str = str(ram_str).upper()
        
        # Check if explicitly specified as MB
        if 'MB' in ram_str:
            # Extract number
            mb_match = re.search(r'(\d+\.?\d*)\s*MB', ram_str)
            if mb_match:
                # Convert MB to GB (rounded up to 0.5 GB minimum for values under 512MB)
                mb_value = float(mb_match.group(1))
                if mb_value < 512:
                    return 0.5  # Minimum 0.5GB for small values
                else:
                    return max(1, int(mb_value / 1024))  # Convert MB to GB, minimum 1GB
        
        # Default GB matching - more flexible pattern to match various formats
        gb_match = re.search(r'(\d+\.?\d*)\s*G?B?', ram_str)
        if gb_match:
            return int(float(gb_match.group(1)))
        
        # Last resort fallback - just try to extract any number
        number_match = re.search(r'(\d+\.?\d*)', ram_str)
        if number_match:
            return int(float(number_match.group(1)))
            
        return 0
