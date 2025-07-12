"""
CanRun Engine - Main orchestration module for Universal Game Compatibility Checker
Integrates all components to provide comprehensive game compatibility analysis for NVIDIA RTX/GTX systems.
"""

import logging
import asyncio
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from .hardware_detector import HardwareDetector, HardwareSpecs
from .requirements_fetcher import RequirementsFetcher, GameRequirements
from .compatibility_analyzer import CompatibilityAnalyzer, CompatibilityAnalysis
from .performance_predictor import PerformancePredictor, PerformancePrediction


@dataclass
class CanRunResult:
    """Complete CanRun analysis result."""
    game_name: str
    timestamp: str
    hardware_specs: HardwareSpecs
    game_requirements: GameRequirements
    compatibility_analysis: CompatibilityAnalysis
    performance_prediction: PerformancePrediction
    cache_used: bool
    analysis_time_ms: int


class CanRunEngine:
    """Main CanRun engine for game compatibility checking."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize CanRun engine with all components."""
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(hours=24)  # Cache for 24 hours
        
        # Initialize components
        self.hardware_detector = HardwareDetector()
        self.requirements_fetcher = RequirementsFetcher()
        self.compatibility_analyzer = CompatibilityAnalyzer()
        self.performance_predictor = PerformancePredictor()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize hardware specs (cached for session)
        self._cached_hardware_specs: Optional[HardwareSpecs] = None
        
        self.logger.info("CanRun engine initialized successfully")
    
    async def check_game_compatibility(self, game_name: str, 
                                     use_cache: bool = True) -> CanRunResult:
        """
        Main entry point for game compatibility checking.
        
        Args:
            game_name: Name of the game to check
            use_cache: Whether to use cached results
            
        Returns:
            Complete CanRun analysis result
        """
        start_time = datetime.now()
        self.logger.info(f"Starting compatibility check for: {game_name}")
        
        try:
            # Check cache first
            if use_cache:
                cached_result = self._load_from_cache(game_name)
                if cached_result:
                    self.logger.info(f"Returning cached result for {game_name}")
                    return cached_result
            
            # Step 1: Get hardware specifications
            hardware_specs = await self._get_hardware_specs()
            self.logger.info("Hardware detection completed")
            
            # Step 2: Fetch game requirements
            game_requirements = await self._fetch_game_requirements(game_name)
            self.logger.info("Game requirements fetched")
            
            # Step 3: Analyze compatibility
            compatibility_analysis = await self._analyze_compatibility(
                game_name, hardware_specs, game_requirements
            )
            self.logger.info("Compatibility analysis completed")
            
            # Step 4: Predict performance
            performance_prediction = await self._predict_performance(
                compatibility_analysis, hardware_specs.gpu_name
            )
            self.logger.info("Performance prediction completed")
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            result = CanRunResult(
                game_name=game_name,
                timestamp=datetime.now().isoformat(),
                hardware_specs=hardware_specs,
                game_requirements=game_requirements,
                compatibility_analysis=compatibility_analysis,
                performance_prediction=performance_prediction,
                cache_used=False,
                analysis_time_ms=int(analysis_time)
            )
            
            # Cache result
            if use_cache:
                self._save_to_cache(game_name, result)
            
            self.logger.info(f"Compatibility check completed for {game_name} in {analysis_time:.2f}ms")
            return result
            
        except Exception as e:
            self.logger.error(f"Compatibility check failed for {game_name}: {e}")
            return self._create_fallback_result(game_name, str(e))
    
    async def get_hardware_info(self) -> HardwareSpecs:
        """Get current hardware specifications."""
        return await self._get_hardware_specs()
    
    async def batch_check_games(self, game_names: List[str], 
                              use_cache: bool = True) -> List[CanRunResult]:
        """Check compatibility for multiple games."""
        self.logger.info(f"Starting batch compatibility check for {len(game_names)} games")
        
        results = []
        for game_name in game_names:
            try:
                result = await self.check_game_compatibility(game_name, use_cache)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch check failed for {game_name}: {e}")
                results.append(self._create_fallback_result(game_name, str(e)))
        
        return results
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
            for cache_file in cache_files:
                os.remove(os.path.join(self.cache_dir, cache_file))
            self.logger.info(f"Cleared {len(cache_files)} cache files")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f)) for f in cache_files
            )
            return {
                'total_files': len(cache_files),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            }
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {'total_files': 0, 'total_size_bytes': 0, 'total_size_mb': 0}
    
    async def _get_hardware_specs(self) -> HardwareSpecs:
        """Get hardware specifications with session caching."""
        if self._cached_hardware_specs is None:
            self._cached_hardware_specs = await self.hardware_detector.detect_hardware()
        return self._cached_hardware_specs
    
    async def _fetch_game_requirements(self, game_name: str) -> GameRequirements:
        """Fetch game requirements."""
        return await self.requirements_fetcher.fetch_requirements(game_name)
    
    async def _analyze_compatibility(self, game_name: str, 
                                   hardware_specs: HardwareSpecs,
                                   game_requirements: GameRequirements) -> CompatibilityAnalysis:
        """Analyze game compatibility."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.compatibility_analyzer.analyze_compatibility,
            game_name,
            hardware_specs,
            game_requirements
        )
    
    async def _predict_performance(self, compatibility_analysis: CompatibilityAnalysis,
                                 gpu_name: str) -> PerformancePrediction:
        """Predict game performance."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.performance_predictor.predict_performance,
            compatibility_analysis,
            gpu_name
        )
    
    def _get_cache_path(self, game_name: str) -> str:
        """Get cache file path for a game."""
        safe_name = "".join(c for c in game_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        return os.path.join(self.cache_dir, f"{safe_name}.json")
    
    def _load_from_cache(self, game_name: str) -> Optional[CanRunResult]:
        """Load cached result if available and not expired."""
        cache_path = self._get_cache_path(game_name)
        
        try:
            if not os.path.exists(cache_path):
                return None
            
            # Check if cache is expired
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - cache_time > self.cache_duration:
                os.remove(cache_path)
                return None
            
            # Load cached data
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Convert back to CanRunResult
            result = self._deserialize_result(cache_data)
            result.cache_used = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to load cache for {game_name}: {e}")
            return None
    
    def _save_to_cache(self, game_name: str, result: CanRunResult) -> None:
        """Save result to cache."""
        cache_path = self._get_cache_path(game_name)
        
        try:
            # Serialize result
            cache_data = self._serialize_result(result)
            
            # Save to file
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Cached result for {game_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to save cache for {game_name}: {e}")
    
    def _serialize_result(self, result: CanRunResult) -> Dict:
        """Serialize CanRunResult to JSON-compatible dict."""
        return {
            'game_name': result.game_name,
            'timestamp': result.timestamp,
            'hardware_specs': asdict(result.hardware_specs),
            'game_requirements': asdict(result.game_requirements),
            'compatibility_analysis': asdict(result.compatibility_analysis),
            'performance_prediction': asdict(result.performance_prediction),
            'cache_used': result.cache_used,
            'analysis_time_ms': result.analysis_time_ms
        }
    
    def _deserialize_result(self, data: Dict) -> CanRunResult:
        """Deserialize JSON dict to CanRunResult."""
        # This is a simplified deserialization - in production, you'd want
        # more robust handling of nested objects with proper type reconstruction
        return CanRunResult(
            game_name=data['game_name'],
            timestamp=data['timestamp'],
            hardware_specs=HardwareSpecs(**data['hardware_specs']),
            game_requirements=GameRequirements(**data['game_requirements']),
            compatibility_analysis=data['compatibility_analysis'],  # Simplified
            performance_prediction=data['performance_prediction'],  # Simplified
            cache_used=data['cache_used'],
            analysis_time_ms=data['analysis_time_ms']
        )
    
    def _create_fallback_result(self, game_name: str, error_msg: str) -> CanRunResult:
        """Create fallback result when analysis fails."""
        return CanRunResult(
            game_name=game_name,
            timestamp=datetime.now().isoformat(),
            hardware_specs=HardwareSpecs(
                gpu_name="Unknown",
                gpu_memory=0,
                cpu_name="Unknown",
                cpu_cores=0,
                cpu_frequency=0.0,
                ram_total=0,
                os_name="Unknown",
                os_version="Unknown",
                directx_version=0.0,
                available_storage=0,
                is_nvidia_gpu=False
            ),
            game_requirements=GameRequirements(
                game_name=game_name,
                minimum_requirements={},
                recommended_requirements={},
                source="fallback",
                last_updated="unknown"
            ),
            compatibility_analysis=None,  # Will be handled by compatibility analyzer fallback
            performance_prediction=None,  # Will be handled by performance predictor fallback
            cache_used=False,
            analysis_time_ms=0
        )


class CanRunAPI:
    """High-level API interface for CanRun engine."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize CanRun API."""
        self.engine = CanRunEngine(cache_dir)
        self.logger = logging.getLogger(__name__)
    
    async def can_run_game(self, game_name: str) -> Dict[str, any]:
        """
        Simple yes/no compatibility check.
        
        Returns:
            Dictionary with simple compatibility result
        """
        result = await self.engine.check_game_compatibility(game_name)
        
        return {
            'can_run': result.compatibility_analysis.can_run_minimum if result.compatibility_analysis else False,
            'can_run_recommended': result.compatibility_analysis.can_run_recommended if result.compatibility_analysis else False,
            'compatibility_level': result.compatibility_analysis.overall_compatibility.value if result.compatibility_analysis else "Unknown",
            'summary': result.compatibility_analysis.summary if result.compatibility_analysis else "Analysis failed",
            'is_nvidia_gpu': result.hardware_specs.is_nvidia_gpu
        }
    
    async def get_performance_estimate(self, game_name: str) -> Dict[str, any]:
        """
        Get performance estimate for a game.
        
        Returns:
            Dictionary with performance predictions
        """
        result = await self.engine.check_game_compatibility(game_name)
        
        if not result.performance_prediction:
            return {'error': 'Performance prediction not available'}
        
        best_setting = None
        if result.performance_prediction.predictions:
            best_setting = max(result.performance_prediction.predictions, key=lambda x: x.expected_fps)
        
        return {
            'game_name': game_name,
            'overall_score': result.performance_prediction.overall_performance_score,
            'best_expected_fps': best_setting.expected_fps if best_setting else 0,
            'best_settings': {
                'resolution': best_setting.resolution.value if best_setting else "Unknown",
                'quality': best_setting.quality_preset.value if best_setting else "Unknown",
                'dlss_enabled': best_setting.dlss_enabled if best_setting else False,
                'rtx_enabled': best_setting.rtx_enabled if best_setting else False
            } if best_setting else {},
            'dlss_support': result.performance_prediction.dlss_support,
            'rtx_support': result.performance_prediction.rtx_support,
            'optimization_suggestions': result.performance_prediction.optimization_suggestions
        }
    
    async def get_hardware_summary(self) -> Dict[str, any]:
        """Get hardware summary."""
        specs = await self.engine.get_hardware_info()
        
        return {
            'gpu': specs.gpu_name,
            'gpu_memory_gb': round(specs.gpu_memory / 1024, 1),
            'cpu': specs.cpu_name,
            'cpu_cores': specs.cpu_cores,
            'ram_gb': round(specs.ram_total / 1024, 1),
            'os': f"{specs.os_name} {specs.os_version}",
            'directx': specs.directx_version,
            'is_nvidia_gpu': specs.is_nvidia_gpu,
            'g_assist_compatible': specs.is_nvidia_gpu
        }


def main():
    """Test the CanRun engine."""
    async def test_engine():
        engine = CanRunEngine()
        api = CanRunAPI()
        
        # Test hardware detection
        print("Testing hardware detection...")
        hardware = await api.get_hardware_summary()
        print(f"Hardware: {hardware}")
        
        # Test simple compatibility check
        print("\nTesting compatibility check...")
        result = await api.can_run_game("Cyberpunk 2077")
        print(f"Cyberpunk 2077 compatibility: {result}")
        
        print("\nCanRun engine test completed successfully")
    
    asyncio.run(test_engine())


if __name__ == "__main__":
    main()