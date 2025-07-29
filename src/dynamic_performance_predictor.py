"""
Dynamic Performance Prediction Module for CanRun
Advanced tiered performance predictions (S-A-B-C-D-F) with dynamic hardware detection and real-time benchmarking.
Focus on NVIDIA RTX/GTX systems with comprehensive laptop support.
"""

import logging
import re
import requests
import platform
import psutil
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    import GPUtil
except ImportError:
    GPUtil = None

try:
    import wmi  # Windows only
except ImportError:
    wmi = None


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


class HardwareDetector:
    """Dynamic hardware detection with cross-platform support"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.system_info = {}
        
    def detect_all(self) -> Dict:
        """Detect all hardware components"""
        self.system_info = {
            'cpu': self._detect_cpu(),
            'gpu': self._detect_gpu(),
            'ram': self._detect_ram(),
            'os': self._detect_os(),
            'display': self._detect_display()
        }
        return self.system_info
    
    def _detect_cpu(self) -> Dict:
        """Detect CPU information"""
        cpu_info = {
            'name': platform.processor(),
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq().max if psutil.cpu_freq() else 0
        }
        
        # Windows-specific detailed CPU info
        if platform.system() == 'Windows' and wmi:
            try:
                c = wmi.WMI()
                for processor in c.Win32_Processor():
                    cpu_info['name'] = processor.Name
                    cpu_info['max_clock_speed'] = processor.MaxClockSpeed
            except Exception as e:
                self.logger.debug(f"WMI CPU detection failed: {e}")
                
        return cpu_info
    
    def _detect_gpu(self) -> Dict:
        """Detect GPU information - NVIDIA focus"""
        gpu_info = {'cards': []}
        
        # Try GPUtil first
        if GPUtil:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    # Filter for NVIDIA only
                    if 'nvidia' in gpu.name.lower() or 'geforce' in gpu.name.lower() or 'rtx' in gpu.name.lower() or 'gtx' in gpu.name.lower():
                        gpu_info['cards'].append({
                            'name': gpu.name,
                            'memory': gpu.memoryTotal,
                            'driver': gpu.driver
                        })
            except Exception as e:
                self.logger.debug(f"GPUtil detection failed: {e}")
        
        # Windows WMI fallback
        if not gpu_info['cards'] and platform.system() == 'Windows' and wmi:
            try:
                c = wmi.WMI()
                for gpu in c.Win32_VideoController():
                    if gpu.Name and gpu.AdapterRAM:
                        gpu_name = gpu.Name.lower()
                        # Filter for NVIDIA only
                        if any(keyword in gpu_name for keyword in ['nvidia', 'geforce', 'rtx', 'gtx']):
                            gpu_info['cards'].append({
                                'name': gpu.Name,
                                'memory': gpu.AdapterRAM / (1024**3) if gpu.AdapterRAM else 0,
                                'driver': gpu.DriverVersion
                            })
            except Exception as e:
                self.logger.debug(f"WMI GPU detection failed: {e}")
                
        return gpu_info
    
    def _detect_ram(self) -> Dict:
        """Detect RAM information"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024**3),  # Convert to GB
            'available': memory.available / (1024**3),
            'used_percent': memory.percent
        }
    
    def _detect_os(self) -> Dict:
        """Detect OS information"""
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'architecture': platform.machine()
        }
    
    def _detect_display(self) -> Dict:
        """Detect display resolution information"""
        display_info = {
            'primary_resolution': {'width': 1920, 'height': 1080},  # Default fallback
            'all_displays': []
        }
        
        try:
            if platform.system() == 'Windows':
                # Windows-specific resolution detection
                try:
                    import tkinter as tk
                    root = tk.Tk()
                    width = root.winfo_screenwidth()
                    height = root.winfo_screenheight()
                    root.destroy()
                    
                    display_info['primary_resolution'] = {'width': width, 'height': height}
                    display_info['all_displays'].append({'width': width, 'height': height, 'primary': True})
                    
                    self.logger.debug(f"Detected primary resolution: {width}x{height}")
                except Exception as e:
                    self.logger.debug(f"Tkinter resolution detection failed: {e}")
                
                # Try WMI for more detailed display info
                if wmi and not display_info['all_displays']:
                    try:
                        c = wmi.WMI()
                        for monitor in c.Win32_VideoController():
                            if monitor.CurrentHorizontalResolution and monitor.CurrentVerticalResolution:
                                width = monitor.CurrentHorizontalResolution
                                height = monitor.CurrentVerticalResolution
                                display_info['primary_resolution'] = {'width': width, 'height': height}
                                display_info['all_displays'].append({'width': width, 'height': height, 'primary': True})
                                self.logger.debug(f"WMI detected resolution: {width}x{height}")
                                break
                    except Exception as e:
                        self.logger.debug(f"WMI display detection failed: {e}")
            
            # Cross-platform fallback using pygame if available
            if not display_info['all_displays']:
                try:
                    import pygame
                    pygame.init()
                    info = pygame.display.Info()
                    width, height = info.current_w, info.current_h
                    pygame.quit()
                    
                    display_info['primary_resolution'] = {'width': width, 'height': height}
                    display_info['all_displays'].append({'width': width, 'height': height, 'primary': True})
                    self.logger.debug(f"Pygame detected resolution: {width}x{height}")
                except Exception as e:
                    self.logger.debug(f"Pygame display detection failed: {e}")
                    
        except Exception as e:
            self.logger.debug(f"Display detection failed: {e}")
        
        return display_info


class PerformanceCalculator:
    """Dynamic performance calculation with pattern matching for NVIDIA GPUs"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # CPU data from JSON config file
        self.cpu_benchmark_data = self._load_cpu_benchmark_data()
        # CPU benchmark scores converted from JSON data
        self.cpu_benchmarks = self._convert_cpu_data_to_benchmarks()
        # GPU data from JSON config file
        self.gpu_hierarchy = self._load_gpu_hierarchy()
        # GPU benchmark scores converted from hierarchy data
        self.gpu_benchmarks = self._convert_gpu_hierarchy_to_benchmarks()
        
    def _load_cpu_benchmark_data(self) -> Dict:
        """Load CPU benchmark data from JSON config file"""
        try:
            # Get the directory where this file is located
            current_dir = Path(__file__).parent
            # Go up one level to canrun directory, then into data
            cpu_benchmarks_path = current_dir.parent / "data" / "cpu_benchmarks.json"
            
            if cpu_benchmarks_path.exists():
                with open(cpu_benchmarks_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"CPU benchmarks file not found at {cpu_benchmarks_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load CPU benchmarks: {e}")
            return {}
    
    def _convert_cpu_data_to_benchmarks(self) -> Dict[str, int]:
        """Convert CPU benchmark data from JSON to pattern-based benchmarks"""
        benchmarks = {}
        
        if not self.cpu_benchmark_data or 'cpu_patterns' not in self.cpu_benchmark_data:
            self.logger.warning("No CPU pattern data found in JSON, using fallback")
            return self._get_fallback_cpu_benchmarks()
        
        cpu_patterns = self.cpu_benchmark_data['cpu_patterns']
        
        # Flatten all pattern groups into a single dictionary
        for group_name, patterns in cpu_patterns.items():
            for pattern, score in patterns.items():
                benchmarks[pattern] = score
        
        self.logger.info(f"Loaded {len(benchmarks)} CPU benchmark patterns from JSON data")
        return benchmarks
    
    def _get_fallback_cpu_benchmarks(self) -> Dict[str, int]:
        """Emergency fallback if JSON completely fails to load"""
        self.logger.error("Using emergency CPU fallback - JSON config failed to load")
        return {
            r'i9|Ryzen 9': 30000,
            r'i7|Ryzen 7': 20000,
            r'i5|Ryzen 5': 12000,
            r'i3|Ryzen 3': 6000,
        }
    
    def _load_gpu_hierarchy(self) -> Dict:
        """Load GPU hierarchy data from JSON config file"""
        try:
            # Get the directory where this file is located
            current_dir = Path(__file__).parent
            # Go up one level to canrun directory, then into data
            gpu_hierarchy_path = current_dir.parent / "data" / "gpu_hierarchy.json"
            
            if gpu_hierarchy_path.exists():
                with open(gpu_hierarchy_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"GPU hierarchy file not found at {gpu_hierarchy_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load GPU hierarchy: {e}")
            return {}
    
    def _convert_gpu_hierarchy_to_benchmarks(self) -> Dict[str, int]:
        """Convert GPU hierarchy data to benchmark patterns for compatibility"""
        benchmarks = {}
        
        if not self.gpu_hierarchy or 'nvidia' not in self.gpu_hierarchy:
            self.logger.warning("No NVIDIA GPU data found in hierarchy, using fallback")
            return self._get_fallback_gpu_benchmarks()
        
        nvidia_gpus = self.gpu_hierarchy['nvidia']
        
        for gpu_name, gpu_data in nvidia_gpus.items():
            # Convert score (0-1000 range) to benchmark score (multiply by appropriate factor)
            score = gpu_data.get('score', 500)
            # Convert to approximate PassMark G3D score range (multiply by ~45 for high-end cards)
            benchmark_score = int(score * 45)
            
            # Create regex pattern for the GPU name
            # Escape special regex characters and make it flexible
            pattern = gpu_name.replace(' ', r'\s*').replace('(', r'\(').replace(')', r'\)')
            benchmarks[pattern] = benchmark_score
            
            # Add common variations
            if 'RTX' in gpu_name:
                # Add laptop variants
                laptop_pattern = pattern.replace('RTX', 'RTX.*(?:Laptop|Mobile)')
                benchmarks[laptop_pattern] = int(benchmark_score * 0.8)  # Laptop GPUs ~20% slower
            elif 'GTX' in gpu_name:
                # Add laptop variants for GTX cards
                laptop_pattern = pattern.replace('GTX', 'GTX.*(?:Laptop|Mobile)')
                benchmarks[laptop_pattern] = int(benchmark_score * 0.75)  # Older laptop GPUs ~25% slower
        
        self.logger.info(f"Loaded {len(benchmarks)} GPU benchmark patterns from hierarchy data")
        return benchmarks
    
    def _get_fallback_gpu_benchmarks(self) -> Dict[str, int]:
        """Emergency fallback if JSON completely fails to load"""
        self.logger.error("Using emergency GPU fallback - JSON config failed to load")
        return {}
    
    def calculate_cpu_score(self, cpu_info: Dict, requirements: Dict) -> float:
        """Calculate CPU performance score (0-100)"""
        cpu_name = cpu_info.get('name', '')
        cpu_score = 0
        
        self.logger.debug(f"Calculating CPU score for: {cpu_name}")
        
        # Find matching benchmark using pattern matching
        for pattern, benchmark in self.cpu_benchmarks.items():
            if re.search(pattern, cpu_name, re.IGNORECASE):
                cpu_score = benchmark
                self.logger.debug(f"CPU matched pattern '{pattern}' with score {benchmark}")
                break
        
        # Fallback: estimate based on cores and frequency
        if cpu_score == 0:
            cores = cpu_info.get('cores', 1)
            freq = cpu_info.get('frequency', 2000)
            cpu_score = cores * freq * 2.5  # Rough estimation
            self.logger.debug(f"CPU fallback estimation: {cores} cores * {freq} MHz = {cpu_score}")
        
        # Normalize against requirements
        req_cpu = requirements.get('recommended', {}).get('processor', '')
        req_score = self._estimate_required_cpu_score(req_cpu)
        
        final_score = min(100, (cpu_score / req_score) * 100) if req_score > 0 else 75
        self.logger.debug(f"Final CPU score: {final_score}")
        return final_score
    
    def calculate_gpu_score(self, gpu_info: Dict, requirements: Dict) -> float:
        """Calculate NVIDIA GPU performance score (0-100)"""
        if not gpu_info.get('cards'):
            self.logger.warning("No NVIDIA GPU detected")
            return 0
            
        gpu_name = gpu_info['cards'][0].get('name', '')
        gpu_score = 0
        
        self.logger.debug(f"Calculating GPU score for: {gpu_name}")
        
        # Find matching benchmark using pattern matching
        for pattern, benchmark in self.gpu_benchmarks.items():
            if re.search(pattern, gpu_name, re.IGNORECASE):
                gpu_score = benchmark
                self.logger.debug(f"GPU matched pattern '{pattern}' with score {benchmark}")
                break
        
        # Fallback: estimate based on memory and generation
        if gpu_score == 0:
            memory = gpu_info['cards'][0].get('memory', 0)
            
            # Estimate based on VRAM and naming patterns
            if 'rtx' in gpu_name.lower():
                if '40' in gpu_name:  # RTX 40 series
                    gpu_score = memory * 3000
                elif '30' in gpu_name:  # RTX 30 series
                    gpu_score = memory * 2500
                elif '20' in gpu_name:  # RTX 20 series
                    gpu_score = memory * 2000
                else:
                    gpu_score = memory * 2200  # Generic RTX
            elif 'gtx' in gpu_name.lower():
                if '16' in gpu_name:  # GTX 16 series
                    gpu_score = memory * 1500
                elif '10' in gpu_name:  # GTX 10 series
                    gpu_score = memory * 1200
                else:
                    gpu_score = memory * 1000  # Older GTX
            else:
                gpu_score = memory * 1500  # Generic NVIDIA
                
            self.logger.debug(f"GPU fallback estimation: {memory}GB * multiplier = {gpu_score}")
        
        # Normalize against requirements
        req_gpu = requirements.get('recommended', {}).get('graphics', '')
        req_score = self._estimate_required_gpu_score(req_gpu)
        
        final_score = min(100, (gpu_score / req_score) * 100) if req_score > 0 else 75
        self.logger.debug(f"Final GPU score: {final_score}")
        return final_score
    
    def calculate_ram_score(self, ram_info: Dict, requirements: Dict) -> float:
        """Calculate RAM performance score (0-100)"""
        available_ram = ram_info.get('total', 0)
        required_ram = requirements.get('recommended', {}).get('memory', 8)
        
        if required_ram == 0:
            required_ram = requirements.get('minimum', {}).get('memory', 4)
        
        score = min(100, (available_ram / required_ram) * 100)
        self.logger.debug(f"RAM score: {available_ram}GB / {required_ram}GB = {score}")
        return score
    
    def _estimate_required_cpu_score(self, cpu_string: str) -> int:
        """Estimate required CPU score from string using JSON config data"""
        if not self.cpu_benchmark_data or 'cpu_patterns' not in self.cpu_benchmark_data:
            self.logger.error("CPU benchmark data not available from JSON config")
            return 8000
        
        cpu_patterns = self.cpu_benchmark_data['cpu_patterns']
        
        # Check generic patterns first
        if 'generic_patterns' in cpu_patterns:
            for pattern, score in cpu_patterns['generic_patterns'].items():
                if re.search(pattern, cpu_string, re.IGNORECASE):
                    return score
        
        # Check all other patterns
        for group_name, patterns in cpu_patterns.items():
            if group_name != 'generic_patterns':
                for pattern, score in patterns.items():
                    if re.search(pattern, cpu_string, re.IGNORECASE):
                        return score
        
        # Use default from JSON config
        if 'fallback_estimation' in self.cpu_benchmark_data:
            return self.cpu_benchmark_data['fallback_estimation'].get('default_middle_range', 8000)
        
        return 8000
    
    def _estimate_required_gpu_score(self, gpu_string: str) -> int:
        """Estimate required NVIDIA GPU score from string"""
        patterns = {
            r'RTX 50\d{2}': 40000,
            r'RTX 40\d{2}': 30000,
            r'RTX 30\d{2}': 20000,
            r'RTX 20\d{2}': 12000,
            r'GTX 16\d{2}': 8000,
            r'GTX 10\d{2}': 6000,
            r'GTX 9\d{2}': 4000,
        }
        
        for pattern, score in patterns.items():
            if re.search(pattern, gpu_string, re.IGNORECASE):
                return score
        
        return 8000  # Default middle-range requirement


class DynamicPerformancePredictor:
    """Dynamic performance predictor with real-time hardware detection for NVIDIA systems"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.hardware_detector = HardwareDetector()
        self.calculator = PerformanceCalculator()
        
        # Component weights as per CanRun spec
        self.weights = {
            'gpu': 0.60,
            'cpu': 0.25,
            'ram': 0.15
        }
        
        self.logger.info("Dynamic performance predictor initialized for NVIDIA RTX/GTX systems")
    
    def assess_performance(self, hardware_specs: Dict = None, game_requirements: Dict = None) -> PerformanceAssessment:
        """
        Generate advanced tiered performance assessment using dynamic detection.
        
        Args:
            hardware_specs: Optional pre-detected hardware specs
            game_requirements: Optional game requirements from Steam API
            
        Returns:
            PerformanceAssessment with tier, score, FPS, and recommendations
        """
        self.logger.info("Generating dynamic performance assessment")
        
        # Detect hardware if not provided
        if hardware_specs is None:
            hardware = self.hardware_detector.detect_all()
        else:
            # Convert from CanRun format to dynamic format
            hardware = {
                'cpu': {
                    'name': hardware_specs.get('cpu_model', ''),
                    'cores': hardware_specs.get('cpu_cores', 4),
                    'threads': hardware_specs.get('cpu_threads', 8),
                    'frequency': hardware_specs.get('cpu_frequency', 3000)
                },
                'gpu': {
                    'cards': [{
                        'name': hardware_specs.get('gpu_model', ''),
                        'memory': hardware_specs.get('gpu_vram_gb', 4),
                        'driver': 'Unknown'
                    }]
                },
                'ram': {
                    'total': hardware_specs.get('ram_total_gb', 8),
                    'available': hardware_specs.get('ram_total_gb', 8) * 0.7,
                    'used_percent': 30
                },
                'display': {
                    'primary_resolution': {'width': 1920, 'height': 1080},  # Default fallback
                    'all_displays': [{'width': 1920, 'height': 1080, 'primary': True}]
                }
            }
        
        # Calculate individual component scores
        scores = {
            'cpu': self.calculator.calculate_cpu_score(hardware['cpu'], game_requirements or {}),
            'gpu': self.calculator.calculate_gpu_score(hardware['gpu'], game_requirements or {}),
            'ram': self.calculator.calculate_ram_score(hardware['ram'], game_requirements or {})
        }
        
        self.logger.debug(f"Component scores: {scores}")
        
        # Calculate base weighted total score
        base_score = int(
            scores['gpu'] * self.weights['gpu'] +
            scores['cpu'] * self.weights['cpu'] +
            scores['ram'] * self.weights['ram']
        )
        
        # Apply adjustments based on minimum vs recommended specs comparison
        total_score = base_score
        if game_requirements:
            # Extract minimum and recommended specs
            min_gpu = game_requirements.get('minimum_gpu', '')
            rec_gpu = game_requirements.get('recommended_gpu', '')
            min_cpu = game_requirements.get('minimum_cpu', '')
            rec_cpu = game_requirements.get('recommended_cpu', '')
            min_ram = game_requirements.get('minimum_ram_gb', 8)
            rec_ram = game_requirements.get('recommended_ram_gb', 16)
            
            # Calculate how much user's hardware exceeds minimum and approaches recommended specs
            min_exceeded_factor = 0
            rec_approach_factor = 0
            
            # GPU comparison - get benchmark scores
            user_gpu_score = scores['gpu']
            min_gpu_benchmark = self._estimate_gpu_benchmark(min_gpu)
            rec_gpu_benchmark = self._estimate_gpu_benchmark(rec_gpu)
            
            if min_gpu_benchmark > 0 and rec_gpu_benchmark > 0:
                # Calculate factors based on how much user's GPU exceeds minimum and approaches recommended
                if user_gpu_score > min_gpu_benchmark:
                    min_exceeded_factor += (user_gpu_score - min_gpu_benchmark) / min_gpu_benchmark
                
                if rec_gpu_benchmark > min_gpu_benchmark:
                    rec_range = rec_gpu_benchmark - min_gpu_benchmark
                    user_in_range = user_gpu_score - min_gpu_benchmark
                    if user_in_range > 0:
                        rec_approach_factor += min(1.0, user_in_range / rec_range)
            
            # Adjust total score based on how hardware compares to game-specific requirements
            # If exceeding minimum by a lot, boost score
            if min_exceeded_factor > 1.5:
                total_score = min(100, int(total_score * 1.1))
            elif min_exceeded_factor > 0.5:
                total_score = min(100, int(total_score * 1.05))
                
            # If approaching or exceeding recommended specs, boost score further
            if rec_approach_factor > 0.8:
                total_score = min(100, int(total_score * 1.1))
            elif rec_approach_factor > 0.5:
                total_score = min(100, int(total_score * 1.05))
            
            # If barely meeting minimum, reduce score
            if min_exceeded_factor < 0.2:
                total_score = max(0, int(total_score * 0.9))
            
            # Apply more game-specific adjustments based on the actual game requirements
            game_name = None
            if game_requirements:
                game_name = (
                    game_requirements.get('game_name', '') or
                    game_requirements.get('minimum_game', '') or
                    game_requirements.get('recommended_game', '') or
                    game_requirements.get('name', '')
                )
                
                # Analyze game requirements vs hardware for dynamic scoring
                if game_name:
                    self.logger.info(f"Applying game-specific adjustments for {game_name}")
                    
                    # Get specs for calculations
                    min_gpu = game_requirements.get('minimum_gpu', '')
                    rec_gpu = game_requirements.get('recommended_gpu', '')
                    gpu_model = hardware['gpu']['cards'][0]['name'] if hardware['gpu']['cards'] else ''
                    
                    # Calculate more precise hardware vs requirements comparison
                    min_gpu_score = self._estimate_gpu_benchmark(min_gpu)
                    rec_gpu_score = self._estimate_gpu_benchmark(rec_gpu)
                    user_gpu_score = 0
                    
                    # Find benchmark of user's GPU
                    for pattern, benchmark in self.calculator.gpu_benchmarks.items():
                        if re.search(pattern, gpu_model, re.IGNORECASE):
                            user_gpu_score = benchmark
                            break
                    
                    # Log the scores for transparency
                    self.logger.info(f"Game: {game_name}, Min GPU Score: {min_gpu_score}, Rec GPU Score: {rec_gpu_score}, User GPU Score: {user_gpu_score}")
                    
                    # Apply sophisticated tiering based on real hardware comparison
                    if min_gpu_score > 0 and user_gpu_score > 0:
                        # If below minimum requirements
                        if user_gpu_score < min_gpu_score:
                            # Set to F tier for below minimum
                            total_score = max(30, int(total_score * 0.65))
                            self.logger.info(f"Hardware below minimum requirements, reducing score to {total_score}")
                        
                        # If between minimum and recommended
                        elif rec_gpu_score > min_gpu_score and user_gpu_score < rec_gpu_score:
                            # Set to C-B tier based on where in the range they fall
                            position = (user_gpu_score - min_gpu_score) / (rec_gpu_score - min_gpu_score)
                            tier_score = 60 + int(position * 20)  # C to B range (60-80)
                            total_score = min(tier_score, total_score)
                            self.logger.info(f"Hardware between min and rec, setting score to {total_score}")
                        
                        # If above recommended
                        elif user_gpu_score >= rec_gpu_score:
                            # How much above recommended?
                            exceed_factor = user_gpu_score / rec_gpu_score
                            if exceed_factor >= 1.5:
                                # Significantly above recommended - S tier
                                total_score = max(total_score, 100)
                                self.logger.info(f"Hardware well above rec, setting to S tier ({total_score})")
                            elif exceed_factor >= 1.2:
                                # Moderately above recommended - A tier
                                total_score = max(total_score, 80)
                                self.logger.info(f"Hardware above rec, setting to A tier ({total_score})")
                            else:
                                # Just above recommended - B tier
                                total_score = max(total_score, 70)
                                self.logger.info(f"Hardware meets rec, setting to B tier ({total_score})")
            
            # Determine tier
            tier = self._get_tier(total_score)
            self.logger.info(f"Final performance tier: {tier.name} with score {total_score}")
        
        # Calculate expected FPS with game-specific adjustments
        expected_fps = self._calculate_expected_fps(tier, scores['gpu'], scores['cpu'], game_requirements, hardware)
        
        # Determine settings and resolution
        recommended_settings, recommended_resolution = self._determine_recommendations(tier, total_score)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(scores)
        
        # Generate upgrade suggestions
        upgrade_suggestions = self._generate_upgrade_suggestions(hardware, scores, tier)
        
        assessment = PerformanceAssessment(
            score=total_score,
            tier=tier,
            tier_description=tier.value[2],
            expected_fps=expected_fps,
            recommended_settings=recommended_settings,
            recommended_resolution=recommended_resolution,
            bottlenecks=bottlenecks,
            upgrade_suggestions=upgrade_suggestions
        )
        
        self.logger.info(f"Dynamic performance assessment: Score {assessment.score}, Tier {assessment.tier.name}")
        
        return assessment
    
    def _get_tier(self, score: float) -> PerformanceTier:
        """Convert score to tier"""
        for tier in PerformanceTier:
            min_score, max_score, _ = tier.value
            if min_score <= score <= max_score:
                return tier
        return PerformanceTier.F
    
    def _calculate_expected_fps(self, tier: PerformanceTier, gpu_score: float, cpu_score: float, game_requirements: Dict = None, hardware: Dict = None) -> int:
        """
        Calculate expected FPS using the research-based formula:
        Predicted FPS = Base_FPS × (User_GPU_Score / Recommended_GPU_Score) × CPU_Modifier × Resolution_Factor
        
        Args:
            tier: Performance tier classification
            gpu_score: GPU score (0-100)
            cpu_score: CPU score (0-100)
            game_requirements: Optional game requirements from Steam API
            
        Returns:
            Expected FPS value calculated using the research formula
        """
        # Base FPS by tier - research-based starting points
        base_fps = {
            PerformanceTier.S: 90,   # Ultra settings, 4K@60fps+
            PerformanceTier.A: 75,   # High settings, 1440p@60fps
            PerformanceTier.B: 60,   # High settings, 1080p@60fps
            PerformanceTier.C: 40,   # Medium settings, 1080p@30fps
            PerformanceTier.D: 30,   # Low settings, 720p@30fps
            PerformanceTier.F: 20    # Below minimum
        }
        
        base_fps_value = base_fps.get(tier, 30)
        
        # Research formula implementation: Predicted FPS = Base_FPS × (User_GPU_Score / Recommended_GPU_Score) × CPU_Modifier × Resolution_Factor
        gpu_ratio = 1.0
        cpu_modifier = 1.0
        resolution_factor = 1.0
        
        # Calculate GPU performance ratio if game requirements available
        if game_requirements:
            rec_gpu = game_requirements.get('recommended_gpu', '')
            if rec_gpu:
                rec_gpu_score = self._estimate_gpu_benchmark(rec_gpu)
                if rec_gpu_score > 0:
                    # Get user's actual GPU benchmark score
                    user_gpu_benchmark = self._get_user_gpu_benchmark_score(game_requirements)
                    if user_gpu_benchmark > 0:
                        gpu_ratio = user_gpu_benchmark / rec_gpu_score
                        self.logger.debug(f"GPU ratio: {user_gpu_benchmark} / {rec_gpu_score} = {gpu_ratio}")
        
        # Calculate CPU modifier (typically 0.8-1.0 as per research)
        if cpu_score >= 85:
            cpu_modifier = 1.0      # No CPU bottleneck
        elif cpu_score >= 70:
            cpu_modifier = 0.95     # Slight CPU limitation
        elif cpu_score >= 60:
            cpu_modifier = 0.9      # Moderate CPU limitation
        elif cpu_score >= 50:
            cpu_modifier = 0.85     # Significant CPU limitation
        else:
            cpu_modifier = 0.8      # Severe CPU bottleneck
        
        # Calculate resolution factor based on actual current resolution
        resolution_factor = 1.0
        if hardware and 'display' in hardware:
            current_res = hardware['display']['primary_resolution']
            width = current_res.get('width', 1920)
            height = current_res.get('height', 1080)
            
            # Calculate total pixels
            current_pixels = width * height
            
            # Reference resolutions and their performance impact
            resolution_benchmarks = {
                3840 * 2160: 0.6,   # 4K - most demanding
                2560 * 1440: 0.8,   # 1440p - moderate demand
                1920 * 1080: 1.0,   # 1080p - baseline
                1280 * 720: 1.4,    # 720p - performance boost
            }
            
            # Find the closest resolution benchmark
            closest_pixels = min(resolution_benchmarks.keys(), key=lambda x: abs(x - current_pixels))
            resolution_factor = resolution_benchmarks[closest_pixels]
            
            # Interpolate for resolutions between benchmarks
            if current_pixels != closest_pixels:
                sorted_resolutions = sorted(resolution_benchmarks.keys())
                for i in range(len(sorted_resolutions) - 1):
                    lower_res = sorted_resolutions[i]
                    upper_res = sorted_resolutions[i + 1]
                    
                    if lower_res <= current_pixels <= upper_res:
                        # Linear interpolation
                        lower_factor = resolution_benchmarks[lower_res]
                        upper_factor = resolution_benchmarks[upper_res]
                        
                        position = (current_pixels - lower_res) / (upper_res - lower_res)
                        resolution_factor = lower_factor + (upper_factor - lower_factor) * position
                        break
            
            self.logger.info(f"Current resolution: {width}x{height} ({current_pixels:,} pixels), factor: {resolution_factor:.2f}")
        else:
            # Fallback: use tier-based resolution factor
            if tier == PerformanceTier.S:
                resolution_factor = 0.6     # Assume 4K for S tier
            elif tier == PerformanceTier.A:
                resolution_factor = 0.8     # Assume 1440p for A tier
            elif tier in [PerformanceTier.B, PerformanceTier.C]:
                resolution_factor = 1.0     # Assume 1080p for B/C tier
            else:
                resolution_factor = 1.4     # Assume 720p for D/F tier
            
            self.logger.info(f"Using tier-based resolution factor: {resolution_factor:.2f}")
        
        # Apply the research formula
        predicted_fps = int(base_fps_value * gpu_ratio * cpu_modifier * resolution_factor)
        
        self.logger.info(f"FPS Formula: {base_fps_value} × {gpu_ratio:.2f} × {cpu_modifier:.2f} × {resolution_factor:.2f} = {predicted_fps}")
        
        fps = predicted_fps
        
        # Game-specific adjustments if available
        if game_requirements:
            game_name = (
                game_requirements.get('game_name', '') or
                game_requirements.get('minimum_game', '') or
                game_requirements.get('recommended_game', '') or
                game_requirements.get('name', '')
            )
            
            if game_name:
                self.logger.info(f"Calculating game-specific FPS for {game_name}")
                
                # Check if the game is known to be well-optimized or demanding
                fps_modifier = 1.0  # Default modifier
                
                # List of known well-optimized games
                well_optimized_games = [
                    'fortnite', 'valorant', 'apex legends', 'rocket league',
                    'league of legends', 'counter-strike', 'counter strike', 'cs2',
                    'overwatch', 'minecraft', 'dota 2', 'rainbow six siege'
                ]
                
                # List of known demanding games
                demanding_games = [
                    'cyberpunk 2077', 'cyberpunk', 'red dead redemption 2', 'red dead redemption',
                    'assassin\'s creed valhalla', 'assassin\'s creed', 'flight simulator',
                    'control', 'metro exodus', 'crysis', 'star citizen', 'elden ring'
                ]
                
                # Apply game-specific adjustments
                game_lower = game_name.lower()
                
                if any(optimized_game in game_lower for optimized_game in well_optimized_games):
                    fps_modifier = 1.2  # 20% FPS boost for well-optimized games
                    self.logger.info(f"Well-optimized game detected, applying 20% FPS boost")
                elif any(demanding_game in game_lower for demanding_game in demanding_games):
                    fps_modifier = 0.8  # 20% FPS reduction for demanding games
                    self.logger.info(f"Demanding game detected, reducing FPS by 20%")
                
                # Modify the base FPS by game optimization factor
                fps = int(fps * fps_modifier)
                
                # Check for specific game engines
                if 'unreal engine' in game_lower or 'unreal' in game_lower:
                    # Unreal Engine games tend to be more GPU-bound
                    if gpu_score < 60:
                        fps = int(fps * 0.9)  # Further reduce for low-end GPUs
                    elif gpu_score > 85:
                        fps = int(fps * 1.1)  # Boost for high-end GPUs
                elif 'unity' in game_lower:
                    # Unity games are often more balanced between CPU and GPU
                    if min(cpu_score, gpu_score) < 60:
                        fps = int(fps * 0.9)  # Reduce for balanced bottleneck
                
                # Compare user's hardware to game requirements
                min_gpu = game_requirements.get('minimum_gpu', '')
                rec_gpu = game_requirements.get('recommended_gpu', '')
                
                # Get benchmark scores
                min_gpu_score = self._estimate_gpu_benchmark(min_gpu)
                rec_gpu_score = self._estimate_gpu_benchmark(rec_gpu)
                
                # Find user's GPU benchmark - use the actual hardware info, not from game_requirements
                gpu_model = ""
                if 'user_gpu_model' in game_requirements:
                    gpu_model = game_requirements.get('user_gpu_model', '')
                else:
                    # Get from hardware data structure
                    gpu_model = hardware['gpu']['cards'][0]['name'] if hardware['gpu']['cards'] else ''
                    
                user_gpu_benchmark = 0
                
                for pattern, benchmark in self.calculator.gpu_benchmarks.items():
                    if re.search(pattern, gpu_model, re.IGNORECASE):
                        user_gpu_benchmark = benchmark
                        break
                
                # Calculate performance ratio if we have valid benchmarks
                if min_gpu_score > 0 and rec_gpu_score > 0 and user_gpu_benchmark > 0:
                    # How much the user exceeds minimum requirements
                    min_ratio = user_gpu_benchmark / min_gpu_score if min_gpu_score > 0 else 1.0
                    
                    # How close the user is to recommended requirements
                    rec_ratio = user_gpu_benchmark / rec_gpu_score if rec_gpu_score > 0 else 0.5
                    
                    # Apply precise adjustments based on hardware vs requirements
                    if min_ratio < 1.0:
                        # Below minimum requirements - significant FPS reduction
                        fps = int(fps * min_ratio * 0.8)
                        self.logger.info(f"Below minimum requirements, reducing FPS to {fps}")
                    elif rec_ratio >= 1.5:
                        # Far exceeds recommended - significant FPS boost
                        fps = int(fps * 1.3)
                        self.logger.info(f"Far exceeds recommended requirements, boosting FPS to {fps}")
                    elif rec_ratio >= 1.0:
                        # Meets or exceeds recommended - moderate FPS boost
                        fps = int(fps * 1.15)
                        self.logger.info(f"Exceeds recommended requirements, boosting FPS to {fps}")
                    else:
                        # Between minimum and recommended - proportional adjustment
                        position = (min_ratio - 1.0) / (1.0 - rec_ratio)
                        fps_factor = 1.0 + (position * 0.15)  # 0-15% boost
                        fps = int(fps * fps_factor)
                        self.logger.info(f"Between min and rec requirements, adjusted FPS to {fps}")
        
        # Standard adjustments based on component scores
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
        
        # Return with reasonable lower bound
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
    
    def _identify_bottlenecks(self, scores: Dict) -> List[str]:
        """Identify system bottlenecks"""
        bottlenecks = []
        
        # Find the lowest scoring component(s)
        min_score = min(scores.values())
        avg_score = sum(scores.values()) / len(scores)
        
        for component, score in scores.items():
            if score <= min_score + 5 and score < avg_score - 10:
                bottlenecks.append(component.upper())
        
        return bottlenecks
    
    def _generate_upgrade_suggestions(self, hardware: Dict, scores: Dict, tier: PerformanceTier) -> List[str]:
        """Generate tier-aware hardware upgrade suggestions for NVIDIA systems"""
        suggestions = []
        
        # S Tier (90-100): Exceptional performance - focus on optimization, not upgrades
        if tier == PerformanceTier.S:
            gpu_name = hardware['gpu']['cards'][0]['name'] if hardware['gpu']['cards'] else ''
            if 'rtx' in gpu_name.lower():
                suggestions.append("Enable DLSS Quality mode for even higher framerates")
                if any(series in gpu_name.lower() for series in ['rtx 20', 'rtx 30', 'rtx 40']):
                    suggestions.append("Enable RTX ray tracing for enhanced visual quality")
            suggestions.append("Consider overclocking GPU for maximum performance")
            suggestions.append("Ensure adequate cooling for sustained peak performance")
            return suggestions
        
        # A Tier (80-89): Excellent performance - minor optimizations
        elif tier == PerformanceTier.A:
            gpu_name = hardware['gpu']['cards'][0]['name'] if hardware['gpu']['cards'] else ''
            if 'rtx' in gpu_name.lower():
                suggestions.append("Enable DLSS for better performance with RTX cards")
            if scores['cpu'] < 80:
                suggestions.append("CPU upgrade could unlock additional performance")
            ram_gb = hardware['ram']['total']
            if ram_gb < 32-1: #(publishers adjust ram to pow(2) values)
                suggestions.append("Consider 32GB RAM for content creation and multitasking")
            return suggestions
        
        # B Tier (70-79): Good performance - targeted improvements
        elif tier == PerformanceTier.B:
            # Focus on the lowest scoring component first
            min_component = min(scores, key=scores.get)
            min_score = scores[min_component]
            
            if min_component == 'gpu' and scores['gpu'] < 75:
                suggestions.append("GPU upgrade recommended - Consider RTX 4060 Ti or RTX 4070")
            elif min_component == 'cpu' and scores['cpu'] < 75:
                suggestions.append("CPU upgrade recommended for better performance")
            elif min_component == 'ram':
                ram_gb = hardware['ram']['total']
                if ram_gb < 16-1:
                    suggestions.append("Upgrade to 16GB+ RAM for optimal performance")
            
            gpu_name = hardware['gpu']['cards'][0]['name'] if hardware['gpu']['cards'] else ''
            if 'rtx' in gpu_name.lower():
                suggestions.append("Enable DLSS for better performance")
            return suggestions
        
        # C Tier (60-69): Adequate performance - clear upgrade path needed
        elif tier == PerformanceTier.C:
            # Prioritize GPU upgrade for C tier
            if scores['gpu'] < 70:
                suggestions.append("GPU upgrade recommended - Consider RTX 4060 or RTX 4070")
            
            # Address other bottlenecks
            if scores['cpu'] < 65:
                suggestions.append("CPU upgrade needed for better performance")
            
            ram_gb = hardware['ram']['total']
            if ram_gb < 16-1:
                suggestions.append("Upgrade to 16GB RAM for modern gaming")
            
            return suggestions
        
        # D Tier (50-59): Minimum performance - essential upgrades needed
        elif tier == PerformanceTier.D:
            suggestions.append("GPU upgrade essential - Consider RTX 3060 or RTX 4060")
            
            if scores['cpu'] < 60:
                suggestions.append("CPU upgrade essential for acceptable performance")
            
            ram_gb = hardware['ram']['total']
            if ram_gb < 16-1:
                suggestions.append("Upgrade to 16GB RAM essential for modern games")
            
            suggestions.append("Consider lowering graphics settings to improve performance")
            return suggestions
        
        # F Tier (0-49): Below minimum - major system upgrade required
        else:  # PerformanceTier.F
            suggestions.append("Major system upgrade required - Current hardware below minimum")
            suggestions.append("GPU upgrade critical - Consider RTX 3060 or newer")
            
            if scores['cpu'] < 50:
                suggestions.append("CPU upgrade critical for any modern gaming")
            
            ram_gb = hardware['ram']['total']
            if ram_gb < 8-1:
                suggestions.append("RAM upgrade to at least 16GB essential")
            
            suggestions.append("Consider building a new gaming system")
            return suggestions
        
    def _estimate_gpu_benchmark(self, gpu_name: str) -> int:
        """
        Estimate GPU benchmark score from name string using JSON hierarchy data.
        
        Args:
            gpu_name: Name of the GPU from requirements
            
        Returns:
            Estimated benchmark score (0 if can't estimate)
        """
        if not gpu_name or not isinstance(gpu_name, str):
            return 0
        
        gpu_name_clean = gpu_name.strip()
        
        # First try exact pattern matching using the calculator's benchmarks (from JSON)
        for pattern, benchmark in self.calculator.gpu_benchmarks.items():
            if re.search(pattern, gpu_name_clean, re.IGNORECASE):
                self.logger.debug(f"GPU requirement '{gpu_name_clean}' matched pattern '{pattern}' with score {benchmark}")
                return benchmark
        
        # Try direct lookup in GPU hierarchy data
        if self.calculator.gpu_hierarchy and 'nvidia' in self.calculator.gpu_hierarchy:
            nvidia_gpus = self.calculator.gpu_hierarchy['nvidia']
            
            # Try exact match first
            for gpu_key, gpu_data in nvidia_gpus.items():
                if gpu_key.lower() == gpu_name_clean.lower():
                    score = gpu_data.get('score', 500)
                    benchmark_score = int(score * 45)  # Convert to PassMark range
                    self.logger.debug(f"GPU requirement '{gpu_name_clean}' found exact match '{gpu_key}' with score {benchmark_score}")
                    return benchmark_score
            
            # Try partial match
            for gpu_key, gpu_data in nvidia_gpus.items():
                if gpu_key.lower() in gpu_name_clean.lower() or gpu_name_clean.lower() in gpu_key.lower():
                    score = gpu_data.get('score', 500)
                    benchmark_score = int(score * 45)  # Convert to PassMark range
                    self.logger.debug(f"GPU requirement '{gpu_name_clean}' found partial match '{gpu_key}' with score {benchmark_score}")
                    return benchmark_score
        
        # Fallback to simplified estimation if no JSON match
        gpu_lower = gpu_name_clean.lower()
        if 'rtx' in gpu_lower:
            if any(model in gpu_lower for model in ['4090', '4080']):
                return 40000  # High-end RTX 40 series
            elif '40' in gpu_lower:
                return 30000  # Mid-range RTX 40 series
            elif any(model in gpu_lower for model in ['3090', '3080']):
                return 35000  # High-end RTX 30 series
            elif '30' in gpu_lower:
                return 25000  # Mid-range RTX 30 series
            elif '20' in gpu_lower:
                return 18000  # RTX 20 series
            else:
                return 20000  # Generic RTX
        elif 'gtx' in gpu_lower:
            if '16' in gpu_lower:
                return 8000   # GTX 16 series
            elif any(model in gpu_lower for model in ['1080', '1070']):
                return 10000  # High-end GTX 10 series
            elif '10' in gpu_lower:
                return 6000   # Mid-range GTX 10 series
            else:
                return 5000   # Generic GTX
        elif 'nvidia' in gpu_lower:
            return 10000      # Generic NVIDIA
        elif any(brand in gpu_lower for brand in ['amd', 'radeon']):
            if 'rx 7' in gpu_lower:
                return 30000  # High-end AMD RX 7000
            elif 'rx 6' in gpu_lower:
                return 20000  # AMD RX 6000
            elif 'rx' in gpu_lower:
                return 10000  # Generic AMD RX
            else:
                return 8000   # Generic AMD
        
        # Fallback for unknown GPU
        self.logger.debug(f"No match found for GPU '{gpu_name_clean}', using fallback score")
        return 5000
    
    def _get_user_gpu_benchmark_score(self, game_requirements: Dict) -> int:
        """
        Get the user's actual GPU benchmark score for the research formula.
        
        Args:
            game_requirements: Game requirements dict that may contain user hardware info
            
        Returns:
            User's GPU benchmark score (0 if can't determine)
        """
        # Try to get GPU model from various sources
        gpu_model = ""
        
        # Check if user GPU model is provided in game requirements
        if 'user_gpu_model' in game_requirements:
            gpu_model = game_requirements.get('user_gpu_model', '')
        elif hasattr(self, 'hardware') and self.hardware.get('gpu', {}).get('cards'):
            # Get from hardware data if available
            gpu_model = self.hardware['gpu']['cards'][0]['name']
        else:
            # Fallback: try to get from current hardware detection
            try:
                hardware = self.hardware_detector.detect_all()
                if hardware.get('gpu', {}).get('cards'):
                    gpu_model = hardware['gpu']['cards'][0]['name']
            except Exception as e:
                self.logger.debug(f"Failed to detect GPU for benchmark score: {e}")
                return 0
        
        if not gpu_model:
            self.logger.debug("No GPU model available for benchmark score calculation")
            return 0
        
        # Find matching benchmark using pattern matching
        for pattern, benchmark in self.calculator.gpu_benchmarks.items():
            if re.search(pattern, gpu_model, re.IGNORECASE):
                self.logger.debug(f"User GPU '{gpu_model}' matched pattern '{pattern}' with score {benchmark}")
                return benchmark
        
        # Fallback estimation if no exact match
        return self._estimate_gpu_benchmark(gpu_model)