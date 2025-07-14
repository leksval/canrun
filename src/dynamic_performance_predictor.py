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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

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
            'os': self._detect_os()
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


class PerformanceCalculator:
    """Dynamic performance calculation with pattern matching for NVIDIA GPUs"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # CPU benchmark scores (approximate PassMark scores)
        self.cpu_benchmarks = self._load_cpu_benchmarks()
        # GPU benchmark scores (approximate PassMark G3D scores) - NVIDIA only
        self.gpu_benchmarks = self._load_gpu_benchmarks()
        
    def _load_cpu_benchmarks(self) -> Dict[str, int]:
        """Load CPU benchmark data dynamically using pattern matching"""
        return {
            # Intel 13th Gen
            r'i9-1[3-4]\d{3}[A-Z]*': 35000,
            r'i7-1[3-4]\d{3}[A-Z]*': 25000,
            r'i5-1[3-4]\d{3}[A-Z]*': 15000,
            r'i3-1[3-4]\d{3}[A-Z]*': 8000,
            
            # Intel 12th Gen
            r'i9-12\d{3}[A-Z]*': 32000,
            r'i7-12\d{3}[A-Z]*': 22000,
            r'i5-12\d{3}[A-Z]*': 13000,
            r'i3-12\d{3}[A-Z]*': 7000,
            
            # AMD Ryzen 7000
            r'Ryzen 9 7\d{3}[A-Z]*': 35000,
            r'Ryzen 7 7800X3D': 38000,  # Special case for 3D V-Cache
            r'Ryzen 7 7\d{3}[A-Z]*': 25000,
            r'Ryzen 5 7\d{3}[A-Z]*': 15000,
            r'Ryzen 3 7\d{3}[A-Z]*': 8000,
            
            # AMD Ryzen 5000
            r'Ryzen 9 5\d{3}[A-Z]*': 30000,
            r'Ryzen 7 5800X3D': 32000,  # Special case for 3D V-Cache
            r'Ryzen 7 5\d{3}[A-Z]*': 22000,
            r'Ryzen 5 5\d{3}[A-Z]*': 13000,
            r'Ryzen 3 5\d{3}[A-Z]*': 7000,
            
            # Apple Silicon
            r'Apple M[1-4]': 20000,
            r'Apple M[1-4] Pro': 30000,
            r'Apple M[1-4] Max': 40000,
        }
    
    def _load_gpu_benchmarks(self) -> Dict[str, int]:
        """Load NVIDIA GPU benchmark data with RTX 5000 series and laptop variants"""
        return {
            # NVIDIA RTX 50 Series (Future-proofing)
            r'RTX 5090': 55000,
            r'RTX 5080': 48000,
            r'RTX 507[0-9]': 40000,
            r'RTX 506[0-9]': 32000,
            
            # NVIDIA RTX 40 Series Desktop
            r'RTX 4090': 45000,
            r'RTX 4080 Super': 40000,
            r'RTX 4080': 38000,
            r'RTX 4070 Ti Super': 35000,
            r'RTX 4070 Ti': 32000,
            r'RTX 4070 Super': 30000,
            r'RTX 4070': 28000,
            r'RTX 4060 Ti': 22000,
            r'RTX 4060': 18000,
            
            # NVIDIA RTX 40 Series Laptop
            r'RTX 4090 Laptop': 38000,
            r'RTX 4080 Laptop': 32000,
            r'RTX 4070 Laptop': 24000,
            r'RTX 4060 Laptop': 16000,
            r'RTX 4050 Laptop': 12000,
            
            # NVIDIA RTX 30 Series Desktop
            r'RTX 3090 Ti': 39000,
            r'RTX 3090': 35000,
            r'RTX 3080 Ti': 32000,
            r'RTX 3080': 28000,
            r'RTX 3070 Ti': 25000,
            r'RTX 3070': 22000,
            r'RTX 3060 Ti': 18000,
            r'RTX 3060': 15000,
            r'RTX 3050': 10000,
            
            # NVIDIA RTX 30 Series Laptop
            r'RTX 3080 Ti Laptop': 26000,
            r'RTX 3080 Laptop': 22000,
            r'RTX 3070 Ti Laptop': 20000,
            r'RTX 3070 Laptop': 18000,
            r'RTX 3060 Laptop': 13000,
            r'RTX 3050 Ti Laptop': 9000,
            r'RTX 3050 Laptop': 8000,
            
            # NVIDIA RTX 20 Series Desktop
            r'RTX 2080 Ti': 20000,
            r'RTX 2080 Super': 18000,
            r'RTX 2080': 16000,
            r'RTX 2070 Super': 15000,
            r'RTX 2070': 13000,
            r'RTX 2060 Super': 12000,
            r'RTX 2060': 10000,
            
            # NVIDIA RTX 20 Series Laptop
            r'RTX 2080 Laptop': 14000,
            r'RTX 2070 Laptop': 11000,
            r'RTX 2060 Laptop': 9000,
            
            # NVIDIA GTX 16 Series Desktop
            r'GTX 1660 Ti': 9000,
            r'GTX 1660 Super': 8500,
            r'GTX 1660': 8000,
            r'GTX 1650 Super': 7000,
            r'GTX 1650': 6000,
            
            # NVIDIA GTX 16 Series Laptop
            r'GTX 1660 Ti Laptop': 7500,
            r'GTX 1660 Ti Mobile': 7500,
            r'GTX 1650 Ti Laptop': 5500,
            r'GTX 1650 Laptop': 5000,
            
            # NVIDIA GTX 10 Series Desktop
            r'GTX 1080 Ti': 12000,
            r'GTX 1080': 10000,
            r'GTX 1070 Ti': 9000,
            r'GTX 1070': 8000,
            r'GTX 1060 6GB': 6500,
            r'GTX 1060 3GB': 5500,
            r'GTX 1060': 6000,
            r'GTX 1050 Ti': 4000,
            r'GTX 1050': 3000,
            
            # NVIDIA GTX 10 Series Laptop
            r'GTX 1080 Laptop': 8500,
            r'GTX 1070 Laptop': 7000,
            r'GTX 1060 Laptop': 5000,
            r'GTX 1050 Ti Laptop': 3200,
            r'GTX 1050 Laptop': 2500,
            
            # NVIDIA GTX 9 Series
            r'GTX 980 Ti': 7000,
            r'GTX 980': 6000,
            r'GTX 970': 5000,
            r'GTX 960': 3500,
            r'GTX 950': 2500,
            
            # Generic patterns for unmatched cards
            r'GeForce.*RTX': 15000,  # Generic RTX fallback
            r'GeForce.*GTX': 5000,   # Generic GTX fallback
            r'NVIDIA.*RTX': 15000,   # Generic NVIDIA RTX
            r'NVIDIA.*GTX': 5000,    # Generic NVIDIA GTX
        }
    
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
        """Estimate required CPU score from string"""
        patterns = {
            r'i9|Ryzen 9': 30000,
            r'i7|Ryzen 7': 20000,
            r'i5|Ryzen 5': 12000,
            r'i3|Ryzen 3': 6000,
            r'Core 2 Duo|Dual.?Core': 2000,
            r'Quad.?Core': 4000,
        }
        
        for pattern, score in patterns.items():
            if re.search(pattern, cpu_string, re.IGNORECASE):
                return score
        
        return 8000  # Default middle-range requirement
    
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
            game_requirements: Optional game requirements
            
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
                }
            }
        
        # Calculate individual scores
        scores = {
            'cpu': self.calculator.calculate_cpu_score(hardware['cpu'], game_requirements or {}),
            'gpu': self.calculator.calculate_gpu_score(hardware['gpu'], game_requirements or {}),
            'ram': self.calculator.calculate_ram_score(hardware['ram'], game_requirements or {})
        }
        
        self.logger.debug(f"Component scores: {scores}")
        
        # Calculate weighted total score
        total_score = int(
            scores['gpu'] * self.weights['gpu'] +
            scores['cpu'] * self.weights['cpu'] +
            scores['ram'] * self.weights['ram']
        )
        
        # Determine tier
        tier = self._get_tier(total_score)
        
        # Calculate expected FPS
        expected_fps = self._calculate_expected_fps(tier, scores['gpu'], scores['cpu'])
        
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
    
    def _calculate_expected_fps(self, tier: PerformanceTier, gpu_score: float, cpu_score: float) -> int:
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
        """Generate hardware upgrade suggestions for NVIDIA systems"""
        suggestions = []
        
        # GPU upgrades
        if scores['gpu'] < 70:
            if tier == PerformanceTier.F or tier == PerformanceTier.D:
                suggestions.append("GPU upgrade essential - Consider RTX 3060 or RTX 4060")
            elif tier == PerformanceTier.C:
                suggestions.append("GPU upgrade recommended - Consider RTX 3070 or RTX 4070")
        
        # CPU upgrades
        if scores['cpu'] < 65:
            suggestions.append("CPU upgrade recommended for better performance")
        
        # RAM upgrades
        ram_gb = hardware['ram']['total']
        if ram_gb < 16:
            suggestions.append("Upgrade to 16GB+ RAM for optimal performance")
        elif ram_gb < 32 and tier == PerformanceTier.S:
            suggestions.append("Consider 32GB RAM for maximum performance")
        
        # DLSS/RTX suggestions
        gpu_name = hardware['gpu']['cards'][0]['name'] if hardware['gpu']['cards'] else ''
        if 'rtx' in gpu_name.lower():
            suggestions.append("Enable DLSS for better performance with RTX cards")
            if any(series in gpu_name.lower() for series in ['rtx 20', 'rtx 30', 'rtx 40']):
                suggestions.append("Consider enabling RTX ray tracing for enhanced visuals")
        
        return suggestions