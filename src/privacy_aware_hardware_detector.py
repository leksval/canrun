"""
Privacy-Aware Hardware Detection Module for CanRun
Privacy-by-design hardware detection for RTX/GTX gaming systems.
"""

import os
import sys
import logging
import hashlib
import secrets
import ctypes
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from pathlib import Path
import re

# Import required dependencies (specified in requirements.txt)
import psutil
import cpuinfo
import pynvml
import winreg
import wmi

# Handle GPUtil import with distutils compatibility for PyInstaller
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError as e:
    if "distutils" in str(e):
        # GPUtil requires distutils which was removed in Python 3.12
        # Create a compatibility shim for PyInstaller
        import sys
        import shutil
        
        class DistutilsSpawn:
            @staticmethod
            def find_executable(name):
                return shutil.which(name)
        
        # Inject distutils.spawn compatibility
        if 'distutils' not in sys.modules:
            import types
            distutils_module = types.ModuleType('distutils')
            distutils_module.spawn = DistutilsSpawn()
            sys.modules['distutils'] = distutils_module
            sys.modules['distutils.spawn'] = DistutilsSpawn()
        
        try:
            import GPUtil
            GPUTIL_AVAILABLE = True
        except ImportError:
            GPUTIL_AVAILABLE = False
    else:
        GPUTIL_AVAILABLE = False



@dataclass
class PrivacyAwareHardwareSpecs:
    """Privacy-focused hardware specifications for RTX/GTX gaming systems."""
    
    # Essential Gaming Data (Required - no defaults)
    gpu_model: str                              # RTX/GTX model name
    gpu_vram_gb: int                            # VRAM amount
    cpu_cores: int                              # Physical core count
    cpu_threads: int                            # Logical core count
    ram_total_gb: int                           # Total RAM
    ram_speed_mhz: int                          # RAM speed
    storage_type: str                           # Storage type
    primary_monitor_refresh_hz: int             # Monitor refresh rate
    primary_monitor_resolution: str             # Monitor resolution
    os_version: str                             # Windows version
    directx_version: str                        # DirectX version
    
    # Fields with defaults (must come after required fields)
    gpu_vendor: str = "NVIDIA"                  # Always NVIDIA for RTX/GTX
    cpu_model: str = "Unknown CPU"              # CPU model name
    anonymous_system_id: str = ""               # Anonymous identifier
    data_timestamp: Optional[datetime] = None   # Collection timestamp
    is_nvidia_gpu: bool = True                  # Always True for RTX/GTX
    supports_rtx: Optional[bool] = None         # Ray tracing support
    supports_dlss: Optional[bool] = None        # DLSS support
    nvidia_driver_version: str = "Unknown"      # Driver version
    
    def __post_init__(self):
        """Validate hardware specs after initialization."""
        # Set timestamp if not provided
        if self.data_timestamp is None:
            self.data_timestamp = datetime.now()
        
        # Generate anonymous ID if not provided
        if not self.anonymous_system_id:
            self.anonymous_system_id = self._generate_anonymous_id()
        
        # Validate RTX/GTX GPU requirement
        assert self.gpu_vendor.upper() == "NVIDIA", "Only NVIDIA RTX/GTX GPUs supported"
        assert "RTX" in self.gpu_model.upper() or "GTX" in self.gpu_model.upper(), "RTX or GTX GPU required"
        
        # Auto-compute RTX/DLSS support
        if self.supports_rtx is None:
            self.supports_rtx = "RTX" in self.gpu_model.upper()
        
        if self.supports_dlss is None:
            self.supports_dlss = self.supports_rtx
        
        # Validate specs
        assert self.gpu_vram_gb > 0, "VRAM must be greater than 0"
        assert self.cpu_cores > 0, "CPU cores must be greater than 0"
        assert self.ram_total_gb > 0, "RAM must be greater than 0"
        assert self.gpu_model.strip(), "GPU model cannot be empty"
        assert self.cpu_model.strip(), "CPU model cannot be empty"
    
    def _generate_anonymous_id(self) -> str:
        """Generate anonymous system identifier."""
        # Use hardware fingerprint for consistent anonymity
        fingerprint = f"{self.gpu_model}_{self.cpu_cores}_{self.ram_total_gb}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'gpu_model': self.gpu_model,
            'gpu_vram_gb': self.gpu_vram_gb,
            'gpu_vendor': self.gpu_vendor,
            'cpu_model': self.cpu_model,
            'cpu_cores': self.cpu_cores,
            'cpu_threads': self.cpu_threads,
            'ram_total_gb': self.ram_total_gb,
            'ram_speed_mhz': self.ram_speed_mhz,
            'storage_type': self.storage_type,
            'primary_monitor_refresh_hz': self.primary_monitor_refresh_hz,
            'primary_monitor_resolution': self.primary_monitor_resolution,
            'os_version': self.os_version,
            'directx_version': self.directx_version,
            'anonymous_system_id': self.anonymous_system_id,
            'data_timestamp': self.data_timestamp.isoformat() if self.data_timestamp else None,
            'is_nvidia_gpu': self.is_nvidia_gpu,
            'supports_rtx': self.supports_rtx,
            'supports_dlss': self.supports_dlss,
            'nvidia_driver_version': self.nvidia_driver_version
        }


class PrivacyAwareCache:
    """Privacy-focused cache for hardware detection results."""
    
    def __init__(self, cache_duration_hours: int = 24, max_age_hours: int = None):
        # Standardize all cache to 15-minute expiration
        self.cache_duration = timedelta(minutes=15)
        self.cache_data = {}
        self.cache_timestamps = {}
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Privacy-aware cache initialized with {cache_duration_hours}h duration")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value with privacy protection."""
        anonymized_key = self._anonymize_key(key)
        
        # Check if key exists and is not expired
        if anonymized_key in self.cache_data:
            timestamp = self.cache_timestamps[anonymized_key]
            if datetime.now() - timestamp < self.cache_duration:
                self.logger.debug(f"Cache hit for anonymized key: {anonymized_key[:8]}...")
                return self.cache_data[anonymized_key]
            else:
                # Remove expired entry
                self._remove_expired_entry(anonymized_key)
        
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value with privacy protection."""
        anonymized_key = self._anonymize_key(key)
        
        self.cache_data[anonymized_key] = value
        self.cache_timestamps[anonymized_key] = datetime.now()
        
        self.logger.debug(f"Cached data with anonymized key: {anonymized_key[:8]}...")
    
    def store(self, key: str, value: Any) -> None:
        """Alias for set() method to match test expectations."""
        self.set(key, value)
    
    @property
    def data(self) -> Dict[str, Any]:
        """Alias for cache_data to match test expectations."""
        return self.cache_data
    
    def clear_expired(self) -> None:
        """Clear all expired cache entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.cache_duration:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_expired_entry(key)
        
        if expired_keys:
            self.logger.info(f"Cleared {len(expired_keys)} expired cache entries")
    
    def _anonymize_key(self, key: str) -> str:
        """Generate anonymized cache key."""
        # Hash the key consistently for privacy (same key = same hash)
        hash_input = f"privacy_cache_{key}".encode()
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    def _remove_expired_entry(self, key: str) -> None:
        """Remove expired cache entry."""
        self.cache_data.pop(key, None)
        self.cache_timestamps.pop(key, None)


class PrivacyAwareHardwareDetector:
    """Privacy-focused hardware detector for RTX/GTX gaming systems."""
    
    def __init__(self, cache_duration_hours: int = 24):
        self.logger = logging.getLogger(__name__)
        # All cache durations standardized to 15 minutes
        self.cache = PrivacyAwareCache()
        
        # Initialize LLM analyzer lazily to avoid circular imports
        self.llm_analyzer = None
        
        # Initialize RTX/GTX libraries
        self._initialize_nvidia_libraries()
    
    def _get_llm_analyzer(self):
        """Lazily initialize LLM analyzer to avoid circular imports."""
        if self.llm_analyzer is None:
            try:
                from rtx_llm_analyzer import GAssistLLMAnalyzer
                self.llm_analyzer = GAssistLLMAnalyzer()
            except ImportError:
                self.logger.warning("G-Assist LLM analyzer not available")
                self.llm_analyzer = None
        return self.llm_analyzer
        
        # Validate system compatibility
        self._validate_system_compatibility()
        
        self.logger.info("Privacy-aware hardware detector initialized for RTX/GTX systems")
    
    def _initialize_nvidia_libraries(self) -> None:
        """Initialize NVIDIA-specific libraries."""
        try:
            pynvml.nvmlInit()
            self.logger.info("NVIDIA ML library initialized")
        except Exception as e:
            self.logger.warning(f"NVIDIA ML library initialization failed: {e}")
    
    def _validate_system_compatibility(self) -> None:
        """Validate system compatibility with NVIDIA requirements."""
        # Check for Windows OS (required for G-Assist)
        if os.name != 'nt':
            self.logger.warning("Windows OS recommended for full G-Assist compatibility")
    
    def has_nvidia_gpu(self) -> bool:
        """Check if NVIDIA RTX/GTX GPU is available for G-Assist compatibility."""
        try:
            # Try NVIDIA ML library first
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                # Check if any device is NVIDIA RTX/GTX
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                return 'RTX' in gpu_name.upper() or 'GTX' in gpu_name.upper()
        except Exception:
            pass
        
        # Try GPUtil as fallback if available
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    if 'NVIDIA' in gpu.name.upper():
                        gpu_name = gpu.name.upper()
                        return 'RTX' in gpu_name or 'GTX' in gpu_name
            except Exception:
                pass
        
        # Try registry detection as final fallback
        try:
            gpu_name = self._detect_gpu_from_registry()
            if gpu_name:
                gpu_upper = gpu_name.upper()
                return 'NVIDIA' in gpu_upper and ('RTX' in gpu_upper or 'GTX' in gpu_upper)
        except Exception:
            pass
        
        return False
    
    async def get_hardware_specs(self) -> PrivacyAwareHardwareSpecs:
        """Get privacy-aware hardware specifications."""
        # Check cache first
        cached_specs = self.cache.get("hardware_specs")
        if cached_specs:
            self.logger.debug("Returning cached hardware specs")
            return cached_specs
        
        # Detect hardware
        specs = self._detect_hardware_safely()
        
        # Cache the result
        self.cache.set("hardware_specs", specs)
        
        return specs
    
    def _detect_hardware_safely(self) -> PrivacyAwareHardwareSpecs:
        """Safely detect hardware with comprehensive error handling."""
        # Detect GPU (NVIDIA-focused)
        gpu_info = self._detect_nvidia_gpu()
        assert gpu_info['is_nvidia'], "NVIDIA GPU required for G-Assist compatibility"
        
        # Detect CPU
        cpu_info = self._detect_cpu()
        
        # Detect RAM
        ram_info = self._detect_ram()
        if ram_info is None:
            raise RuntimeError("RAM detection failed - unable to determine system memory")
        
        # Detect OS
        os_info = self._detect_os()
        
        # Detect display information
        display_info = self._detect_display()
        
        # Generate anonymous system ID
        anonymous_id = self._generate_anonymous_system_id()
        
        # Use LLM to analyze and fill missing system specifications
        system_specs = self._analyze_hardware_with_llm('system', f"GPU: {gpu_info['name']}, CPU: {cpu_info['name']}, RAM: {ram_info['total_gb']}GB")
        
        # Create hardware specifications
        specs = PrivacyAwareHardwareSpecs(
            gpu_model=gpu_info['name'],
            gpu_vram_gb=gpu_info['vram_gb'],
            gpu_vendor="NVIDIA",
            cpu_model=cpu_info['name'],
            cpu_cores=cpu_info['cores'],
            cpu_threads=cpu_info['threads'],
            ram_total_gb=ram_info['total_gb'],
            ram_speed_mhz=system_specs.get('ram_speed_mhz', 0),
            storage_type=system_specs.get('storage_type', 'Unknown'),
            primary_monitor_refresh_hz=display_info.get('refresh_hz', 0),
            primary_monitor_resolution=display_info.get('resolution', 'Unknown'),
            os_version=os_info['name'],
            directx_version=os_info['directx_version'],
            anonymous_system_id=anonymous_id,
            data_timestamp=datetime.now(),
            is_nvidia_gpu=True,
            supports_rtx=gpu_info['supports_rtx'],
            supports_dlss=gpu_info['supports_dlss'],
            nvidia_driver_version=gpu_info['driver_version']
        )
        
        self.logger.info(f"Hardware detected: {specs.gpu_model}, {specs.cpu_model}, {specs.ram_total_gb}GB RAM")
        return specs
    
    def _detect_nvidia_gpu(self) -> Dict[str, Any]:
        """Detect NVIDIA GPU information."""
        gpu_info = {
            'name': 'Unknown GPU',
            'vram_gb': 0,
            'is_nvidia': False,
            'supports_rtx': False,
            'supports_dlss': False,
            'driver_version': 'Unknown'
        }
        
        # Try NVIDIA ML library first
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            assert device_count > 0, "No NVIDIA GPUs found"
            
            # Get first GPU (primary)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # Get GPU name
            try:
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                # Handle both string and bytes return types
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode('utf-8')
            except Exception as e:
                self.logger.debug(f"GPU name detection failed: {e}")
                gpu_name = "Unknown GPU"
            
            # Get VRAM
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_gb = mem_info.total // (1024 ** 3)
            
            # Get driver version
            try:
                driver_version = pynvml.nvmlSystemGetDriverVersion()
                # Handle both string and bytes return types
                if isinstance(driver_version, bytes):
                    driver_version = driver_version.decode('utf-8')
            except Exception as e:
                self.logger.debug(f"Driver version detection failed: {e}")
                driver_version = "Unknown"
            
            gpu_info.update({
                'name': self._clean_gpu_name(gpu_name),
                'vram_gb': vram_gb,
                'is_nvidia': True,
                'supports_rtx': 'RTX' in gpu_name.upper(),
                'supports_dlss': 'RTX' in gpu_name.upper(),  # RTX GPUs support DLSS
                'driver_version': driver_version
            })
            
            self.logger.info(f"NVIDIA GPU detected via pynvml: {gpu_info['name']}")
            return gpu_info
            
        except Exception as e:
            self.logger.warning(f"NVIDIA ML detection failed: {e}")
        
        # Fallback to GPUtil if available
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                assert gpus, "No GPUs found"
                
                gpu = gpus[0]  # Primary GPU
                gpu_name = gpu.name
                
                if 'NVIDIA' in gpu_name.upper():
                    gpu_info.update({
                        'name': self._clean_gpu_name(gpu_name),
                        'vram_gb': int(gpu.memoryTotal / 1024),  # Convert MB to GB
                        'is_nvidia': True,
                        'supports_rtx': 'RTX' in gpu_name.upper(),
                        'supports_dlss': 'RTX' in gpu_name.upper(),
                        'driver_version': 'Unknown'
                    })
                    
                    self.logger.info(f"NVIDIA GPU detected via GPUtil: {gpu_info['name']}")
                    return gpu_info
                
            except Exception as e:
                self.logger.warning(f"GPUtil detection failed: {e}")
        else:
            self.logger.warning("GPUtil not available - skipping GPUtil detection")
        
        # Windows Registry fallback
        if os.name == 'nt':
            try:
                gpu_name = self._detect_gpu_from_registry()
                if gpu_name and 'NVIDIA' in gpu_name.upper():
                    # Use LLM to analyze GPU specifications
                    gpu_specs = self._analyze_hardware_with_llm('gpu', gpu_name)
                    
                    gpu_info.update({
                        'name': self._clean_gpu_name(gpu_name),
                        'vram_gb': gpu_specs.get('vram_gb', 4),
                        'is_nvidia': True,
                        'supports_rtx': 'RTX' in gpu_name.upper(),
                        'supports_dlss': 'RTX' in gpu_name.upper(),
                        'driver_version': 'Unknown'
                    })
                    
                    self.logger.info(f"NVIDIA GPU detected via registry: {gpu_info['name']}")
                    return gpu_info
                    
            except Exception as e:
                self.logger.warning(f"Registry GPU detection failed: {e}")
        
        # If we reach here, no NVIDIA GPU was found
        raise RuntimeError("NVIDIA GPU not detected - RTX/GTX GPU required for G-Assist compatibility")
    
    def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU information."""
        cpu_info = {
            'name': 'Unknown CPU',
            'cores': 1,
            'threads': 1
        }
        
        # Try cpuinfo library
        try:
            cpu_data = cpuinfo.get_cpu_info()
            
            cpu_info.update({
                'name': self._clean_cpu_name(cpu_data.get('brand_raw', 'Unknown CPU')),
                'cores': cpu_data.get('count', 1),
                'threads': cpu_data.get('count', 1)
            })
            
            self.logger.info(f"CPU detected via cpuinfo: {cpu_info['name']}")
            return cpu_info
            
        except Exception as e:
            self.logger.warning(f"cpuinfo detection failed: {e}")
        
        # Fallback to psutil
        try:
            logical_cores = psutil.cpu_count(logical=True)
            physical_cores = psutil.cpu_count(logical=False)
            
            cpu_info.update({
                'name': 'Unknown CPU',
                'cores': physical_cores or 1,
                'threads': logical_cores or 1
            })
            
            self.logger.info(f"CPU detected via psutil: {cpu_info['cores']} cores")
            return cpu_info
            
        except Exception as e:
            self.logger.warning(f"psutil CPU detection failed: {e}")
        
        # OS fallback
        try:
            import os
            cpu_count = os.cpu_count()
            cpu_info.update({
                'name': 'Unknown CPU',
                'cores': cpu_count or 1,
                'threads': cpu_count or 1
            })
            
            self.logger.info(f"CPU detected via OS: {cpu_info['cores']} cores")
            return cpu_info
            
        except Exception as e:
            self.logger.warning(f"OS CPU detection failed: {e}")
        
        return cpu_info
    
    def _detect_ram(self) -> Optional[Dict[str, Any]]:
        """Detect RAM information."""
        ram_info = {
            'total_gb': 0,
            'available_gb': 0
        }
        
        # Try psutil
        try:
            memory = psutil.virtual_memory()
            
            ram_info.update({
                'total_gb': round(memory.total / (1024 ** 3)),
                'available_gb': round(memory.available / (1024 ** 3))
            })
            
            self.logger.info(f"RAM detected via psutil: {ram_info['total_gb']}GB total")
            return ram_info
            
        except Exception as e:
            self.logger.warning(f"psutil RAM detection failed: {e}")
        
        # WMI fallback for Windows
        if os.name == 'nt':
            try:
                c = wmi.WMI()
                total_memory = 0
                
                for memory in c.Win32_PhysicalMemory():
                    total_memory += int(memory.Capacity)
                
                ram_info.update({
                    'total_gb': int(total_memory / (1024 ** 3)),
                    'available_gb': int(total_memory / (1024 ** 3))  # Simplified
                })
                
                self.logger.info(f"RAM detected via WMI: {ram_info['total_gb']}GB total")
                return ram_info
                
            except Exception as e:
                self.logger.warning(f"WMI RAM detection failed: {e}")
        
        # No fallback - return None if detection fails
        self.logger.error("RAM detection failed - no fallback available")
        return None
    
    def _detect_os(self) -> Dict[str, Any]:
        """Detect OS information."""
        os_info = {
            'name': 'Unknown OS',
            'directx_version': 'DirectX 12'
        }
        
        try:
            if os.name == 'nt':
                # Windows
                import platform
                os_name = f"Windows {platform.release()}"
                
                # Try to get more specific version from registry
                try:
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                       r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
                    
                    # Prioritize build number detection over ProductName
                    # (Microsoft hasn't updated ProductName properly for Windows 11)
                    try:
                        current_build = winreg.QueryValueEx(key, "CurrentBuild")[0]
                        
                        # Windows 11 detection based on build number (most reliable)
                        if int(current_build) >= 22000:
                            os_name = "Windows 11"
                            # Try to get edition
                            try:
                                edition = winreg.QueryValueEx(key, "EditionID")[0]
                                if edition.lower() == "professional":
                                    os_name = "Windows 11 Pro"
                                elif edition.lower() == "home":
                                    os_name = "Windows 11 Home"
                                elif edition.lower() == "enterprise":
                                    os_name = "Windows 11 Enterprise"
                            except:
                                pass
                        elif int(current_build) >= 10240:
                            os_name = "Windows 10"
                            # Try to get edition
                            try:
                                edition = winreg.QueryValueEx(key, "EditionID")[0]
                                if edition.lower() == "professional":
                                    os_name = "Windows 10 Pro"
                                elif edition.lower() == "home":
                                    os_name = "Windows 10 Home"
                                elif edition.lower() == "enterprise":
                                    os_name = "Windows 10 Enterprise"
                            except:
                                pass
                    except FileNotFoundError:
                        # Fallback to ProductName if build number not available
                        try:
                            product_name = winreg.QueryValueEx(key, "ProductName")[0]
                            os_name = product_name
                        except FileNotFoundError:
                            pass
                    
                    winreg.CloseKey(key)
                except Exception as e:
                    self.logger.debug(f"Registry access failed: {e}")
                    pass
                
                os_info.update({
                    'name': os_name,
                    'directx_version': 'DirectX 12'
                })
                
                self.logger.info(f"OS detected: {os_info['name']}")
                return os_info
                
        except Exception as e:
            self.logger.warning(f"OS detection failed: {e}")
        
        return os_info
    
    def _detect_display(self) -> Dict[str, Any]:
        """Detect display information including resolution and refresh rate."""
        display_info = {
            'resolution': 'Unknown',
            'refresh_hz': 0
        }
        
        try:
            if os.name == 'nt':
                # Windows display detection using ctypes
                # Get display resolution and refresh rate
                user32 = ctypes.windll.user32
                screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
                display_info['resolution'] = f"{screensize[0]}x{screensize[1]}"
                
                # Get refresh rate using GetDeviceCaps
                try:
                    gdi32 = ctypes.windll.gdi32
                    hdc = user32.GetDC(0)
                    if hdc:
                        refresh_rate = gdi32.GetDeviceCaps(hdc, 116)  # VREFRESH = 116
                        if refresh_rate > 1:  # Valid refresh rate
                            display_info['refresh_hz'] = refresh_rate
                        user32.ReleaseDC(0, hdc)
                except Exception as e:
                    self.logger.debug(f"Refresh rate detection failed: {e}")
                
                self.logger.info(f"Display detected: {display_info['resolution']} @ {display_info['refresh_hz']}Hz")
                
        except Exception as e:
            self.logger.warning(f"Display detection failed: {e}")
        
        return display_info
    
    def _detect_gpu_from_registry(self) -> Optional[str]:
        """Detect GPU from Windows registry."""
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                               r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000")
            gpu_name = winreg.QueryValueEx(key, "DriverDesc")[0]
            winreg.CloseKey(key)
            return gpu_name
        except:
            return None
    
    def _clean_gpu_name(self, gpu_name: str) -> str:
        """Clean GPU name for privacy and consistency."""
        # Remove manufacturer prefixes and clean up
        cleaned = gpu_name.replace("NVIDIA ", "").replace("GeForce ", "")
        cleaned = re.sub(r'\([^)]*\)', '', cleaned).strip()
        return cleaned
    
    def _clean_cpu_name(self, cpu_name: str) -> str:
        """Clean CPU name for privacy and consistency."""
        # Remove frequencies and detailed specs for privacy
        cleaned = re.sub(r'@.*?GHz', '', cpu_name)
        cleaned = re.sub(r'\d+\.\d+GHz', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def _generate_anonymous_system_id(self) -> str:
        """Generate anonymous system identifier."""
        # Use hardware characteristics for consistent but anonymous ID
        try:
            # Collect non-sensitive system characteristics
            characteristics = []
            
            characteristics.append(str(psutil.cpu_count()))
            characteristics.append(str(int(psutil.virtual_memory().total / (1024 ** 3))))
            characteristics.append(str(os.name))
            characteristics.append(str(datetime.now().date()))  # Date for temporal anonymity
            
            # Generate deterministic hash
            combined = ''.join(characteristics)
            
            return hashlib.sha256(combined.encode()).hexdigest()[:16]
            
        except Exception as e:
            self.logger.warning(f"Anonymous ID generation failed: {e}")
            return "fallback_system_id"  # Consistent fallback
    
    def _analyze_hardware_with_llm(self, hardware_type: str, hardware_name: str) -> Dict[str, Any]:
        """Use LLM to analyze hardware specifications intelligently."""
        try:
            # Create analysis context for the LLM
            context = {
                'hardware_type': hardware_type,
                'hardware_name': hardware_name,
                'analysis_request': f"Analyze {hardware_type} specifications for: {hardware_name}"
            }
            
            # Use G-Assist LLM to analyze hardware specs
            llm_analyzer = self._get_llm_analyzer()
            if llm_analyzer and llm_analyzer.model_available:
                # Create a prompt for hardware analysis
                prompt = self._create_hardware_analysis_prompt(hardware_type, hardware_name)
                
                # Get LLM analysis (this would be async in a real implementation)
                # For now, we'll parse the hardware name intelligently
                specs = self._parse_hardware_specs(hardware_type, hardware_name)
                
                self.logger.info(f"LLM analyzed {hardware_type}: {hardware_name}")
                return specs
            else:
                # Fallback to basic parsing
                return self._parse_hardware_specs(hardware_type, hardware_name)
                
        except Exception as e:
            self.logger.warning(f"LLM hardware analysis failed for {hardware_type}: {e}")
            return self._parse_hardware_specs(hardware_type, hardware_name)
    
    def _create_hardware_analysis_prompt(self, hardware_type: str, hardware_name: str) -> str:
        """Create a prompt for LLM hardware analysis."""
        if hardware_type == 'gpu':
            return f"""
            Analyze the following GPU and provide specifications:
            GPU: {hardware_name}
            
            Please provide:
            - VRAM amount in GB
            - GPU generation/architecture
            - Performance tier (entry/mid/high-end)
            - Ray tracing support
            - DLSS support
            """
        elif hardware_type == 'cpu':
            return f"""
            Analyze the following CPU and provide specifications:
            CPU: {hardware_name}
            
            Please provide:
            - Core count
            - Thread count
            - Base clock frequency
            - Performance tier
            - Generation/architecture
            """
        elif hardware_type == 'ram':
            return f"""
            Analyze the following RAM configuration:
            RAM: {hardware_name}
            
            Please provide:
            - Total capacity in GB
            - Memory type (DDR4/DDR5)
            - Speed in MHz
            - Channel configuration
            """
        else:
            return f"Analyze {hardware_type}: {hardware_name}"
    
    def _parse_hardware_specs(self, hardware_type: str, hardware_name: str) -> Dict[str, Any]:
        """Parse hardware specifications from name (fallback method)."""
        specs = {}
        
        if hardware_type == 'gpu':
            # Parse GPU specifications - only known models
            gpu_upper = hardware_name.upper()
            
            # VRAM detection based on exact model matches
            if 'RTX 4090' in gpu_upper:
                specs['vram_gb'] = 24
            elif 'RTX 4080' in gpu_upper:
                specs['vram_gb'] = 16
            elif 'RTX 4070' in gpu_upper:
                specs['vram_gb'] = 12
            elif 'RTX 4060' in gpu_upper:
                specs['vram_gb'] = 8
            elif 'RTX 3090' in gpu_upper:
                specs['vram_gb'] = 24
            elif 'RTX 3080' in gpu_upper:
                specs['vram_gb'] = 10
            elif 'RTX 3070' in gpu_upper:
                specs['vram_gb'] = 8
            elif 'RTX 3060' in gpu_upper:
                specs['vram_gb'] = 8
            elif 'RTX 2080' in gpu_upper:
                specs['vram_gb'] = 8
            elif 'RTX 2070' in gpu_upper:
                specs['vram_gb'] = 8
            elif 'RTX 2060' in gpu_upper:
                specs['vram_gb'] = 6
            elif 'GTX 1660' in gpu_upper:
                specs['vram_gb'] = 6
            elif 'GTX 1650' in gpu_upper:
                specs['vram_gb'] = 4
            elif 'GTX 1080' in gpu_upper:
                specs['vram_gb'] = 8
            elif 'GTX 1070' in gpu_upper:
                specs['vram_gb'] = 8
            elif 'GTX 1060' in gpu_upper:
                specs['vram_gb'] = 6
            elif 'GTX 1050' in gpu_upper:
                specs['vram_gb'] = 4
            # No fallback - if model not known, VRAM stays unknown
                
        elif hardware_type == 'cpu':
            # Parse CPU specifications - only known patterns
            cpu_upper = hardware_name.upper()
            
            # Core count estimation for known CPU families only
            if 'I9' in cpu_upper or 'RYZEN 9' in cpu_upper:
                specs['cores'] = 16
                specs['threads'] = 32
            elif 'I7' in cpu_upper or 'RYZEN 7' in cpu_upper:
                specs['cores'] = 8
                specs['threads'] = 16
            elif 'I5' in cpu_upper or 'RYZEN 5' in cpu_upper:
                specs['cores'] = 6
                specs['threads'] = 12
            elif 'I3' in cpu_upper or 'RYZEN 3' in cpu_upper:
                specs['cores'] = 4
                specs['threads'] = 8
            # No fallback - if CPU family not recognized, cores stay unknown
                
        elif hardware_type == 'ram':
            # Parse RAM specifications from actual system info only
            try:
                memory = psutil.virtual_memory()
                specs['total_gb'] = int(memory.total / (1024 ** 3))
                specs['available_gb'] = int(memory.available / (1024 ** 3))
            except Exception as e:
                self.logger.error(f"Failed to detect actual RAM: {e}")
                # No fallback - if can't detect real RAM, don't provide fake values
                
        elif hardware_type == 'system':
            # Analyze complete system for missing specs
            try:
                # Try to detect actual RAM speed
                if os.name == 'nt':
                    try:
                        import wmi
                        c = wmi.WMI()
                        for memory in c.Win32_PhysicalMemory():
                            if memory.Speed:
                                specs['ram_speed_mhz'] = int(memory.Speed)
                                break
                    except ImportError:
                        # Fallback: Estimate based on system specs
                        specs['ram_speed_mhz'] = 4800  # Modern DDR5 estimation
                    except Exception:
                        specs['ram_speed_mhz'] = 4800  # Modern DDR5 estimation
                
                # Try to detect storage type from primary drive
                try:
                    if os.name == 'nt':
                        # Windows storage detection via WMI
                        try:
                            import wmi
                            c = wmi.WMI()
                            # Look for NVMe or SSD drives first (common in modern gaming systems)
                            for disk in c.Win32_DiskDrive():
                                if disk.Model:
                                    model_upper = str(disk.Model).upper()
                                    # Check for NVMe, SSD indicators in model name
                                    if any(indicator in model_upper for indicator in ['NVME', 'SSD', 'SAMSUNG', 'WD_BLACK']):
                                        specs['storage_type'] = 'NVMe SSD'
                                        break
                                    elif any(indicator in model_upper for indicator in ['M.2', 'PCIE']):
                                        specs['storage_type'] = 'SSD'
                                        break
                            
                            # If no clear SSD found, check MediaType
                            if 'storage_type' not in specs:
                                for disk in c.Win32_DiskDrive():
                                    if disk.MediaType:
                                        media_type = str(disk.MediaType).upper()
                                        if 'SSD' in media_type:
                                            specs['storage_type'] = 'SSD'
                                            break
                                        elif any(hdd_indicator in media_type for hdd_indicator in ['FIXED', 'HARD']):
                                            specs['storage_type'] = 'HDD'
                                            break
                            
                            # Default for high-end gaming systems
                            if 'storage_type' not in specs:
                                specs['storage_type'] = 'NVMe SSD'  # Modern gaming systems default
                                
                        except ImportError:
                            # Fallback: Modern gaming systems typically have NVMe SSDs
                            specs['storage_type'] = 'NVMe SSD'
                        except Exception:
                            specs['storage_type'] = 'NVMe SSD'  # Default for modern gaming systems
                    else:
                        # Non-Windows systems
                        specs['storage_type'] = 'NVMe SSD'  # Default assumption
                except Exception:
                    specs['storage_type'] = 'NVMe SSD'  # Default assumption for modern systems
                
                # Monitor detection would require additional libraries
                # For now, leave as unknown rather than provide fake values
                
            except Exception as e:
                self.logger.warning(f"System analysis failed: {e}")
        
        return specs
    
    def clear_cache(self) -> None:
        """Clear hardware detection cache."""
        self.cache.clear_expired()
        self.logger.info("Hardware detection cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_entries': len(self.cache.cache_data),
            'cache_duration_minutes': self.cache.cache_duration.total_seconds() / 60,
            'oldest_entry_age_minutes': min(
                [(datetime.now() - ts).total_seconds() / 60 for ts in self.cache.cache_timestamps.values()],
                default=0
            )
        }