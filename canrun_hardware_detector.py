"""
CanRun Hardware Detector - Standalone module for NVIDIA G-Assist Plugin
Privacy-focused hardware detection for RTX/GTX gaming systems.
Flattened structure with all dependencies bundled for PyInstaller compatibility.
"""

import hashlib
import json
import logging
import os
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cpuinfo  # py-cpuinfo package
import psutil

from canrun_model_loader import match_gpu_key

# Platform detection
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"

# Set feature flags
PYNVML_AVAILABLE = False
WINREG_AVAILABLE = False
WMI_AVAILABLE = False
GPUTIL_AVAILABLE = False
CTYPES_AVAILABLE = IS_WINDOWS

# Platform-specific imports with proper fallbacks
if IS_WINDOWS:
    try:
        import ctypes
        CTYPES_AVAILABLE = True
    except ImportError:
        CTYPES_AVAILABLE = False

    try:
        import pynvml
        PYNVML_AVAILABLE = True
    except (ImportError, OSError):
        PYNVML_AVAILABLE = False

    try:
        import winreg
        WINREG_AVAILABLE = True
    except ImportError:
        WINREG_AVAILABLE = False

    try:
        import wmi
        WMI_AVAILABLE = True
    except (ImportError, OSError):
        WMI_AVAILABLE = False

# Handle GPUtil import with platform compatibility
if IS_WINDOWS:
    try:
        # Create distutils compatibility for Python 3.12+
        try:
            import distutils.spawn  # noqa: F401 - GPUtil imports this module by name
        except ImportError:
            import shutil

            class DistutilsSpawn:
                @staticmethod
                def find_executable(name):
                    return shutil.which(name)

            if 'distutils' not in sys.modules:
                import types
                distutils_module = types.ModuleType('distutils')
                distutils_module.spawn = DistutilsSpawn()
                sys.modules['distutils'] = distutils_module
                sys.modules['distutils.spawn'] = DistutilsSpawn()

        import GPUtil
        GPUTIL_AVAILABLE = True
    except (ImportError, OSError):
        GPUTIL_AVAILABLE = False


@dataclass
class HardwareSpecs:
    """Hardware specifications for RTX/GTX gaming systems."""

    # Essential Gaming Data (Required)
    gpu_model: str
    gpu_vram_gb: int
    cpu_cores: int
    cpu_threads: int
    ram_total_gb: int
    ram_speed_mhz: int
    storage_type: str
    primary_monitor_refresh_hz: int
    primary_monitor_resolution: str
    os_version: str
    directx_version: str

    # Fields with defaults
    gpu_vendor: str = "NVIDIA"
    cpu_model: str = "Unknown CPU"
    anonymous_system_id: str = ""
    data_timestamp: datetime | None = None
    is_nvidia_gpu: bool = True
    supports_rtx: bool | None = None
    supports_dlss: bool | None = None
    nvidia_driver_version: str = "Unknown"
    total_storage_gb: int = 0
    drives: list[dict[str, Any]] = None
    gpu_power_limit_w: int = 0

    def __post_init__(self):
        """Validate hardware specs after initialization."""
        if self.data_timestamp is None:
            self.data_timestamp = datetime.now()

        if not self.anonymous_system_id:
            self.anonymous_system_id = self._generate_anonymous_id()

        if self.drives is None:
            self.drives = []

        # Validate RTX/GTX GPU requirement
        if not (self.gpu_vendor.upper() == "NVIDIA" and
                ("RTX" in self.gpu_model.upper() or "GTX" in self.gpu_model.upper())):
            # Allow for detection failures but log warning
            logging.getLogger(__name__).warning("Non-NVIDIA RTX/GTX GPU detected")

        # Auto-compute RTX/DLSS support
        if self.supports_rtx is None:
            self.supports_rtx = "RTX" in self.gpu_model.upper()

        if self.supports_dlss is None:
            self.supports_dlss = self.supports_rtx

        # Validate specs
        assert self.gpu_vram_gb >= 0, "VRAM must be non-negative"
        assert self.cpu_cores > 0, "CPU cores must be greater than 0"
        assert self.ram_total_gb > 0, "RAM must be greater than 0"

    def _generate_anonymous_id(self) -> str:
        """Generate anonymous system identifier."""
        fingerprint = f"{self.gpu_model}_{self.cpu_cores}_{self.ram_total_gb}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
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
            'total_storage_gb': self.total_storage_gb,
            'drives': self.drives,
            'primary_monitor_refresh_hz': self.primary_monitor_refresh_hz,
            'primary_monitor_resolution': self.primary_monitor_resolution,
            'os_version': self.os_version,
            'directx_version': self.directx_version,
            'anonymous_system_id': self.anonymous_system_id,
            'data_timestamp': self.data_timestamp.isoformat() if self.data_timestamp else None,
            'is_nvidia_gpu': self.is_nvidia_gpu,
            'supports_rtx': self.supports_rtx,
            'supports_dlss': self.supports_dlss,
            'nvidia_driver_version': self.nvidia_driver_version,
            'gpu_power_limit_w': self.gpu_power_limit_w,
        }


# Highest graphics power NVIDIA allows for each laptop GPU, without Dynamic Boost.
LAPTOP_MAX_TGP_W = {
    "rtx 5090 laptop": 150, "rtx 5080 laptop": 150, "rtx 5070 ti laptop": 115, "rtx 5070 laptop": 100,
    "rtx 5060 laptop": 100, "rtx 5050 laptop": 100,
    "rtx 4090 laptop": 150, "rtx 4080 laptop": 150, "rtx 4070 laptop": 115, "rtx 4060 laptop": 115,
    "rtx 4050 laptop": 115,
    "rtx 3080 ti laptop": 150, "rtx 3080 laptop": 150, "rtx 3070 ti laptop": 125, "rtx 3070 laptop": 125,
    "rtx 3060 laptop": 115, "rtx 3050 ti laptop": 80, "rtx 3050 laptop": 80,
}
LOW_POWER_SHARE = 0.7


def laptop_power_note(gpu_model: str, power_limit_w: int) -> str:
    """Warning for a laptop GPU set well below the power its model can run at, else ""."""
    if not power_limit_w or power_limit_w <= 0:
        return ""
    key = match_gpu_key(gpu_model, {name: {} for name in LAPTOP_MAX_TGP_W})
    if key is None or "laptop" not in (gpu_model or "").lower():
        return ""
    max_w = LAPTOP_MAX_TGP_W[key]
    if power_limit_w >= max_w * LOW_POWER_SHARE:
        return ""
    return f"Power limit {power_limit_w} W of {max_w} W: expect lower FPS"


class CanRunHardwareDetector:
    """Hardware detector for RTX/GTX gaming systems."""

    def __init__(self, cache_duration_hours: int = 24):  # noqa: ARG002
        # Callers may pass cache_duration_hours. Specs stay cached for 15 minutes.
        self.logger = logging.getLogger(__name__)
        self.cache_duration = timedelta(minutes=15)
        self.cache_data = {}
        self.cache_timestamps = {}

        # Load hardware databases
        self.gpu_database = self._load_gpu_database()
        self.cpu_database = self._load_cpu_database()

        # Initialize NVIDIA libraries
        self._initialize_nvidia_libraries()

        self.logger.info("CanRun hardware detector initialized for RTX/GTX systems")

    def _initialize_nvidia_libraries(self) -> None:
        """Initialize NVIDIA-specific libraries."""
        if IS_WINDOWS and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.logger.info("NVIDIA ML library initialized")
            except Exception as e:
                self.logger.warning('NVIDIA ML library initialization failed: %s', e)
        else:
            self.logger.debug("NVIDIA ML library not available on this platform")

    def has_nvidia_gpu(self) -> bool:
        """Check if NVIDIA RTX/GTX GPU is available."""
        # For Windows - try multiple detection methods
        if IS_WINDOWS:
            if PYNVML_AVAILABLE:
                try:
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        gpu_name = pynvml.nvmlDeviceGetName(handle)
                        if isinstance(gpu_name, bytes):
                            gpu_name = gpu_name.decode('utf-8')
                        return 'RTX' in gpu_name.upper() or 'GTX' in gpu_name.upper()
                except Exception:
                    pass

            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        if 'NVIDIA' in gpu.name.upper():
                            gpu_name = gpu.name.upper()
                            return 'RTX' in gpu_name or 'GTX' in gpu_name
                except Exception:
                    pass

            if WINREG_AVAILABLE:
                try:
                    gpu_name = self._detect_gpu_from_registry()
                    if gpu_name:
                        gpu_upper = gpu_name.upper()
                        return 'NVIDIA' in gpu_upper and ('RTX' in gpu_upper or 'GTX' in gpu_upper)
                except Exception:
                    pass

        # For Linux - try lspci command
        elif IS_LINUX:
            try:
                import subprocess
                output = subprocess.check_output(['lspci', '-vnn'], text=True, timeout=5)
                nvidia_lines = [line for line in output.split('\n') if 'NVIDIA' in line]
                for line in nvidia_lines:
                    line_upper = line.upper()
                    if 'RTX' in line_upper or 'GTX' in line_upper:
                        return True
            except Exception:
                pass

        # For macOS - NVIDIA GPUs are less common
        elif IS_MACOS:
            try:
                import subprocess
                output = subprocess.check_output(['system_profiler', 'SPDisplaysDataType'], text=True, timeout=5)
                if 'NVIDIA' in output:
                    output_upper = output.upper()
                    return 'RTX' in output_upper or 'GTX' in output_upper
            except Exception:
                pass

        return False

    async def get_hardware_specs(self) -> HardwareSpecs:
        """Get hardware specifications."""
        # Check cache first
        cache_key = "hardware_specs"
        if cache_key in self.cache_data:
            timestamp = self.cache_timestamps[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                self.logger.debug("Returning cached hardware specs")
                return self.cache_data[cache_key]

        # Detect hardware
        specs = self._detect_hardware_safely()

        # Cache the result
        self.cache_data[cache_key] = specs
        self.cache_timestamps[cache_key] = datetime.now()

        return specs

    def _detect_hardware_safely(self) -> HardwareSpecs:
        """Safely detect hardware with comprehensive error handling."""
        # Detect GPU
        gpu_info = self._detect_nvidia_gpu()

        # Detect CPU
        cpu_info = self._detect_cpu()

        # Detect RAM
        ram_info = self._detect_ram()
        if ram_info is None:
            self.logger.warning("RAM detection failed - using default values")
            ram_info = {'total_gb': 16, 'speed_mhz': 3200}

        # Detect OS
        os_info = self._detect_os()

        # Detect display information
        display_info = self._detect_display()

        # Detect storage
        storage_info = self._detect_storage()

        # Generate anonymous system ID
        anonymous_id = self._generate_anonymous_system_id()

        # Create hardware specifications
        specs = HardwareSpecs(
            gpu_model=gpu_info['name'],
            gpu_vram_gb=gpu_info['vram_gb'],
            gpu_vendor="NVIDIA",
            cpu_model=cpu_info['name'],
            cpu_cores=cpu_info['cores'],
            cpu_threads=cpu_info['threads'],
            ram_total_gb=ram_info['total_gb'],
            ram_speed_mhz=ram_info.get('speed_mhz', 3200),
            storage_type=storage_info.get('type', 'NVMe SSD'),
            primary_monitor_refresh_hz=display_info.get('refresh_hz', 60),
            primary_monitor_resolution=display_info.get('resolution', '1920x1080'),
            os_version=os_info['name'],
            directx_version=os_info['directx_version'],
            anonymous_system_id=anonymous_id,
            data_timestamp=datetime.now(),
            is_nvidia_gpu=gpu_info['is_nvidia'],
            supports_rtx=gpu_info['supports_rtx'],
            supports_dlss=gpu_info['supports_dlss'],
            nvidia_driver_version=gpu_info['driver_version'],
            total_storage_gb=storage_info.get('total_gb', 1000),
            drives=storage_info.get('drives', []),
            gpu_power_limit_w=int(gpu_info.get('power_limit_w') or 0),
        )

        self.logger.info('Hardware detected: %s, %s, %sGB RAM', specs.gpu_model, specs.cpu_model, specs.ram_total_gb)
        return specs

    @staticmethod
    def _nvml_power_limit_w(handle) -> int:
        """Board power limit in watts from NVML; 0 when the driver does not report one."""
        for getter in ("nvmlDeviceGetEnforcedPowerLimit", "nvmlDeviceGetPowerManagementLimit"):
            try:
                milliwatts = getattr(pynvml, getter)(handle)
            except Exception:
                continue
            if isinstance(milliwatts, (int, float)) and 5_000 <= milliwatts <= 1_000_000:
                return round(milliwatts / 1000)
        return 0

    def _detect_nvidia_gpu(self) -> dict[str, Any]:
        """Detect NVIDIA GPU information."""
        gpu_info = {
            'name': 'Unknown GPU',
            'vram_gb': 4,
            'is_nvidia': False,
            'supports_rtx': False,
            'supports_dlss': False,
            'driver_version': 'Unknown'
        }

        # Try NVIDIA ML library first
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                    gpu_name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(gpu_name, bytes):
                        gpu_name = gpu_name.decode('utf-8')

                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    raw_vram_gb = mem_info.total // (1024 ** 3)
                    rounded_vram_gb = self._round_up_to_standard_vram_capacity(raw_vram_gb)

                    try:
                        driver_version = pynvml.nvmlSystemGetDriverVersion()
                        if isinstance(driver_version, bytes):
                            driver_version = driver_version.decode('utf-8')
                    except Exception:
                        driver_version = "Unknown"

                    gpu_info.update({
                        'name': self._clean_gpu_name(gpu_name),
                        'vram_gb': rounded_vram_gb,
                        'is_nvidia': True,
                        'supports_rtx': 'RTX' in gpu_name.upper(),
                        'supports_dlss': 'RTX' in gpu_name.upper(),
                        'driver_version': driver_version,
                        'power_limit_w': self._nvml_power_limit_w(handle),
                    })

                    self.logger.info('NVIDIA GPU detected via pynvml: %s (%sGB -> %sGB VRAM)', gpu_info['name'], raw_vram_gb, rounded_vram_gb)
                    return gpu_info
            except Exception as e:
                self.logger.warning('NVIDIA ML detection failed: %s', e)

        # Fallback to GPUtil if available
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_name = gpu.name

                    if 'NVIDIA' in gpu_name.upper():
                        gpu_info.update({
                            'name': self._clean_gpu_name(gpu_name),
                            'vram_gb': int(gpu.memoryTotal / 1024),
                            'is_nvidia': True,
                            'supports_rtx': 'RTX' in gpu_name.upper(),
                            'supports_dlss': 'RTX' in gpu_name.upper(),
                            'driver_version': 'Unknown'
                        })

                        self.logger.info('NVIDIA GPU detected via GPUtil: %s', gpu_info['name'])
                        return gpu_info
            except Exception as e:
                self.logger.warning('GPUtil detection failed: %s', e)

        # Windows Registry fallback
        if IS_WINDOWS and WINREG_AVAILABLE:
            try:
                gpu_name = self._detect_gpu_from_registry()
                if gpu_name and 'NVIDIA' in gpu_name.upper():
                    vram_gb = self._estimate_vram_from_gpu_name(gpu_name)

                    gpu_info.update({
                        'name': self._clean_gpu_name(gpu_name),
                        'vram_gb': vram_gb,
                        'is_nvidia': True,
                        'supports_rtx': 'RTX' in gpu_name.upper(),
                        'supports_dlss': 'RTX' in gpu_name.upper(),
                        'driver_version': 'Unknown'
                    })

                    self.logger.info('NVIDIA GPU detected via registry: %s', gpu_info['name'])
                    return gpu_info
            except Exception as e:
                self.logger.warning('Registry GPU detection failed: %s', e)

        smi_gpu = self._detect_nvidia_gpu_from_smi()
        if smi_gpu:
            return smi_gpu

        # If no NVIDIA GPU found, return basic info
        self.logger.warning("NVIDIA GPU not detected - using fallback")
        return gpu_info

    def _detect_nvidia_gpu_from_smi(self) -> dict[str, Any] | None:
        """Read the GPU from nvidia-smi when the NVML library file cannot be loaded."""
        import subprocess
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if completed.returncode != 0 or not completed.stdout.strip():
                return None
            name, memory, driver = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
            memory_value = int(memory.split()[0])
            raw_vram_gb = memory_value // 1024 if "MiB" in memory else memory_value
            rounded_vram_gb = self._round_up_to_standard_vram_capacity(raw_vram_gb)
            self.logger.info('NVIDIA GPU detected via nvidia-smi: %s', name)
            return {
                'name': self._clean_gpu_name(name),
                'vram_gb': rounded_vram_gb,
                'is_nvidia': True,
                'supports_rtx': 'RTX' in name.upper(),
                'supports_dlss': 'RTX' in name.upper(),
                'driver_version': driver or 'Unknown'
            }
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.logger.warning('nvidia-smi detection failed: %s', exc)
            return None

    def _detect_cpu(self) -> dict[str, Any]:
        """Detect CPU information."""
        cpu_info = {
            'name': 'Unknown CPU',
            'cores': 4,
            'threads': 8
        }

        # Try cpuinfo library
        try:
            cpu_data = cpuinfo.get_cpu_info()

            cpu_info.update({
                'name': self._clean_cpu_name(cpu_data.get('brand_raw', 'Unknown CPU')),
                'cores': cpu_data.get('count', 4),
                'threads': cpu_data.get('count', 4)
            })

            # Try to get actual thread count from psutil
            logical_cores = psutil.cpu_count(logical=True)
            physical_cores = psutil.cpu_count(logical=False)
            if logical_cores and physical_cores:
                cpu_info['cores'] = physical_cores
                cpu_info['threads'] = logical_cores

            self.logger.info('CPU detected: %s', cpu_info['name'])
            return cpu_info
        except Exception as e:
            self.logger.warning('CPU detection failed: %s', e)

        # Fallback to psutil
        try:
            logical_cores = psutil.cpu_count(logical=True)
            physical_cores = psutil.cpu_count(logical=False)

            cpu_info.update({
                'name': 'Unknown CPU',
                'cores': physical_cores or 4,
                'threads': logical_cores or 8
            })

            return cpu_info
        except Exception as e:
            self.logger.warning('psutil CPU detection failed: %s', e)

        return cpu_info

    def _detect_ram(self) -> dict[str, Any] | None:
        """Detect RAM information."""
        try:
            memory = psutil.virtual_memory()

            # Detect raw RAM and round up to standard capacity
            raw_ram_gb = round(memory.total / (1024 ** 3))
            rounded_ram_gb = self._round_up_to_standard_ram_capacity(raw_ram_gb)

            ram_info = {
                'total_gb': rounded_ram_gb,
                'available_gb': round(memory.available / (1024 ** 3)),
                'speed_mhz': 3200  # Default modern RAM speed
            }

            # Try to get actual RAM speed on Windows
            if IS_WINDOWS and WMI_AVAILABLE:
                try:
                    c = wmi.WMI()
                    for memory_module in c.Win32_PhysicalMemory():
                        if memory_module.Speed:
                            ram_info['speed_mhz'] = int(memory_module.Speed)
                            break
                except Exception:
                    pass

            self.logger.info('RAM detected: %sGB -> %sGB @ %sMHz', raw_ram_gb, rounded_ram_gb, ram_info['speed_mhz'])
            return ram_info
        except Exception as e:
            self.logger.warning('RAM detection failed: %s', e)
            return None

    def _detect_os(self) -> dict[str, Any]:
        """Detect OS information."""
        os_info = {
            'name': 'Unknown OS',
            'directx_version': 'N/A'
        }

        try:
            if IS_WINDOWS:
                os_name = f"Windows {platform.release()}"

                if WINREG_AVAILABLE:
                    try:
                        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                           r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")

                        current_build = winreg.QueryValueEx(key, "CurrentBuild")[0]

                        if int(current_build) >= 22000:
                            os_name = "Windows 11"
                        elif int(current_build) >= 10240:
                            os_name = "Windows 10"

                        winreg.CloseKey(key)
                    except Exception:
                        pass

                os_info.update({
                    'name': os_name,
                    'directx_version': 'DirectX 12'
                })
            elif IS_LINUX:
                os_info.update({
                    'name': f"Linux {platform.release()}",
                    'directx_version': 'Vulkan/OpenGL'
                })
            elif IS_MACOS:
                os_info.update({
                    'name': f"macOS {platform.mac_ver()[0]}",
                    'directx_version': 'Metal'
                })
        except Exception as e:
            self.logger.warning('OS detection failed: %s', e)

        return os_info

    def _detect_display(self) -> dict[str, Any]:
        """Detect display information."""
        display_info = {
            'resolution': '1920x1080',
            'refresh_hz': 60
        }

        try:
            if IS_WINDOWS and CTYPES_AVAILABLE:
                user32 = ctypes.windll.user32
                screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
                display_info['resolution'] = f"{screensize[0]}x{screensize[1]}"

                try:
                    gdi32 = ctypes.windll.gdi32
                    hdc = user32.GetDC(0)
                    if hdc:
                        refresh_rate = gdi32.GetDeviceCaps(hdc, 116)
                        if refresh_rate > 1:
                            display_info['refresh_hz'] = refresh_rate
                        user32.ReleaseDC(0, hdc)
                except Exception:
                    pass
        except Exception as e:
            self.logger.debug('Display detection failed: %s', e)

        return display_info

    def _detect_storage(self) -> dict[str, Any]:
        """Detect storage information."""
        storage_info = {
            'type': 'NVMe SSD',
            'total_gb': 1000,
            'drives': []
        }

        try:
            if IS_WINDOWS and WMI_AVAILABLE:
                c = wmi.WMI()
                drives = []
                total_storage_gb = 0

                for disk in c.Win32_DiskDrive():
                    if disk.Model:
                        drive_info = {
                            'model': disk.Model,
                            'type': 'Unknown',
                            'size_gb': 0
                        }

                        model_upper = str(disk.Model).upper()
                        if any(indicator in model_upper for indicator in ['NVME', 'SSD']):
                            drive_info['type'] = 'NVMe SSD'
                        elif 'SSD' in model_upper:
                            drive_info['type'] = 'SSD'
                        else:
                            drive_info['type'] = 'HDD'

                        if disk.Size:
                            try:
                                size_gb = int(int(disk.Size) / (1024**3))
                                drive_info['size_gb'] = size_gb
                                total_storage_gb += size_gb
                            except Exception:
                                pass

                        drives.append(drive_info)

                if drives:
                    storage_info['drives'] = drives
                    storage_info['total_gb'] = total_storage_gb

                    # Set primary storage type
                    if any(drive['type'] == 'NVMe SSD' for drive in drives):
                        storage_info['type'] = 'NVMe SSD'
                    elif any(drive['type'] == 'SSD' for drive in drives):
                        storage_info['type'] = 'SSD'
                    else:
                        storage_info['type'] = 'HDD'
        except Exception as e:
            self.logger.debug('Storage detection failed: %s', e)

        return storage_info

    def _detect_gpu_from_registry(self) -> str | None:
        """Detect GPU from Windows registry."""
        if not (IS_WINDOWS and WINREG_AVAILABLE):
            return None

        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                              r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000")
            gpu_name = winreg.QueryValueEx(key, "DriverDesc")[0]
            winreg.CloseKey(key)
            return gpu_name
        except Exception:
            return None

    def _clean_gpu_name(self, gpu_name: str) -> str:
        """Clean GPU name for consistency."""
        cleaned = gpu_name.replace("NVIDIA ", "").replace("GeForce ", "")
        return re.sub(r'\([^)]*\)', '', cleaned).strip()

    def _clean_cpu_name(self, cpu_name: str) -> str:
        """Clean CPU name for consistency."""
        cleaned = re.sub(r'@.*?GHz', '', cpu_name)
        cleaned = re.sub(r'\d+\.\d+GHz', '', cleaned)
        return re.sub(r'\s+', ' ', cleaned).strip()

    def _load_gpu_database(self) -> dict[str, Any]:
        """Load GPU database from JSON file - data-driven approach."""
        try:
            # Try multiple possible locations for the JSON file
            possible_paths = [
                "data/gpu_hierarchy.json",
                "canrun/data/gpu_hierarchy.json",
                "gpu_hierarchy.json"
            ]

            for json_path in possible_paths:
                if os.path.exists(json_path):
                    with open(json_path, encoding='utf-8') as f:
                        data = json.load(f)
                        nvidia_data = data.get('nvidia', {})
                        self.logger.info('Loaded GPU database from %s with %s entries', json_path, len(nvidia_data))
                        return nvidia_data

            self.logger.warning("GPU database JSON file not found")
            return {}
        except Exception as e:
            self.logger.error('Failed to load GPU database: %s', e)
            return {}

    def _load_cpu_database(self) -> dict[str, Any]:
        """Load CPU database from JSON file - data-driven approach."""
        try:
            # Try multiple possible locations for the JSON file
            possible_paths = [
                "data/cpu_benchmarks.json",
                "canrun/data/cpu_benchmarks.json",
                "cpu_benchmarks.json"
            ]

            for json_path in possible_paths:
                if os.path.exists(json_path):
                    with open(json_path, encoding='utf-8') as f:
                        data = json.load(f)
                        # Flatten CPU patterns into single lookup
                        cpu_lookup = {}
                        if 'cpu_patterns' in data:
                            for _group, patterns in data['cpu_patterns'].items():
                                cpu_lookup.update(patterns)
                        self.logger.info('Loaded CPU database from %s with %s patterns', json_path, len(cpu_lookup))
                        return cpu_lookup

            self.logger.warning("CPU database JSON file not found")
            return {}
        except Exception as e:
            self.logger.error('Failed to load CPU database: %s', e)
            return {}

    def _estimate_vram_from_gpu_name(self, gpu_name: str) -> int:
        """Estimate VRAM from GPU model name using only JSON database."""
        gpu_lower = gpu_name.lower()

        # Search in loaded GPU database
        if self.gpu_database:
            gpu_key = match_gpu_key(gpu_lower, self.gpu_database)
            if gpu_key is not None:
                gpu_info = self.gpu_database[gpu_key]
                vram = gpu_info.get('vram_gb', gpu_info.get('memory'))
                if vram is not None:
                    self.logger.debug('Found VRAM for %s: %sGB from database', gpu_name, vram)
                    return vram

        # If not found in database, cannot determine VRAM accurately
        self.logger.warning('VRAM data not found for GPU: %s', gpu_name)
        return 4  # Minimal fallback only

    def _round_up_to_standard_ram_capacity(self, detected_gb: int) -> int:
        """Round up RAM to standard capacities (8, 16, 32, 64, 128GB)."""
        standard_capacities = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]

        for capacity in standard_capacities:
            # If detected is within 15% of standard capacity, round up to it
            if detected_gb <= capacity and detected_gb >= capacity * 0.85:
                return capacity

        # If no standard capacity fits, return detected value
        return detected_gb

    def _round_up_to_standard_vram_capacity(self, detected_gb: int) -> int:
        """Round up VRAM to standard capacities (4, 6, 8, 12, 16, 24GB)."""
        standard_capacities = [3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48]

        for capacity in standard_capacities:
            # If detected is within 15% of standard capacity, round up to it
            if detected_gb <= capacity and detected_gb >= capacity * 0.85:
                return capacity

        # If no standard capacity fits, return detected value
        return detected_gb

    def _generate_anonymous_system_id(self) -> str:
        """Generate anonymous system identifier."""
        try:
            characteristics = [
                str(psutil.cpu_count()),
                str(int(psutil.virtual_memory().total / (1024 ** 3))),
                str(os.name),
                str(datetime.now().date())
            ]

            combined = ''.join(characteristics)
            return hashlib.sha256(combined.encode()).hexdigest()[:16]
        except Exception as e:
            self.logger.warning('Anonymous ID generation failed: %s', e)
            return "fallback_system_id"

    def clear_cache(self) -> None:
        """Clear hardware detection cache."""
        self.cache_data.clear()
        self.cache_timestamps.clear()
        self.logger.info("Hardware detection cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_entries': len(self.cache_data),
            'cache_duration_minutes': self.cache_duration.total_seconds() / 60,
            'oldest_entry_age_minutes': min(
                [(datetime.now() - ts).total_seconds() / 60 for ts in self.cache_timestamps.values()],
                default=0
            )
        }

    def detect_primary_display(self) -> dict[str, Any]:
        """Detect primary display information."""
        display_info = {
            'width': 1920,
            'height': 1080,
            'method': 'fallback',
            'device': 'Unknown',
            'is_primary': True
        }

        try:
            if IS_WINDOWS and CTYPES_AVAILABLE:
                user32 = ctypes.windll.user32
                width = user32.GetSystemMetrics(0)
                height = user32.GetSystemMetrics(1)

                display_info.update({
                    'width': width,
                    'height': height,
                    'method': 'windows_api',
                    'device': 'Primary Monitor',
                    'is_primary': True
                })

                self.logger.info('Primary display detected: %sx%s', width, height)
        except Exception as e:
            self.logger.debug('Primary display detection failed: %s', e)

        return display_info

    def get_resolution_tier(self, width: int, height: int) -> str:
        """Get resolution tier classification."""
        total_pixels = width * height

        if total_pixels >= 3840 * 2160 * 0.9:
            return "4K"
        if total_pixels >= 2560 * 1440 * 0.9:
            return "1440p"
        if total_pixels >= 1920 * 1080 * 0.9:
            return "1080p"
        return "720p"

    def get_all_monitors(self) -> list[dict[str, Any]]:
        """Get information about all monitors."""
        monitors = []

        try:
            if IS_WINDOWS and CTYPES_AVAILABLE:
                # For now, just return primary monitor
                # TODO: Implement multi-monitor detection
                primary = self.detect_primary_display()
                primary.update({
                    'x': 0,
                    'y': 0
                })
                monitors.append(primary)
            else:
                # Fallback for non-Windows
                monitors.append({
                    'width': 1920,
                    'height': 1080,
                    'method': 'fallback',
                    'device': 'Default Monitor',
                    'is_primary': True,
                    'x': 0,
                    'y': 0
                })
        except Exception as e:
            self.logger.debug('Monitor detection failed: %s', e)
            monitors.append({
                'width': 1920,
                'height': 1080,
                'method': 'fallback',
                'device': 'Default Monitor',
                'is_primary': True,
                'x': 0,
                'y': 0
            })

        return monitors

    def get_display_resolution(self) -> dict[str, Any]:
        """Get display resolution with override support."""
        # Check environment variables first
        env_width = os.environ.get('CANRUN_DISPLAY_WIDTH')
        env_height = os.environ.get('CANRUN_DISPLAY_HEIGHT')

        if env_width and env_height:
            try:
                return {
                    'width': int(env_width),
                    'height': int(env_height),
                    'method': 'environment_override'
                }
            except ValueError:
                pass

        # Check config file override
        try:
            config_path = Path(__file__).parent / 'config.json'
            if config_path.exists():
                with config_path.open(encoding="utf-8") as f:
                    config = json.load(f)

                display_override = config.get('display_override', {})
                if display_override.get('enabled', False):
                    width = display_override.get('width')
                    height = display_override.get('height')
                    if width and height:
                        return {
                            'width': width,
                            'height': height,
                            'method': 'config_override'
                        }
        except Exception as e:
            self.logger.debug('Config file read failed: %s', e)

        # Fall back to detection
        return self.detect_primary_display()
