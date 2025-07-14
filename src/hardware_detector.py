"""
Hardware Detection Module for CanRun
Detects system hardware specifications including GPU, CPU, RAM, and DirectX version.
"""

import os
import sys
import json
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from pathlib import Path

# Import libraries with fallback handling
try:
    import psutil
except ImportError:
    psutil = None

try:
    import GPUtil
except ImportError:
    GPUtil = None

try:
    import cpuinfo
except ImportError:
    cpuinfo = None

try:
    import pynvml
except ImportError:
    pynvml = None

try:
    import winreg
except ImportError:
    winreg = None


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable
        base_path = sys._MEIPASS
    else:
        # Running as normal Python script
        base_path = Path(__file__).parent.parent
    return os.path.join(base_path, relative_path)


@dataclass
class HardwareSpecs:
    """Data class for storing hardware specifications."""
    gpu_name: str
    gpu_memory: int  # MB
    gpu_vendor: str
    cpu_name: str
    cpu_cores: int
    cpu_freq: float  # GHz
    ram_total: int  # MB
    ram_available: int  # MB
    directx_version: str
    os_version: str


class HardwareDetector:
    """Detects and analyzes system hardware specifications."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gpu_hierarchy = self._load_gpu_hierarchy()
    
    def _load_gpu_hierarchy(self) -> Dict:
        """Load GPU hierarchy data for performance analysis."""
        try:
            data_path = get_resource_path("data/gpu_hierarchy.json")
            with open(data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load GPU hierarchy: {e}")
            return {}
    
    def detect_hardware(self) -> HardwareSpecs:
        """Detect all hardware specifications."""
        gpu_info = self._detect_gpu()
        cpu_info = self._detect_cpu()
        ram_info = self._detect_ram()
        directx_version = self._detect_directx()
        os_version = self._detect_os()
        
        return HardwareSpecs(
            gpu_name=gpu_info.get('name', 'Unknown GPU'),
            gpu_memory=gpu_info.get('memory', 0),
            gpu_vendor=gpu_info.get('vendor', 'Unknown'),
            cpu_name=cpu_info.get('name', 'Unknown CPU'),
            cpu_cores=cpu_info.get('cores', 0),
            cpu_freq=cpu_info.get('freq', 0.0),
            ram_total=ram_info.get('total', 0),
            ram_available=ram_info.get('available', 0),
            directx_version=directx_version,
            os_version=os_version
        )
    
    def _detect_gpu(self) -> Dict:
        """Detect GPU information using multiple methods."""
        gpu_info = {'name': 'Unknown GPU', 'memory': 0, 'vendor': 'Unknown'}
        
        # Method 1: Try NVIDIA ML API
        try:
            if pynvml:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_info = {
                        'name': name,
                        'memory': memory_info.total // (1024 * 1024),  # Convert to MB
                        'vendor': 'NVIDIA'
                    }
                    pynvml.nvmlShutdown()
                    return gpu_info
        except Exception as e:
            self.logger.debug(f"NVIDIA ML detection failed: {e}")
        
        # Method 2: Try GPUtil
        try:
            if GPUtil:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    vendor = 'NVIDIA' if 'nvidia' in gpu.name.lower() else 'AMD' if 'amd' in gpu.name.lower() else 'Unknown'
                    gpu_info = {
                        'name': gpu.name,
                        'memory': int(gpu.memoryTotal),
                        'vendor': vendor
                    }
                    return gpu_info
        except Exception as e:
            self.logger.debug(f"GPUtil detection failed: {e}")
        
        # Method 3: Try WMI on Windows
        try:
            if sys.platform.startswith('win') and winreg:
                import wmi
                c = wmi.WMI()
                for gpu in c.Win32_VideoController():
                    if gpu.Name:
                        memory_mb = 0
                        if gpu.AdapterRAM:
                            memory_mb = gpu.AdapterRAM // (1024 * 1024)
                        
                        vendor = 'Unknown'
                        if 'nvidia' in gpu.Name.lower():
                            vendor = 'NVIDIA'
                        elif 'amd' in gpu.Name.lower() or 'radeon' in gpu.Name.lower():
                            vendor = 'AMD'
                        elif 'intel' in gpu.Name.lower():
                            vendor = 'Intel'
                        
                        gpu_info = {
                            'name': gpu.Name,
                            'memory': memory_mb,
                            'vendor': vendor
                        }
                        break
        except Exception as e:
            self.logger.debug(f"WMI detection failed: {e}")
        
        return gpu_info
    
    def _detect_cpu(self) -> Dict:
        """Detect CPU information."""
        cpu_info = {'name': 'Unknown CPU', 'cores': 0, 'freq': 0.0}
        
        # Method 1: Try cpuinfo
        try:
            if cpuinfo:
                info = cpuinfo.get_cpu_info()
                cpu_info = {
                    'name': info.get('brand_raw', 'Unknown CPU'),
                    'cores': info.get('count', 0),
                    'freq': info.get('hz_advertised_friendly', '0.0 GHz').replace(' GHz', '')
                }
                # Convert frequency to float
                try:
                    cpu_info['freq'] = float(cpu_info['freq'])
                except:
                    cpu_info['freq'] = 0.0
        except Exception as e:
            self.logger.debug(f"cpuinfo detection failed: {e}")
        
        # Method 2: Try psutil
        try:
            if psutil:
                cpu_info['cores'] = psutil.cpu_count(logical=False) or psutil.cpu_count()
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    cpu_info['freq'] = cpu_freq.current / 1000  # Convert MHz to GHz
        except Exception as e:
            self.logger.debug(f"psutil CPU detection failed: {e}")
        
        return cpu_info
    
    def _detect_ram(self) -> Dict:
        """Detect RAM information."""
        ram_info = {'total': 0, 'available': 0}
        
        try:
            if psutil:
                memory = psutil.virtual_memory()
                ram_info = {
                    'total': memory.total // (1024 * 1024),  # Convert to MB
                    'available': memory.available // (1024 * 1024)
                }
        except Exception as e:
            self.logger.debug(f"RAM detection failed: {e}")
        
        return ram_info
    
    def _detect_directx(self) -> str:
        """Detect DirectX version on Windows."""
        try:
            if sys.platform.startswith('win') and winreg:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                   r"SOFTWARE\Microsoft\DirectX") as key:
                    version, _ = winreg.QueryValueEx(key, "Version")
                    return f"DirectX {version}"
        except Exception as e:
            self.logger.debug(f"DirectX detection failed: {e}")
        
        return "DirectX 12"  # Default assumption for modern systems
    
    def _detect_os(self) -> str:
        """Detect operating system version."""
        try:
            if sys.platform.startswith('win'):
                import platform
                return f"Windows {platform.version()}"
            else:
                return sys.platform
        except Exception as e:
            self.logger.debug(f"OS detection failed: {e}")
            return "Unknown OS"
    
    def get_gpu_performance_score(self, gpu_name: str) -> Optional[int]:
        """Get performance score for a GPU from hierarchy data."""
        if not self.gpu_hierarchy:
            return None
        
        # Clean and normalize GPU name
        gpu_clean = gpu_name.replace('NVIDIA GeForce ', '').replace('AMD Radeon ', '')
        
        # Check NVIDIA GPUs
        nvidia_gpus = self.gpu_hierarchy.get('nvidia', {})
        for gpu_key, gpu_data in nvidia_gpus.items():
            if gpu_key.lower() in gpu_clean.lower():
                return gpu_data.get('score', 0)
        
        # Check AMD GPUs
        amd_gpus = self.gpu_hierarchy.get('amd', {})
        for gpu_key, gpu_data in amd_gpus.items():
            if gpu_key.lower() in gpu_clean.lower():
                return gpu_data.get('score', 0)
        
        return None
    
    def get_gpu_tier(self, gpu_name: str) -> Optional[str]:
        """Get performance tier for a GPU."""
        if not self.gpu_hierarchy:
            return None
        
        gpu_clean = gpu_name.replace('NVIDIA GeForce ', '').replace('AMD Radeon ', '')
        
        # Check all GPU categories
        for category in ['nvidia', 'amd']:
            gpus = self.gpu_hierarchy.get(category, {})
            for gpu_key, gpu_data in gpus.items():
                if gpu_key.lower() in gpu_clean.lower():
                    return gpu_data.get('tier', 'Unknown')
        
        return None


def main():
    """Test the hardware detector."""
    detector = HardwareDetector()
    specs = detector.detect_hardware()
    
    print("Hardware Detection Results:")
    print(f"GPU: {specs.gpu_name} ({specs.gpu_memory} MB)")
    print(f"CPU: {specs.cpu_name} ({specs.cpu_cores} cores, {specs.cpu_freq:.2f} GHz)")
    print(f"RAM: {specs.ram_total} MB ({specs.ram_available} MB available)")
    print(f"DirectX: {specs.directx_version}")
    print(f"OS: {specs.os_version}")
    
    # Test GPU performance lookup
    gpu_score = detector.get_gpu_performance_score(specs.gpu_name)
    gpu_tier = detector.get_gpu_tier(specs.gpu_name)
    if gpu_score:
        print(f"GPU Performance Score: {gpu_score}")
    if gpu_tier:
        print(f"GPU Tier: {gpu_tier}")


if __name__ == "__main__":
    main()