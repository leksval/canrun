"""
Display Detection Module for CanRun
Simple, accurate display resolution detection using proper Windows APIs.
"""

import sys
import os
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path


class DisplayDetector:
    """Simple, accurate display detection using proper Windows APIs"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def get_display_resolution(self) -> Dict:
        """Get display resolution with emergency override support"""
        
        # Check for emergency environment variable override
        # This is only for edge cases where detection fails
        override = self._check_emergency_override()
        if override:
            return override
        
        # Otherwise use automatic detection
        return self.detect_primary_display()
    
    def detect_primary_display(self) -> Dict:
        """Get the primary display resolution accurately"""
        
        if sys.platform == 'win32':
            return self._detect_windows()
        else:
            return self._detect_fallback()
    
    def _check_emergency_override(self) -> Optional[Dict]:
        """Check for emergency environment variable override - only for edge cases"""
        
        # Only check environment variables - no config file clutter
        try:
            env_width = os.environ.get('CANRUN_DISPLAY_WIDTH')
            env_height = os.environ.get('CANRUN_DISPLAY_HEIGHT')
            
            if env_width and env_height:
                width = int(env_width)
                height = int(env_height)
                
                # Validate the override values
                if self._is_resolution_valid(width, height):
                    self.logger.warning(f"Using emergency environment override: {width}x{height}")
                    self.logger.warning("Note: This should only be used if automatic detection fails")
                    return {
                        'width': width,
                        'height': height,
                        'device': 'EMERGENCY_OVERRIDE',
                        'method': 'environment_override',
                        'is_primary': True
                    }
                else:
                    self.logger.error(f"Invalid override resolution: {width}x{height}")
        except (ValueError, TypeError) as e:
            self.logger.debug(f"No valid environment override: {e}")
        
        return None
    
    def _detect_windows(self) -> Dict:
        """Use proper Windows API for accurate detection"""
        
        # Try win32api first (most accurate)
        result = self._detect_windows_win32api()
        if result:
            return result
        
        # Fallback to ctypes
        result = self._detect_windows_ctypes()
        if result:
            return result
        
        # Last resort fallback
        return self._detect_fallback()
    
    def _detect_windows_win32api(self) -> Optional[Dict]:
        """Use win32api for the most accurate detection"""
        try:
            import win32api
            
            # Get all monitors
            monitors = win32api.EnumDisplayMonitors()
            
            if not monitors:
                self.logger.warning("No monitors detected via win32api")
                return None
            
            # Find primary monitor
            for monitor in monitors:
                # Get detailed info for each monitor
                info = win32api.GetMonitorInfo(monitor[0])
                
                # Check if this is the primary monitor
                # MONITORINFOF_PRIMARY = 1
                if info['Flags'] & 1:
                    work_area = info['Work']
                    monitor_area = info['Monitor']
                    
                    # Use monitor area for full resolution
                    width = monitor_area[2] - monitor_area[0]
                    height = monitor_area[3] - monitor_area[1]
                    
                    # Sanity check
                    if self._is_resolution_valid(width, height):
                        self.logger.info(f"Primary monitor detected: {width}x{height} ({info['Device']})")
                        
                        return {
                            'width': width,
                            'height': height,
                            'device': info['Device'],
                            'method': 'win32api',
                            'is_primary': True,
                            'work_area': {
                                'width': work_area[2] - work_area[0],
                                'height': work_area[3] - work_area[1]
                            }
                        }
                    else:
                        self.logger.warning(f"Invalid resolution detected: {width}x{height}")
            
            # If no primary found (shouldn't happen), use first monitor
            if monitors:
                info = win32api.GetMonitorInfo(monitors[0][0])
                monitor_area = info['Monitor']
                width = monitor_area[2] - monitor_area[0]
                height = monitor_area[3] - monitor_area[1]
                
                if self._is_resolution_valid(width, height):
                    self.logger.warning(f"No primary monitor flagged, using first: {width}x{height}")
                    return {
                        'width': width,
                        'height': height,
                        'device': info['Device'],
                        'method': 'win32api',
                        'is_primary': False
                    }
                    
        except ImportError:
            self.logger.debug("win32api not available")
        except Exception as e:
            self.logger.error(f"win32api detection failed: {e}")
            
        return None
    
    def _detect_windows_ctypes(self) -> Optional[Dict]:
        """Fallback using ctypes if win32api not available"""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            
            # Set DPI awareness to get actual resolution
            try:
                user32.SetProcessDPIAware()
            except Exception as e:
                self.logger.debug(f"Could not set DPI awareness: {e}")
            
            # Get primary monitor dimensions
            width = user32.GetSystemMetrics(0)   # SM_CXSCREEN
            height = user32.GetSystemMetrics(1)  # SM_CYSCREEN
            
            if self._is_resolution_valid(width, height):
                self.logger.info(f"Primary monitor via ctypes: {width}x{height}")
                
                return {
                    'width': width,
                    'height': height,
                    'device': 'PRIMARY',
                    'method': 'ctypes',
                    'is_primary': True
                }
            else:
                self.logger.warning(f"Invalid resolution from ctypes: {width}x{height}")
                
        except Exception as e:
            self.logger.error(f"ctypes detection failed: {e}")
            
        return None
    
    def _detect_fallback(self) -> Dict:
        """Simple fallback for non-Windows or when APIs fail"""
        self.logger.warning("Using fallback resolution detection")
        
        # Try tkinter as last resort, but with DPI awareness
        try:
            import tkinter as tk
            root = tk.Tk()
            
            # Try to set DPI awareness in tkinter
            try:
                root.tk.call('tk', 'scaling', 1.0)
            except:
                pass
            
            # Hide the window
            root.withdraw()
            
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            
            # Validate resolution
            if self._is_resolution_valid(width, height):
                self.logger.info(f"Display via tkinter: {width}x{height}")
                return {
                    'width': width,
                    'height': height,
                    'device': 'DISPLAY',
                    'method': 'tkinter',
                    'is_primary': True
                }
            else:
                self.logger.warning(f"Invalid tkinter resolution: {width}x{height}")
                
        except Exception as e:
            self.logger.debug(f"tkinter detection failed: {e}")
        
        # Ultimate fallback
        self.logger.info("Using default resolution 1920x1080")
        return {
            'width': 1920,
            'height': 1080,
            'device': 'DEFAULT',
            'method': 'default',
            'is_primary': True
        }
    
    def _is_resolution_valid(self, width: int, height: int) -> bool:
        """Validate that a resolution is reasonable"""
        
        # Check minimum size
        if width < 800 or height < 600:
            return False
        
        # Check maximum size (8K is 7680x4320)
        if width > 7680 or height > 4320:
            return False
        
        # Check aspect ratio (between 4:3 and 32:9)
        aspect_ratio = width / height
        if aspect_ratio < 1.0 or aspect_ratio > 5.33:
            return False
        
        return True
    
    def get_all_monitors(self) -> List[Dict]:
        """Get information about all connected monitors"""
        
        if sys.platform != 'win32':
            # Non-Windows: return single monitor
            primary = self.detect_primary_display()
            return [primary]
        
        try:
            import win32api
            
            monitors = []
            for monitor in win32api.EnumDisplayMonitors():
                info = win32api.GetMonitorInfo(monitor[0])
                
                monitor_area = info['Monitor']
                work_area = info['Work']
                
                width = monitor_area[2] - monitor_area[0]
                height = monitor_area[3] - monitor_area[1]
                
                if self._is_resolution_valid(width, height):
                    monitors.append({
                        'device': info['Device'],
                        'width': width,
                        'height': height,
                        'x': monitor_area[0],
                        'y': monitor_area[1],
                        'work_area': {
                            'width': work_area[2] - work_area[0],
                            'height': work_area[3] - work_area[1],
                            'x': work_area[0],
                            'y': work_area[1]
                        },
                        'is_primary': bool(info['Flags'] & 1)
                    })
            
            # Sort so primary is first
            monitors.sort(key=lambda m: not m['is_primary'])
            
            if monitors:
                self.logger.info(f"Detected {len(monitors)} monitor(s)")
                return monitors
            else:
                self.logger.warning("No valid monitors detected")
                
        except Exception as e:
            self.logger.error(f"Failed to enumerate monitors: {e}")
        
        # Fallback to single monitor
        primary = self.detect_primary_display()
        return [primary]
    
    def get_resolution_tier(self, width: int, height: int) -> str:
        """Categorize resolution into standard tiers based on performance impact"""
        
        total_pixels = width * height
        
        # 4K and above (includes ultrawide resolutions closer to 4K performance)
        if total_pixels >= 3840 * 2160 * 0.9:  # True 4K
            return '4K'
        elif total_pixels >= 2560 * 1440 * 1.4:  # High-res ultrawide (5M+ pixels)
            return '4K'  # Performance closer to 4K
        elif total_pixels >= 2560 * 1440 * 0.9:  # Standard 1440p
            return '1440p'
        elif total_pixels >= 1920 * 1080 * 0.9:  # 1080p
            return '1080p'
        elif total_pixels >= 1280 * 720 * 0.9:  # 720p
            return '720p'
        else:
            return 'SD'  # Standard definition


def main():
    """Test the display detector"""
    import logging
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    detector = DisplayDetector()
    
    print("=== Display Detection Test ===\n")
    
    # Test primary display detection
    primary = detector.get_display_resolution()
    print(f"Primary Display: {primary['width']}x{primary['height']}")
    print(f"  Device: {primary['device']}")
    print(f"  Method: {primary['method']}")
    print(f"  Is Primary: {primary.get('is_primary', False)}")
    
    if 'work_area' in primary:
        work = primary['work_area']
        print(f"  Work Area: {work['width']}x{work['height']}")
    
    tier = detector.get_resolution_tier(primary['width'], primary['height'])
    print(f"  Resolution Tier: {tier}")
    
    print("\n=== All Monitors ===\n")
    
    # Test all monitors detection
    all_monitors = detector.get_all_monitors()
    for i, monitor in enumerate(all_monitors, 1):
        print(f"Monitor {i}: {monitor['width']}x{monitor['height']}")
        print(f"  Device: {monitor['device']}")
        print(f"  Position: ({monitor.get('x', 0)}, {monitor.get('y', 0)})")
        print(f"  Primary: {monitor.get('is_primary', False)}")
        print()


if __name__ == "__main__":
    main()