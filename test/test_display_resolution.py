"""Test display resolution detection"""
import ctypes
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from canrun_hardware_detector import CanRunHardwareDetector
import asyncio

def test_display_detection():
    """Test display detection methods"""
    print("Testing Display Resolution Detection")
    print("=" * 50)
    
    # Method 1: Direct Windows API
    if sys.platform == "win32":
        try:
            user32 = ctypes.windll.user32
            width = user32.GetSystemMetrics(0)
            height = user32.GetSystemMetrics(1)
            print(f"Windows API Detection: {width}x{height}")
        except Exception as e:
            print(f"Windows API failed: {e}")
    
    # Method 2: CanRunHardwareDetector
    detector = CanRunHardwareDetector()
    display_info = detector.detect_primary_display()
    print(f"Hardware Detector: {display_info['width']}x{display_info['height']} (method: {display_info['method']})")
    
    # Method 3: get_display_resolution (with override support)
    resolution = detector.get_display_resolution()
    print(f"Get Display Resolution: {resolution['width']}x{resolution['height']} (method: {resolution['method']})")
    
    # Test async method
    async def test_async():
        specs = await detector.get_hardware_specs()
        return specs.primary_monitor_resolution
    
    async_result = asyncio.run(test_async())
    print(f"Async Hardware Specs: {async_result}")
    
    # Resolution tier
    if display_info['width'] and display_info['height']:
        tier = detector.get_resolution_tier(display_info['width'], display_info['height'])
        print(f"Resolution Tier: {tier}")

if __name__ == "__main__":
    test_display_detection()