"""
Test display resolution detection fix for G-Assist plugin.
Verifies that RTX 4090 systems show "4K" resolution instead of hardcoded "1080p".
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from canrun_hardware_detector import CanRunHardwareDetector


def test_hardware_detector_display_resolution():
    """Test that hardware detector properly detects display resolution."""
    print("\n[TEST] Hardware Detector Display Resolution")
    
    detector = CanRunHardwareDetector()
    
    # Test primary display detection
    primary_display = detector.detect_primary_display()
    print(f"Primary Display: {primary_display['width']}x{primary_display['height']} (method: {primary_display['method']})")
    
    # Test resolution classification
    width = primary_display['width']
    height = primary_display['height']
    tier = detector.get_resolution_tier(width, height)
    print(f"Resolution Tier: {tier}")
    
    # Test get_display_resolution (with override support)
    display_resolution = detector.get_display_resolution()
    print(f"Display Resolution: {display_resolution['width']}x{display_resolution['height']} (method: {display_resolution['method']})")
    
    # Verify resolution tier classification logic
    test_resolutions = [
        (3840, 2160, "4K"),
        (2560, 1440, "1440p"), 
        (1920, 1080, "1080p"),
        (1280, 720, "720p")
    ]
    
    print("\nResolution Classification Tests:")
    for width, height, expected_tier in test_resolutions:
        actual_tier = detector.get_resolution_tier(width, height)
        status = "PASS" if actual_tier == expected_tier else "FAIL"
        print(f"  {width}x{height} -> {actual_tier} (expected: {expected_tier}) [{status}]")
        assert actual_tier == expected_tier, f"Expected {expected_tier}, got {actual_tier} for {width}x{height}"
    
    return primary_display, tier


def test_display_resolution_consistency():
    """Test consistency between different detection methods."""
    print("\n[TEST] Display Resolution Consistency")
    
    detector = CanRunHardwareDetector()
    
    # Get resolution from different methods
    primary_display = detector.detect_primary_display()
    display_resolution = detector.get_display_resolution()
    
    # Check consistency
    primary_width = primary_display['width']
    primary_height = primary_display['height']
    resolution_width = display_resolution['width']
    resolution_height = display_resolution['height']
    
    print(f"Primary Display: {primary_width}x{primary_height}")
    print(f"Get Display Resolution: {resolution_width}x{resolution_height}")
    
    # They should match unless there's an override
    if display_resolution['method'] != 'config_override' and display_resolution['method'] != 'environment_override':
        assert primary_width == resolution_width, f"Width mismatch: {primary_width} vs {resolution_width}"
        assert primary_height == resolution_height, f"Height mismatch: {primary_height} vs {resolution_height}"
        print("[SUCCESS] Display resolution methods are consistent")
    else:
        print(f"[INFO] Override detected (method: {display_resolution['method']}), consistency check skipped")


async def main():
    """Run all display resolution tests."""
    print("Testing Display Resolution Detection Fix for G-Assist Plugin")
    print("=" * 60)
    
    try:
        # Test 1: Hardware detector display resolution
        primary_display, tier = test_hardware_detector_display_resolution()
        
        # Test 2: Consistency check
        test_display_resolution_consistency()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] All display resolution tests passed!")
        print(f"Current System: {primary_display['width']}x{primary_display['height']} -> {tier}")
        
        # Check if this is likely a high-end system
        if tier == "4K":
            print("[INFO] 4K display detected - perfect for RTX 4090 testing!")
        elif tier == "1440p":
            print("[INFO] 1440p display detected - good for high-end GPU testing")
        else:
            print("[INFO] Lower resolution display - resolution detection logic verified")
            
    except Exception as e:
        print(f"\n[ERROR] Display resolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)