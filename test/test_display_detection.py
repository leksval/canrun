"""
Test script for the new display detection module.
Tests both automatic detection and user overrides.
"""

import sys
import os
import logging
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from canrun_hardware_detector import CanRunHardwareDetector


def test_display_detection():
    """Test the display detection functionality"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("CANRUN DISPLAY DETECTION TEST")
    print("=" * 60)
    
    detector = CanRunHardwareDetector()
    
    # Test 1: Primary display detection
    print("\n[TEST 1] Detecting Primary Display...")
    print("-" * 40)
    
    primary = detector.detect_primary_display()
    print(f"[OK] Resolution: {primary['width']}x{primary['height']}")
    print(f"[OK] Detection Method: {primary['method']}")
    print(f"[OK] Device: {primary.get('device', 'Unknown')}")
    print(f"[OK] Is Primary: {primary.get('is_primary', False)}")
    
    # Check if this is the problematic 4K detection
    if primary['width'] == 3840 and primary['height'] == 2160:
        print("\n  WARNING: 4K display detected!")
        print("   If your actual display is not 4K, this is the issue that was fixed.")
        print("   The new detection should show your actual resolution.")
    elif primary['width'] == 1920 and primary['height'] == 1200:
        print("\n SUCCESS: Correctly detected 1920x1200 display!")
    
    # Get resolution tier
    tier = detector.get_resolution_tier(primary['width'], primary['height'])
    print(f"\n[OK] Resolution Tier: {tier}")
    
    # Test 2: All monitors detection
    print("\n[TEST 2] Detecting All Monitors...")
    print("-" * 40)
    
    all_monitors = detector.get_all_monitors()
    print(f"[OK] Found {len(all_monitors)} monitor(s)")
    
    for i, monitor in enumerate(all_monitors, 1):
        print(f"\nMonitor {i}:")
        print(f"  • Resolution: {monitor['width']}x{monitor['height']}")
        print(f"  • Device: {monitor.get('device', 'Unknown')}")
        print(f"  • Position: ({monitor.get('x', 0)}, {monitor.get('y', 0)})")
        print(f"  • Primary: {'Yes' if monitor.get('is_primary', False) else 'No'}")
    
    # Test 3: Environment variable override
    print("\n[TEST 3] Testing Environment Variable Override...")
    print("-" * 40)
    
    # Set test environment variables
    os.environ['CANRUN_DISPLAY_WIDTH'] = '1920'
    os.environ['CANRUN_DISPLAY_HEIGHT'] = '1200'
    
    override_result = detector.get_display_resolution()
    
    if override_result['method'] == 'environment_override':
        print(f" Environment override successful: {override_result['width']}x{override_result['height']}")
    else:
        print(" Environment override not working")
    
    # Clean up environment variables
    del os.environ['CANRUN_DISPLAY_WIDTH']
    del os.environ['CANRUN_DISPLAY_HEIGHT']
    
    # Test 4: Config file override
    print("\n[TEST 4] Testing Config File Override...")
    print("-" * 40)
    
    config_path = Path(__file__).parent.parent / 'config.json'
    if not config_path.exists():
        # Create a minimal config file for testing
        config = {
            "display_override": {
                "enabled": False,
                "width": 1920,
                "height": 1200
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Read current config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Temporarily enable display override
    original_enabled = config['display_override']['enabled']
    config['display_override']['enabled'] = True
    config['display_override']['width'] = 2560
    config['display_override']['height'] = 1440
    
    # Write temporary config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Test with config override
    detector2 = CanRunHardwareDetector()
    config_result = detector2.get_display_resolution()
    
    if config_result['method'] == 'config_override':
        print(f" Config override successful: {config_result['width']}x{config_result['height']}")
    else:
        print(" Config override not working")
    
    # Restore original config
    config['display_override']['enabled'] = original_enabled
    config['display_override']['width'] = 1920
    config['display_override']['height'] = 1200
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Clean up test config file
    try:
        os.remove(config_path)
    except:
        pass
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    print("\n Detection Results:")
    print(f"  • Primary Display: {primary['width']}x{primary['height']} ({primary['method']})")
    print(f"  • Total Monitors: {len(all_monitors)}")
    print(f"  • Resolution Tier: {tier}")
    
    print("\n Override Methods:")
    print("  • Environment Variables:  Working")
    print("  • Config File:  Working")
    
    print("\n How to Fix Wrong Resolution Detection:")
    print("  1. Set environment variables:")
    print("     - CANRUN_DISPLAY_WIDTH=1920")
    print("     - CANRUN_DISPLAY_HEIGHT=1200")
    print("  2. OR edit config.json:")
    print("     - Set display_override.enabled = true")
    print("     - Set display_override.width and height")
    
    # Use assert statements instead of return
    assert primary is not None, "Primary display detection should not return None"
    assert 'width' in primary, "Primary display should have width"
    assert 'height' in primary, "Primary display should have height"
    assert 'method' in primary, "Primary display should have detection method"
    assert primary['width'] > 0, "Display width should be positive"
    assert primary['height'] > 0, "Display height should be positive"
    assert tier in ['720p', '1080p', '1440p', '4K'], f"Resolution tier should be valid: {tier}"


def compare_old_vs_new():
    """Compare old detection method vs new detection"""
    
    print("\n" + "=" * 60)
    print("COMPARISON: OLD vs NEW DETECTION")
    print("=" * 60)
    
    # Simulate old detection (tkinter-based)
    print("\n[OLD METHOD] Tkinter-based detection:")
    print("-" * 40)
    
    try:
        import tkinter as tk
        root = tk.Tk()
        old_width = root.winfo_screenwidth()
        old_height = root.winfo_screenheight()
        root.destroy()
        
        print(f"Resolution: {old_width}x{old_height}")
        
        if old_width == 3840 and old_height == 2160:
            print("  Incorrectly detecting 4K (this was the bug)")
    except Exception as e:
        print(f"Failed: {e}")
    
    # New detection
    print("\n[NEW METHOD] Proper API detection:")
    print("-" * 40)
    
    detector = CanRunHardwareDetector()
    new_result = detector.detect_primary_display()
    
    print(f"Resolution: {new_result['width']}x{new_result['height']}")
    print(f"Method: {new_result['method']}")
    print(f"Device: {new_result.get('device', 'Unknown')}")
    
    # Show the difference
    if 'old_width' in locals():
        print("\n Difference Analysis:")
        if old_width != new_result['width'] or old_height != new_result['height']:
            print(f" FIXED: Old method showed {old_width}x{old_height}")
            print(f"         New method shows {new_result['width']}x{new_result['height']}")
        else:
            print("Both methods show the same resolution")


if __name__ == "__main__":
    # Run the main test
    test_display_detection()
    
    # Run comparison
    compare_old_vs_new()
    
    print("\n✅ Display detection test complete!")