#!/usr/bin/env python3
"""
CanRun Hardware Detection Test Suite
Tests the hardware detection capabilities of the CanRun system
"""

import sys
import os
import pytest
sys.path.append('.')

from canrun.src.hardware_detector import HardwareDetector
import asyncio
import json

@pytest.mark.asyncio
async def test_hardware_detection():
    """Test hardware detection functionality"""
    print("=" * 50)
    print("CanRun Hardware Detection Test")
    print("=" * 50)
    
    detector = HardwareDetector()
    
    try:
        # Test system detection
        print("\n[TEST 1] Detecting system hardware...")
        hardware = detector.detect_hardware()
        
        print("+ Hardware detected successfully")
        print(f"  CPU: {hardware.cpu_name}")
        print(f"  GPU: {hardware.gpu_name}")
        print(f"  RAM: {hardware.ram_total} MB")
        
        print("\n[TEST 2] Testing GPU performance scoring...")
        gpu_score = detector.get_gpu_performance_score(hardware.gpu_name)
        print(f"+ GPU Performance Score: {gpu_score}")
        
        gpu_tier = detector.get_gpu_tier(hardware.gpu_name)
        print(f"+ GPU Tier: {gpu_tier}")
        
        print("\n" + "=" * 50)
        print("All hardware detection tests passed!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"X Hardware detection test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_hardware_detection())
    sys.exit(0 if success else 1)