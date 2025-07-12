#!/usr/bin/env python3
"""
CanRun Hardware Detection Test Suite
Tests the hardware detection capabilities of the CanRun system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hardware_detector import HardwareDetector
import asyncio
import json

async def test_hardware_detection():
    """Test hardware detection functionality"""
    print("=" * 50)
    print("CanRun Hardware Detection Test")
    print("=" * 50)
    
    detector = HardwareDetector()
    
    try:
        # Test system detection
        print("\n[TEST 1] Detecting system hardware...")
        hardware = await detector.detect_system_hardware()
        
        print(f"✓ Hardware detected successfully")
        print(f"  CPU: {hardware.get('cpu', {}).get('name', 'Unknown')}")
        print(f"  GPU: {hardware.get('gpu', {}).get('name', 'Unknown')}")
        print(f"  RAM: {hardware.get('memory', {}).get('total', 'Unknown')} GB")
        
        # Test individual components
        print("\n[TEST 2] Testing individual component detection...")
        
        # CPU test
        cpu_info = await detector.detect_cpu()
        print(f"✓ CPU: {cpu_info.get('name', 'Unknown')} - {cpu_info.get('cores', 0)} cores")
        
        # GPU test  
        gpu_info = await detector.detect_gpu()
        print(f"✓ GPU: {gpu_info.get('name', 'Unknown')} - {gpu_info.get('memory', 0)} GB VRAM")
        
        # Memory test
        memory_info = await detector.detect_memory()
        print(f"✓ Memory: {memory_info.get('total', 0)} GB total, {memory_info.get('available', 0)} GB available")
        
        # Storage test
        storage_info = await detector.detect_storage()
        print(f"✓ Storage: {len(storage_info)} drives detected")
        
        print("\n[TEST 3] Performance scoring...")
        cpu_score = detector.calculate_cpu_score(cpu_info)
        gpu_score = detector.calculate_gpu_score(gpu_info)
        
        print(f"✓ CPU Performance Score: {cpu_score}")
        print(f"✓ GPU Performance Score: {gpu_score}")
        
        print("\n" + "=" * 50)
        print("All hardware detection tests passed!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"✗ Hardware detection test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_hardware_detection())
    sys.exit(0 if success else 1)