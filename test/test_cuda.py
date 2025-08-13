#!/usr/bin/env python3
"""
CUDA Detection Test Script
Tests PyTorch CUDA availability and provides detailed diagnostics
"""

import sys
import os

print("=== CUDA Detection Diagnostics ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Test PyTorch import
try:
    import torch
    print(f"[OK] PyTorch imported successfully: {torch.__version__}")
    
    # Test CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version (PyTorch): {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        
        # Test basic tensor operations
        try:
            x = torch.rand(5, 3).cuda()
            print(f"[OK] GPU tensor creation successful: {x.device}")
        except Exception as e:
            print(f"[FAIL] GPU tensor creation failed: {e}")
    else:
        print("[FAIL] CUDA not detected by PyTorch")
        print("Possible causes:")
        print("1. PyTorch CPU-only version installed")
        print("2. CUDA driver/runtime mismatch")
        print("3. Environment path issues")
        
        # Check if we can see CUDA files
        cuda_paths = [
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
            "C:\\Program Files\\NVIDIA Corporation\\NVIDIA GPU Computing Toolkit\\CUDA"
        ]
        
        for path in cuda_paths:
            if os.path.exists(path):
                print(f"[OK] CUDA toolkit found at: {path}")
                try:
                    versions = os.listdir(path)
                    print(f"  Available versions: {versions}")
                except:
                    pass
            else:
                print(f"[FAIL] CUDA toolkit not found at: {path}")
    
except ImportError as e:
    print(f"[FAIL] Failed to import PyTorch: {e}")

# Test XGBoost GPU
print("\n=== XGBoost GPU Support ===")
try:
    import xgboost as xgb
    print(f"[OK] XGBoost imported: {xgb.__version__}")
    
    # Test XGBoost GPU training
    try:
        import numpy as np
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        
        # Try GPU training with XGBoost 2.0+ API
        model = xgb.XGBRegressor(device='cuda', tree_method='hist')
        model.fit(X, y)
        print("[OK] XGBoost GPU training successful")
    except Exception as e:
        print(f"[FAIL] XGBoost GPU training failed: {e}")
        
        # Try CPU training
        try:
            model = xgb.XGBRegressor(tree_method='hist')
            model.fit(X, y)
            print("[OK] XGBoost CPU training successful")
        except Exception as e2:
            print(f"[FAIL] XGBoost CPU training also failed: {e2}")
        
except ImportError as e:
    print(f"[FAIL] Failed to import XGBoost: {e}")

print("\n=== Environment Variables ===")
cuda_env_vars = ['CUDA_PATH', 'CUDA_HOME', 'PATH']
for var in cuda_env_vars:
    value = os.environ.get(var, 'Not set')
    if var == 'PATH':
        # Check if CUDA is in PATH
        cuda_in_path = any('cuda' in p.lower() for p in value.split(';')) if value != 'Not set' else False
        print(f"{var}: CUDA {'found' if cuda_in_path else 'not found'} in PATH")
    else:
        print(f"{var}: {value}")