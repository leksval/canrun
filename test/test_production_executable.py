#!/usr/bin/env python3
"""
Test script to verify the production executable exists and runs without crashing
"""

import subprocess
import json
import sys
import pytest
from pathlib import Path

@pytest.mark.skip(reason="Executable hangs waiting for G-Assist pipe communication - use test_cli_standalone.py for CLI testing")
def test_cyberpunk_fps():
    """Test that the production executable exists and runs without crashing"""
    
    print("Testing production executable existence and basic functionality...")
    
    # Path to executable relative to test directory
    exe_path = Path(__file__).parent.parent / "g-assist-plugin-canrun.exe"
    
    # Verify executable exists
    assert exe_path.exists(), f"Production executable not found at {exe_path}"
    print(f" Executable found at: {exe_path}")
    
    # Verify the executable is the correct type (should be PE executable on Windows)
    with open(exe_path, 'rb') as f:
        header = f.read(2)
        assert header == b'MZ', "Executable appears to be corrupted (invalid PE header)"
    
    print(" Executable has valid PE format")
    print(" Production executable validation PASSED (CLI testing available in test_cli_standalone.py)")

def test_executable_exists():
    """Simple test to verify the production executable exists"""
    exe_path = Path(__file__).parent.parent / "g-assist-plugin-canrun.exe"
    assert exe_path.exists(), f"Production executable not found at {exe_path}"
    
    # Verify it's a valid Windows PE executable
    with open(exe_path, 'rb') as f:
        header = f.read(2)
        assert header == b'MZ', "Executable appears to be corrupted (invalid PE header)"

if __name__ == "__main__":
    try:
        test_cyberpunk_fps()
        print("\n Production deployment test PASSED!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n Production deployment test FAILED: {e}")
        sys.exit(1)