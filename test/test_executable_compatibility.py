"""Test suite for g-assist-plugin-canrun.exe compatibility"""

import os
import hashlib
import subprocess
import pytest
from pathlib import Path

EXE_NAME = "g-assist-plugin-canrun.exe"
EXPECTED_HASH = "a1b2c3d4e5f6..."  # TODO: Get actual v7.5.0 hash

@pytest.fixture
def exe_path():
    path = Path(__file__).parent.parent / EXE_NAME
    assert path.exists(), f"{EXE_NAME} not found at {path}"
    return path

def test_executable_exists(exe_path):
    """Verify executable exists"""
    assert exe_path.is_file()

@pytest.mark.skip(reason="Executable CLI tests may hang due to network initialization - use test_cli_standalone.py instead")
def test_executable_cli_initialize(exe_path):
    """Test CLI mode initialize command - SKIPPED: Use standalone CLI test instead"""
    pass

@pytest.mark.skip(reason="Executable CLI tests may hang due to network initialization - use test_cli_standalone.py instead")
def test_executable_cli_hardware_detection(exe_path):
    """Test CLI mode hardware detection - SKIPPED: Use standalone CLI test instead"""
    pass

@pytest.mark.skip(reason="Executable CLI tests may hang due to network initialization - use test_cli_standalone.py instead")
def test_executable_cli_game_check(exe_path):
    """Test CLI mode game compatibility check - SKIPPED: Use standalone CLI test instead"""
    pass

@pytest.mark.skip(reason="Executable CLI tests may hang due to network initialization - use test_cli_standalone.py instead")
def test_executable_cli_shutdown(exe_path):
    """Test CLI mode shutdown command - SKIPPED: Use standalone CLI test instead"""
    pass