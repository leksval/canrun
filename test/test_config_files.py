"""Test suite for config.json and manifest.json validation"""

import json
import pytest
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.json"
MANIFEST_PATH = Path(__file__).parent.parent / "manifest.json"

def test_config_exists():
    """Verify config.json exists"""
    assert CONFIG_PATH.exists(), "config.json not found"

def test_config_structure():
    """Verify config.json has required structure"""
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    
    assert "display_override" in config
    assert "steam_api" in config
    assert "ml_model" in config
    
    # Check nested fields
    assert isinstance(config["display_override"]["enabled"], bool)
    assert isinstance(config["steam_api"]["timeout"], int)
    assert isinstance(config["ml_model"]["path"], str)

def test_manifest_exists():
    """Verify manifest.json exists"""
    assert MANIFEST_PATH.exists(), "manifest.json not found"

def test_manifest_nvidia_fields():
    """Verify NVIDIA-specific fields in manifest"""
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    
    assert "nvidia_requirements" in manifest
    nvidia = manifest["nvidia_requirements"]
    
    assert "min_driver" in nvidia
    assert "gpu_series" in nvidia
    assert "os" in nvidia
    
    assert isinstance(nvidia["gpu_series"], list)
    assert all(series in ["RTX", "GTX"] for series in nvidia["gpu_series"])

def test_backward_compatibility():
    """Verify old config formats still work"""
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    
    # Test old display override format
    old_display = {"width": 1920, "height": 1080}
    assert "width" in config["display_override"]
    assert "height" in config["display_override"]