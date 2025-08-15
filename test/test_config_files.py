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
    """Verify config.json has basic structure"""
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    
    assert "plugin_info" in config
    assert "windows_pipe_config" in config
    assert "logging_config" in config
    assert "canrun_config" in config
    
    # Check nested fields
    assert isinstance(config["plugin_info"]["version"], str)
    assert isinstance(config["windows_pipe_config"]["BUFFER_SIZE"], int)
    assert isinstance(config["logging_config"]["log_level"], str)

def test_manifest_exists():
    """Verify manifest.json exists"""
    assert MANIFEST_PATH.exists(), "manifest.json not found"

def test_manifest_basic_structure():
    """Verify manifest.json has basic G-Assist structure"""
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    
    assert "manifestVersion" in manifest
    assert "executable" in manifest
    assert "functions" in manifest
    
    assert isinstance(manifest["manifestVersion"], int)
    assert isinstance(manifest["executable"], str)
    assert isinstance(manifest["functions"], list)
    assert len(manifest["functions"]) > 0
    
    # Check first function has required fields
    func = manifest["functions"][0]
    assert "name" in func
    assert "description" in func
    assert "properties" in func

def test_config_version_updated():
    """Verify config version has been updated to 8.8.8"""
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    
    assert config["plugin_info"]["version"] == "8.8.8"