#!/usr/bin/env python3
"""
Test G-Assist Plugin Installation Script
Verifies that the installation script correctly sets up the plugin directory structure
"""

import os
import json
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent

def test_installation_script_exists():
    """Test that the installation script exists and is accessible"""
    script_path = PLUGIN_DIR / "install_g_assist_plugin.bat"
    assert script_path.is_file(), "install_g_assist_plugin.bat must exist"
    print("PASS: Installation script exists")

def test_required_files_exist():
    """Test that all required files for the Protocol V2 plugin exist"""
    required_files = [
        "plugin.py",
        "manifest.json",
        "config.json",
        "requirements.txt",
        "libs/gassist_sdk/__init__.py",
    ]
    
    required_dirs = [
        "data",
        "libs/gassist_sdk",
    ]
    
    for file_name in required_files:
        file_path = PLUGIN_DIR / file_name
        assert file_path.is_file(), f"Required file missing: {file_name}"
        print(f"PASS: Required file exists: {file_name}")
    
    for dir_name in required_dirs:
        dir_path = PLUGIN_DIR / dir_name
        assert dir_path.is_dir(), f"Required directory missing: {dir_name}"
        print(f"PASS: Required directory exists: {dir_name}")

def test_manifest_compliance():
    """Test that manifest.json complies with G-Assist Protocol V2"""
    manifest_path = PLUGIN_DIR / "manifest.json"
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    assert manifest["manifestVersion"] == 1
    assert manifest["name"] == "canrun"
    assert manifest["executable"] == "plugin.py"
    assert manifest["persistent"] is True
    assert manifest["protocol_version"] == "2.0"
    
    assert "functions" in manifest, "manifest.json must have functions array"
    assert len(manifest["functions"]) >= 1, "Must have at least one function"
    
    check_compatibility = None
    for func in manifest["functions"]:
        if func["name"] == "check_compatibility":
            check_compatibility = func
            break
    
    assert check_compatibility is not None, "check_compatibility function must be defined"
    assert "description" in check_compatibility, "check_compatibility function must have description"
    assert "tags" in check_compatibility, "check_compatibility function must have tags"
    assert len(check_compatibility["tags"]) >= 5, "check_compatibility function should have comprehensive tags"
    assert "properties" in check_compatibility, "check_compatibility function must have properties"
    assert "required" in check_compatibility, "check_compatibility function must have required array"
    assert "game_name" in check_compatibility["required"], "game_name must be required parameter"
    
    print("PASS: manifest.json complies with G-Assist plugin standards")

def test_plugin_directory_structure():
    """Test that plugin follows the Protocol V2 directory structure"""
    assert PLUGIN_DIR.name == "canrun", "Directory name must match plugin invocation name"
    assert (PLUGIN_DIR / "plugin.py").is_file(), "executable must be plugin.py"
    assert (PLUGIN_DIR / "libs" / "gassist_sdk" / "__init__.py").is_file()
    
    print("PASS: Plugin directory structure follows G-Assist conventions")

def test_installation_script_structure():
    """Test that the installation script has proper structure"""
    script_path = PLUGIN_DIR / "install_g_assist_plugin.bat"
    
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    assert "net session" in script_content, "Script must check for admin privileges"
    assert "NVIDIA Corporation" in script_content, "Script must detect NVIDIA G-Assist paths"
    assert "nvtopps" in script_content, "Script must support the G-Assist plugins path"
    assert "plugin.py" in script_content, "Script must install plugin.py"
    assert "manifest.json" in script_content, "Script must verify manifest exists"
    assert "config.json" in script_content, "Script must verify config exists"
    assert "requirements.txt" in script_content, "Script must install Python dependencies"
    assert "gassist_sdk" in script_content, "Script must copy the vendored SDK"
    assert "canrun_steam_fps.py" in script_content, "Script must copy the Steam FPS client"
    assert "xcopy" in script_content or "copy" in script_content, "Script must copy files"
    
    print("PASS: Installation script has proper structure")

def test_g_assist_paths_logic():
    """Test the official G-Assist plugin install path"""
    script_path = PLUGIN_DIR / "install_g_assist_plugin.bat"
    script_content = script_path.read_text(encoding="utf-8")
    official_path = "%PROGRAMDATA%\\NVIDIA Corporation\\nvtopps\\rise\\plugins\\canrun"
    assert official_path in script_content, "Installer must target the official plugins directory"
    
    print("PASS: G-Assist path detection logic is correct")

def test_file_permissions_check():
    """Test that script handles file permissions appropriately"""
    script_path = PLUGIN_DIR / "install_g_assist_plugin.bat"
    
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    # Test for error handling
    assert "errorLevel" in script_content, "Script must check for command errors"
    assert "pause" in script_content, "Script must pause on errors for user visibility"
    
    # Test for directory creation
    assert "mkdir" in script_content, "Script must create directories"
    
    print("PASS: Installation script handles permissions and errors correctly")

def test_installation_verification():
    """Test that installation script includes verification steps"""
    script_path = PLUGIN_DIR / "install_g_assist_plugin.bat"
    
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    # Test for post-installation verification
    assert "Verifying installation" in script_content, "Script must verify installation"
    assert "exist" in script_content, "Script must check if files were copied successfully"
    
    # Test for user instructions
    assert "restart" in script_content.lower(), "Script must instruct user to restart G-Assist"
    assert "voice commands" in script_content.lower(), "Script must provide usage examples"
    
    print("PASS: Installation script includes verification and user instructions")

def run_all_tests():
    """Run all G-Assist installation tests"""
    print("=== G-Assist Plugin Installation Tests ===")
    print()
    
    tests = [
        test_installation_script_exists,
        test_required_files_exist,
        test_manifest_compliance,
        test_plugin_directory_structure,
        test_installation_script_structure,
        test_g_assist_paths_logic,
        test_file_permissions_check,
        test_installation_verification
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            print(f"Running {test_func.__name__}...")
            test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"FAIL: {e}")
            print()
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            print()
            failed += 1
    
    print("=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("All tests passed! G-Assist plugin installation should work correctly.")
        return True
    else:
        print("Some tests failed. Installation may have issues.")
        return False

if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)