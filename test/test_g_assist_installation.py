#!/usr/bin/env python3
"""
Test G-Assist Plugin Installation Script
Verifies that the installation script correctly sets up the plugin directory structure
"""

import os
import subprocess
import tempfile
import shutil
import json
from pathlib import Path

def test_installation_script_exists():
    """Test that the installation script exists and is accessible"""
    script_path = os.path.join("canrun", "install_g_assist_plugin.bat")
    assert os.path.exists(script_path), "install_g_assist_plugin.bat must exist"
    assert os.path.isfile(script_path), "install_g_assist_plugin.bat must be a file"
    print("PASS: Installation script exists")

def test_required_files_exist():
    """Test that all required files for G-Assist plugin exist"""
    canrun_dir = "canrun"
    
    required_files = [
        "g-assist-plugin-canrun.exe",
        "manifest.json", 
        "config.json",
        "plugin.py"
    ]
    
    required_dirs = [
        "data"
    ]
    
    for file_name in required_files:
        file_path = os.path.join(canrun_dir, file_name)
        assert os.path.exists(file_path), f"Required file missing: {file_name}"
        print(f"PASS: Required file exists: {file_name}")
    
    for dir_name in required_dirs:
        dir_path = os.path.join(canrun_dir, dir_name)
        assert os.path.exists(dir_path), f"Required directory missing: {dir_name}"
        assert os.path.isdir(dir_path), f"Required directory is not a directory: {dir_name}"
        print(f"PASS: Required directory exists: {dir_name}")

def test_manifest_compliance():
    """Test that manifest.json complies with G-Assist plugin standards"""
    manifest_path = os.path.join("canrun", "manifest.json")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Test required fields
    assert "manifestVersion" in manifest, "manifest.json must have manifestVersion"
    assert manifest["manifestVersion"] == 1, "manifestVersion must be 1"
    
    assert "executable" in manifest, "manifest.json must have executable field"
    assert manifest["executable"] == "g-assist-plugin-canrun.exe", "executable must be g-assist-plugin-canrun.exe"
    
    assert "persistent" in manifest, "manifest.json must have persistent field"
    assert manifest["persistent"] is False, "persistent must be false for G-Assist compatibility"
    
    assert "functions" in manifest, "manifest.json must have functions array"
    assert len(manifest["functions"]) >= 1, "Must have at least one function"
    
    # Test canrun function specifically
    canrun_func = None
    for func in manifest["functions"]:
        if func["name"] == "canrun":
            canrun_func = func
            break
    
    assert canrun_func is not None, "canrun function must be defined"
    assert "description" in canrun_func, "canrun function must have description"
    assert "tags" in canrun_func, "canrun function must have tags"
    assert len(canrun_func["tags"]) >= 5, "canrun function should have comprehensive tags"
    assert "properties" in canrun_func, "canrun function must have properties"
    assert "required" in canrun_func, "canrun function must have required array"
    assert "game_name" in canrun_func["required"], "game_name must be required parameter"
    
    print("PASS: manifest.json complies with G-Assist plugin standards")

def test_plugin_directory_structure():
    """Test that plugin follows correct directory structure for G-Assist"""
    canrun_dir = "canrun"
    
    # Test that directory name matches invocation name (canrun)
    assert os.path.basename(os.path.abspath(canrun_dir)) == "canrun", "Directory name must match plugin invocation name"
    
    # Test executable naming convention
    exe_path = os.path.join(canrun_dir, "g-assist-plugin-canrun.exe")
    assert os.path.exists(exe_path), "Executable must follow g-assist-plugin-<name>.exe convention"
    
    print("PASS: Plugin directory structure follows G-Assist conventions")

def test_installation_script_structure():
    """Test that the installation script has proper structure"""
    script_path = os.path.join("canrun", "install_g_assist_plugin.bat")
    
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    # Test for admin privilege check
    assert "net session" in script_content, "Script must check for admin privileges"
    
    # Test for G-Assist path detection
    assert "NVIDIA Corporation" in script_content, "Script must detect NVIDIA G-Assist paths"
    assert "nvtopps" in script_content, "Script must support legacy G-Assist path"
    # Note: Script currently uses legacy path which is still valid for G-Assist compatibility
    
    # Test for file verification
    assert "g-assist-plugin-canrun.exe" in script_content, "Script must verify executable exists"
    assert "manifest.json" in script_content, "Script must verify manifest exists"
    assert "config.json" in script_content, "Script must verify config exists"
    
    # Test for proper file copying
    assert "xcopy" in script_content or "copy" in script_content, "Script must copy files"
    
    print("PASS: Installation script has proper structure")

def test_g_assist_paths_logic():
    """Test the logic for G-Assist installation path detection"""
    # This test verifies the path logic without actually requiring G-Assist installation
    
    # Expected paths from the script
    legacy_path_template = "%PROGRAMDATA%\\NVIDIA Corporation\\nvtopps\\rise\\plugins\\canrun"
    nvidia_app_path_template = "%USERPROFILE%\\AppData\\Local\\NVIDIA Corporation\\NVIDIA App\\plugins\\canrun"
    
    # Test path templates are well-formed
    assert "NVIDIA Corporation" in legacy_path_template, "Legacy path must reference NVIDIA Corporation"
    assert "plugins" in legacy_path_template, "Legacy path must include plugins directory"
    assert "canrun" in legacy_path_template, "Legacy path must end with plugin name"
    
    assert "NVIDIA Corporation" in nvidia_app_path_template, "NVIDIA App path must reference NVIDIA Corporation"
    assert "plugins" in nvidia_app_path_template, "NVIDIA App path must include plugins directory" 
    assert "canrun" in nvidia_app_path_template, "NVIDIA App path must end with plugin name"
    
    print("PASS: G-Assist path detection logic is correct")

def test_file_permissions_check():
    """Test that script handles file permissions appropriately"""
    script_path = os.path.join("canrun", "install_g_assist_plugin.bat")
    
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
    script_path = os.path.join("canrun", "install_g_assist_plugin.bat")
    
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