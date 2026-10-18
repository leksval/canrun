#!/usr/bin/env python3
"""
Test G-Assist Plugin Installation Script
Verifies that the installation script correctly sets up the plugin directory structure
"""

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

def test_plugin_directory_structure():
    """Test that plugin follows the Protocol V2 directory structure"""
    assert PLUGIN_DIR.name == "canrun", "Directory name must match plugin invocation name"
    assert (PLUGIN_DIR / "plugin.py").is_file(), "executable must be plugin.py"
    assert (PLUGIN_DIR / "libs" / "gassist_sdk" / "__init__.py").is_file()
    
    print("PASS: Plugin directory structure follows G-Assist conventions")

def run_all_tests():
    """Run all G-Assist installation tests"""
    print("=== G-Assist Plugin Installation Tests ===")
    print()
    
    tests = [
        test_installation_script_exists,
        test_required_files_exist,
        test_plugin_directory_structure,
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
