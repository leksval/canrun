#!/usr/bin/env python3
"""
CanRun Test Suite Runner
Runs all CanRun tests and reports results
"""

import sys
import os
import subprocess
import asyncio
from pathlib import Path

# Add src to path
sys.path.append('..')

def run_test_script(script_path):
    """Run a test script and return success status"""
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=60)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Test timed out after 60 seconds"
    except Exception as e:
        return False, "", f"Error running test: {e}"

def main():
    """Run all CanRun tests"""
    print("=" * 60)
    print("CanRun Complete Test Suite")
    print("=" * 60)
    
    # Get test directory
    test_dir = Path(__file__).parent
    
    # Define test scripts in order
    test_scripts = [
        ("Hardware Detection", "test_hardware_detection.py"),
        ("Privacy Aware Hardware", "test_privacy_aware_hardware.py"),
        ("Compatibility Analysis", "test_compatibility_analysis.py"),
        ("Performance Prediction", "test_performance_prediction.py"),
        ("Minimum Requirements", "test_minimum_requirements.py"),
        ("Steam API Integration", "test_steam_api_integration.py"),
        ("LLM Analysis", "test_llm_analysis.py"),
        ("CanRun Integration", "test_canrun_integration.py"),
        ("Plugin Responses", "test_plugin_responses.py"),
        ("Plugin Core", "test_plugin.py"),
        ("Project Validation", "test_project_validation.py"),
        ("Diablo Compatibility", "test_diablo_compatibility.py"),
        ("Diablo Matching", "test_diablo_matching.py"),
        ("Performance Debug", "test_performance_debug.py")
    ]
    
    results = []
    total_tests = len(test_scripts)
    passed_tests = 0
    
    print(f"\nRunning {total_tests} test suites...\n")
    
    for i, (test_name, script_name) in enumerate(test_scripts, 1):
        print(f"[{i}/{total_tests}] Running {test_name} Tests...")
        print("-" * 50)
        
        script_path = test_dir / script_name
        if not script_path.exists():
            print(f"X Test script not found: {script_name}")
            results.append((test_name, False, "Script not found"))
            continue
        
        success, stdout, stderr = run_test_script(script_path)
        
        if success:
            print(f"+ {test_name} tests PASSED")
            passed_tests += 1
            results.append((test_name, True, ""))
        else:
            print(f"X {test_name} tests FAILED")
            results.append((test_name, False, stderr))
            
            # Show error details
            if stderr:
                print(f"Error: {stderr}")
        
        print()
    
    # Print summary
    print("=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, success, error in results:
        status = "+ PASS" if success else "X FAIL"
        print(f"{status:8} {test_name}")
        if error and not success:
            print(f"         Error: {error}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} test suites passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! CanRun is ready for use.")
        return 0
    else:
        print(f"\nX {total_tests - passed_tests} test suite(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)