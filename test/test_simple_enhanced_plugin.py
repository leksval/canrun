#!/usr/bin/env python3
"""
Simple Enhanced G-Assist Plugin Test
Quick test to verify basic functionality without timeouts.
"""

import json
import subprocess
import sys
import os

def test_simple_hardware_detection():
    """Test simple hardware detection with shorter timeout."""
    print("🧪 SIMPLE HARDWARE DETECTION TEST")
    print("-" * 40)
    
    command = {
        "tool_calls": [
            {
                "func": "detect_hardware",
                "params": {}
            }
        ]
    }
    
    command_json = json.dumps(command)
    print(f"📨 Sending: {command_json}")
    
    try:
        # Use uv run to ensure proper environment
        process = subprocess.Popen(
            ["uv", "run", "python", "plugin.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="."
        )
        
        # Shorter timeout
        stdout, stderr = process.communicate(input=command_json + "\n", timeout=15)
        
        print(f"📤 stdout: {stdout}")
        if stderr:
            print(f"⚠️  stderr: {stderr}")
        
        if "<<END>>" in stdout:
            response_part = stdout.split("<<END>>")[0]
            try:
                response_data = json.loads(response_part)
                print("✅ Valid JSON response received")
                if response_data.get("success"):
                    print("✅ Hardware detection successful")
                    return True
                else:
                    print(f"❌ Hardware detection failed: {response_data.get('message')}")
                    return False
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON: {e}")
                return False
        else:
            print("❌ No proper response format")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 15 seconds")
        process.kill()
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_cli_interface():
    """Test command line interface."""
    print("\n🧪 CLI INTERFACE TEST")
    print("-" * 40)
    
    try:
        result = subprocess.run(
            ["uv", "run", "python", "plugin.py", "--function", "detect_hardware"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        print(f"Exit code: {result.returncode}")
        print(f"stdout: {result.stdout}")
        if result.stderr:
            print(f"stderr: {result.stderr}")
        
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout)
                if response.get("success"):
                    print("✅ CLI test successful")
                    return True
                else:
                    print(f"❌ CLI response failed: {response.get('message')}")
                    return False
            except json.JSONDecodeError:
                print("❌ CLI output not valid JSON")
                return False
        else:
            print("❌ CLI test failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ CLI test timed out")
        return False
    except Exception as e:
        print(f"❌ CLI test error: {e}")
        return False

def test_manifest_validation():
    """Test manifest.json validation."""
    print("\n🧪 MANIFEST VALIDATION TEST")
    print("-" * 40)
    
    try:
        with open("manifest.json", "r") as f:
            manifest = json.load(f)
        
        print("✅ Manifest JSON is valid")
        
        # Check required fields
        required_fields = ["name", "version", "functions"]
        for field in required_fields:
            if field in manifest:
                print(f"✅ {field}: {manifest[field] if field != 'functions' else f'{len(manifest[field])} functions'}")
            else:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Check enhanced functions
        functions = manifest.get("functions", [])
        enhanced_functions = ["steam_compare", "compare_games", "auto_detect"]
        
        for func_name in enhanced_functions:
            if any(f.get("name") == func_name for f in functions):
                print(f"✅ Enhanced function found: {func_name}")
            else:
                print(f"❌ Missing enhanced function: {func_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Manifest validation failed: {e}")
        return False

def main():
    """Run simple tests."""
    print("🚀 SIMPLE ENHANCED PLUGIN TEST SUITE")
    print("=" * 50)
    
    results = []
    
    # Test 1: Manifest validation
    results.append(test_manifest_validation())
    
    # Test 2: CLI interface
    results.append(test_cli_interface())
    
    # Test 3: Simple hardware detection
    results.append(test_simple_hardware_detection())
    
    print("\n" + "=" * 50)
    print("🎯 TEST RESULTS")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed - check plugin implementation")

if __name__ == "__main__":
    main()