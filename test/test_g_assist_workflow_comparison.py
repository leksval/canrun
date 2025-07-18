#!/usr/bin/env python3
"""
Compare G-Assist Plugin Workflow vs Terminal Workflow
This test helps identify the exact differences between working terminal mode and G-Assist mode.
"""

import json
import subprocess
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_terminal_workflow():
    """Test the working terminal workflow."""
    print("🧪 Testing Terminal Workflow (Known Working)")
    print("=" * 50)
    
    plugin_path = project_root / "plugin.py"
    
    try:
        # Run in CLI mode
        result = subprocess.run(
            [sys.executable, str(plugin_path), "--function", "detect_hardware"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=15
        )
        
        print(f"📤 Command: python plugin.py --function detect_hardware")
        print(f"📥 Exit Code: {result.returncode}")
        print(f"📥 stdout: {repr(result.stdout)}")
        print(f"📥 stderr: {repr(result.stderr[:500])}..." if len(result.stderr) > 500 else f"📥 stderr: {repr(result.stderr)}")
        
        # Parse stdout as JSON
        try:
            response = json.loads(result.stdout)
            print("✅ TERMINAL MODE: JSON parsing successful")
            print(f"   Success: {response.get('success')}")
            print(f"   Message length: {len(response.get('message', ''))}")
        except json.JSONDecodeError as e:
            print(f"❌ TERMINAL MODE: JSON parsing failed: {e}")
            
    except Exception as e:
        print(f"❌ TERMINAL MODE: Test failed: {e}")

def test_g_assist_workflow():
    """Test the G-Assist workflow."""
    print("\n🧪 Testing G-Assist Workflow (Problematic)")
    print("=" * 50)
    
    plugin_path = project_root / "plugin.py"
    
    try:
        # Run in G-Assist mode (no arguments, stdin input)
        process = subprocess.Popen(
            [sys.executable, str(plugin_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(project_root)
        )
        
        # Send G-Assist command
        command = {"tool_calls": [{"func": "detect_hardware", "params": {}}]}
        command_json = json.dumps(command)
        
        print(f"📤 Command: {command_json}")
        print(f"📤 Mode: G-Assist (stdin/stdout)")
        
        # Send command and close stdin
        stdout, stderr = process.communicate(input=command_json + "\n", timeout=15)
        
        print(f"📥 Exit Code: {process.returncode}")
        print(f"📥 stdout: {repr(stdout)}")
        print(f"📥 stderr: {repr(stderr[:500])}..." if len(stderr) > 500 else f"📥 stderr: {repr(stderr)}")
        
        # Check for <<END>> marker
        if stdout.endswith('<<END>>'):
            json_part = stdout[:-7]  # Remove <<END>>
            try:
                response = json.loads(json_part)
                print("✅ G-ASSIST MODE: JSON parsing successful")
                print(f"   Success: {response.get('success')}")
                print(f"   Message length: {len(response.get('message', ''))}")
                print("✅ G-ASSIST MODE: <<END>> termination marker present")
            except json.JSONDecodeError as e:
                print(f"❌ G-ASSIST MODE: JSON parsing failed: {e}")
        else:
            print("❌ G-ASSIST MODE: Missing <<END>> termination marker")
            
    except subprocess.TimeoutExpired:
        print("❌ G-ASSIST MODE: Timeout")
        process.kill()
    except Exception as e:
        print(f"❌ G-ASSIST MODE: Test failed: {e}")

def test_g_assist_mode_detection():
    """Test G-Assist mode detection logic."""
    print("\n🧪 Testing G-Assist Mode Detection")
    print("=" * 40)
    
    plugin_path = project_root / "plugin.py"
    
    # Test 1: No arguments (should trigger G-Assist mode)
    print("🔍 Test 1: No arguments")
    try:
        process = subprocess.Popen(
            [sys.executable, str(plugin_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(project_root)
        )
        
        # Just send EOF to see mode detection
        stdout, stderr = process.communicate(input="", timeout=5)
        
        if "G-Assist mode detected" in stderr:
            print("✅ G-Assist mode correctly detected")
        else:
            print("❌ G-Assist mode not detected")
            print(f"   stderr: {stderr[:200]}...")
            
    except Exception as e:
        print(f"❌ Mode detection test failed: {e}")

def compare_workflows():
    """Compare the two workflows side by side."""
    print("\n📊 Workflow Comparison Summary")
    print("=" * 50)
    
    print("TERMINAL MODE:")
    print("  ✅ Uses CLI arguments (--function)")
    print("  ✅ Outputs clean JSON to stdout")
    print("  ✅ No termination markers")
    print("  ✅ Synchronous execution")
    print("  ✅ Direct function call")
    
    print("\nG-ASSIST MODE:")
    print("  ✅ Uses stdin/stdout communication")
    print("  ✅ Outputs JSON + <<END>> to stdout")
    print("  ✅ Async event loop")
    print("  ✅ Tool call processing")
    print("  ❓ May have buffering issues")
    print("  ❓ May have timing issues")

if __name__ == "__main__":
    test_terminal_workflow()
    test_g_assist_workflow()
    test_g_assist_mode_detection()
    compare_workflows()