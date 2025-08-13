#!/usr/bin/env python3
"""
G-Assist Command Simulation Test
Simulates G-Assist sending commands to our plugin to verify communication protocol.
"""

import asyncio
import json
import pytest
import subprocess
import sys
import time
from typing import Dict, Any

def create_g_assist_command(function_name: str, params: Dict[str, Any]) -> str:
    """Create a G-Assist style command JSON."""
    command = {
        "tool_calls": [
            {
                "func": function_name,
                "params": params
            }
        ]
    }
    return json.dumps(command)

@pytest.mark.asyncio
async def test_plugin_communication():
    """Test G-Assist style communication with our plugin."""
    print("🧪 G-ASSIST COMMAND SIMULATION TEST")
    print("=" * 60)
    
    # Test commands to simulate
    test_commands = [
        {
            "name": "Hardware Detection",
            "function": "detect_hardware",
            "params": {}
        },
        {
            "name": "Diablo 4 Compatibility Check",
            "function": "check_compatibility", 
            "params": {"game_name": "Diablo 4"}
        },
        {
            "name": "Cyberpunk 2077 Compatibility Check",
            "function": "check_compatibility",
            "params": {"game_name": "Cyberpunk 2077"}
        },
        {
            "name": "Performance Prediction",
            "function": "predict_performance",
            "params": {"game_name": "Elden Ring", "settings": "High", "resolution": "1440p"}
        }
    ]
    
    for i, test_case in enumerate(test_commands, 1):
        print(f"\n🎯 TEST {i}: {test_case['name']}")
        print("-" * 40)
        
        # Create G-Assist style command
        command_json = create_g_assist_command(test_case['function'], test_case['params'])
        print(f"📨 Sending command: {command_json}")
        
        try:
            # Launch plugin as subprocess
            process = subprocess.Popen(
                [sys.executable, "canrun/plugin.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd="."
            )
            
            # Send command and get response
            stdout, stderr = process.communicate(input=command_json + "\n", timeout=30)
            
            print(f"📤 Plugin stdout: {stdout}")
            if stderr:
                print(f"⚠️  Plugin stderr: {stderr}")
            
            # Check if response contains <<END>> marker
            if "<<END>>" in stdout:
                response_part = stdout.split("<<END>>")[0]
                try:
                    response_data = json.loads(response_part)
                    print(f"✅ Valid JSON response: {json.dumps(response_data, indent=2)}")
                    
                    if response_data.get("success"):
                        print("✅ Command executed successfully")
                    else:
                        print(f"❌ Command failed: {response_data.get('message', 'Unknown error')}")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON response: {e}")
            else:
                print("❌ No proper G-Assist response format (missing <<END>>)")
                
        except subprocess.TimeoutExpired:
            print("❌ Plugin test timed out")
            process.kill()
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 G-ASSIST SIMULATION TEST COMPLETE")

def test_manifest_parsing():
    """Test if manifest.json can be parsed correctly."""
    print("\n🔍 TESTING MANIFEST.JSON PARSING")
    print("-" * 40)
    
    try:
        with open("canrun/manifest.json", "r") as f:
            manifest = json.load(f)
            
        print("✅ Manifest JSON is valid")
        print(f"📋 Plugin name: {manifest.get('name', 'UNKNOWN')}")
        print(f"📋 Version: {manifest.get('version', 'UNKNOWN')}")
        print(f"📋 Executable: {manifest.get('executable', 'UNKNOWN')}")
        
        functions = manifest.get('functions', [])
        print(f"📋 Functions defined: {len(functions)}")
        
        for func in functions:
            func_name = func.get('name', 'UNKNOWN')
            description = func.get('description', 'No description')
            tags = func.get('tags', [])
            print(f"   🔧 {func_name}: {len(tags)} tags")
            
            # Check for compatibility function specifically
            if func_name == "check_compatibility":
                print(f"   🎯 check_compatibility description length: {len(description)} chars")
                print(f"   🎯 check_compatibility tags: {tags}")
                
    except Exception as e:
        print(f"❌ Manifest parsing failed: {e}")

def test_executable_exists():
    """Test if the plugin executable exists and is accessible."""
    print("\n🔍 TESTING PLUGIN EXECUTABLE")
    print("-" * 40)
    
    import os
    
    executable_path = "./canrun/g-assist-plugin-canrun.exe"
    if os.path.exists(executable_path):
        print(f"✅ Executable found: {executable_path}")
        print(f"📦 File size: {os.path.getsize(executable_path)} bytes")
        
        # Test if executable can be launched
        try:
            result = subprocess.run(
                [executable_path, "--function", "detect_hardware"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print("✅ Executable runs successfully")
                print(f"📤 Output preview: {result.stdout[:200]}...")
            else:
                print(f"❌ Executable failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Executable test failed: {e}")
    else:
        print(f"❌ Executable not found: {executable_path}")

async def main():
    """Run all G-Assist simulation tests."""
    print("🚀 CANRUN G-ASSIST INTEGRATION DIAGNOSTIC SUITE")
    print("=" * 60)
    
    # Test 1: Manifest parsing
    test_manifest_parsing()
    
    # Test 2: Executable existence
    test_executable_exists()
    
    # Test 3: Plugin communication simulation
    await test_plugin_communication()
    
    print("\n🎯 DIAGNOSTIC COMPLETE")
    print("Check canrun_plugin.log for detailed plugin communication logs")

if __name__ == "__main__":
    asyncio.run(main())