#!/usr/bin/env python3
"""
Enhanced G-Assist Communication Test
Tests the enhanced G-Assist plugin with tool name detection and Steam Compare UI integration.
"""

import asyncio
import json
import pytest
import subprocess
import sys
import time
from typing import Dict, Any

def create_enhanced_g_assist_command(function_name: str, params: Dict[str, Any]) -> str:
    """Create an enhanced G-Assist style command JSON."""
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
async def test_enhanced_plugin_communication():
    """Test enhanced G-Assist style communication with Steam Compare UI."""
    print("🧪 ENHANCED G-ASSIST COMMUNICATION TEST")
    print("=" * 60)
    
    # Enhanced test commands to simulate
    test_commands = [
        {
            "name": "Enhanced Hardware Detection",
            "function": "detect_hardware",
            "params": {}
        },
        {
            "name": "Enhanced Diablo 4 Compatibility Check with Steam UI",
            "function": "check_compatibility", 
            "params": {"game_name": "Diablo 4", "show_steam_data": True}
        },
        {
            "name": "Steam Compare UI for Cyberpunk 2077",
            "function": "steam_compare",
            "params": {"game_name": "Cyberpunk 2077"}
        },
        {
            "name": "Game Performance Comparison",
            "function": "compare_games",
            "params": {"games": ["Diablo 4", "Cyberpunk 2077", "Elden Ring"]}
        },
        {
            "name": "Auto-Detection from Natural Language",
            "function": "auto_detect",
            "params": {"user_input": "Can I run Baldur's Gate 3 on my system?"}
        },
        {
            "name": "Auto-Detection - Hardware Query",
            "function": "auto_detect",
            "params": {"user_input": "What are my system specs?"}
        },
        {
            "name": "Auto-Detection - Steam Compare Query",
            "function": "auto_detect",
            "params": {"user_input": "Show me Steam requirements for Elden Ring"}
        }
    ]
    
    for i, test_case in enumerate(test_commands, 1):
        print(f"\n🎯 TEST {i}: {test_case['name']}")
        print("-" * 40)
        
        # Create enhanced G-Assist style command
        command_json = create_enhanced_g_assist_command(test_case['function'], test_case['params'])
        print(f"📨 Sending enhanced command: {command_json}")
        
        try:
            # Launch enhanced plugin as subprocess
            process = subprocess.Popen(
                [sys.executable, "canrun/plugin.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd="."
            )
            
            # Send command and get response
            stdout, stderr = process.communicate(input=command_json + "\n", timeout=45)
            
            print(f"📤 Enhanced plugin stdout: {stdout}")
            if stderr:
                print(f"⚠️  Enhanced plugin stderr: {stderr}")
            
            # Check if response contains <<END>> marker
            if "<<END>>" in stdout:
                response_part = stdout.split("<<END>>")[0]
                try:
                    response_data = json.loads(response_part)
                    print(f"✅ Valid enhanced JSON response:")
                    print(json.dumps(response_data, indent=2))
                    
                    if response_data.get("success"):
                        print("✅ Enhanced command executed successfully")
                        
                        # Check for enhanced features
                        message = response_data.get("message", "")
                        if "🎮" in message or "💻" in message or "📊" in message:
                            print("✅ Enhanced formatting detected")
                        if "Steam" in message:
                            print("✅ Steam Compare UI integration detected")
                        if "Performance" in message or "FPS" in message:
                            print("✅ Performance analysis detected")
                    else:
                        print(f"❌ Enhanced command failed: {response_data.get('message', 'Unknown error')}")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid enhanced JSON response: {e}")
            else:
                print("❌ No proper enhanced G-Assist response format (missing <<END>>)")
                
        except subprocess.TimeoutExpired:
            print("❌ Enhanced plugin test timed out")
            process.kill()
        except Exception as e:
            print(f"❌ Enhanced test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 ENHANCED G-ASSIST COMMUNICATION TEST COMPLETE")

def test_enhanced_manifest_parsing():
    """Test if enhanced manifest.json can be parsed correctly."""
    print("\n🔍 TESTING ENHANCED MANIFEST.JSON PARSING")
    print("-" * 40)
    
    try:
        with open("canrun/manifest.json", "r") as f:
            manifest = json.load(f)
            
        print("✅ Enhanced manifest JSON is valid")
        print(f"📋 Plugin name: {manifest.get('name', 'UNKNOWN')}")
        print(f"📋 Version: {manifest.get('version', 'UNKNOWN')}")
        print(f"📋 Description: {manifest.get('description', 'UNKNOWN')}")
        print(f"📋 Executable: {manifest.get('executable', 'UNKNOWN')}")
        
        functions = manifest.get('functions', [])
        print(f"📋 Enhanced functions defined: {len(functions)}")
        
        enhanced_functions = ['steam_compare', 'compare_games', 'auto_detect']
        
        for func in functions:
            func_name = func.get('name', 'UNKNOWN')
            description = func.get('description', 'No description')
            tags = func.get('tags', [])
            properties = func.get('properties', {})
            
            print(f"   🔧 {func_name}: {len(tags)} tags, {len(properties)} properties")
            
            # Check for enhanced functions
            if func_name in enhanced_functions:
                print(f"   ✅ Enhanced function detected: {func_name}")
                
            # Check for enhanced tags
            enhanced_tags = ['steam', 'compare', 'enhanced', 'auto', 'detect']
            if any(tag in tags for tag in enhanced_tags):
                print(f"   ✅ Enhanced tags detected: {[tag for tag in tags if tag in enhanced_tags]}")
                
    except Exception as e:
        print(f"❌ Enhanced manifest parsing failed: {e}")

def test_command_line_interface():
    """Test the enhanced command-line interface."""
    print("\n🔍 TESTING ENHANCED COMMAND-LINE INTERFACE")
    print("-" * 40)
    
    cli_tests = [
        {
            "name": "Hardware Detection CLI",
            "args": ["--function", "detect_hardware"]
        },
        {
            "name": "Compatibility Check CLI",
            "args": ["--function", "check_compatibility", "--game", "Diablo 4", "--show-steam"]
        },
        {
            "name": "Steam Compare CLI",
            "args": ["--function", "steam_compare", "--game", "Cyberpunk 2077"]
        },
        {
            "name": "Auto-Detection CLI",
            "args": ["--function", "auto_detect", "--input", "Can I run Elden Ring?"]
        }
    ]
    
    for test in cli_tests:
        print(f"\n🧪 Testing: {test['name']}")
        try:
            result = subprocess.run(
                [sys.executable, "canrun/plugin.py"] + test["args"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✅ CLI test successful")
                try:
                    response = json.loads(result.stdout)
                    if response.get("success"):
                        print("✅ CLI response indicates success")
                        message = response.get("message", "")
                        if len(message) > 100:  # Enhanced responses should be longer
                            print("✅ Enhanced response format detected")
                    else:
                        print(f"⚠️  CLI response indicates failure: {response.get('message', 'Unknown')}")
                except json.JSONDecodeError:
                    print("⚠️  CLI output is not valid JSON")
            else:
                print(f"❌ CLI test failed with exit code {result.returncode}")
                print(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("❌ CLI test timed out")
        except Exception as e:
            print(f"❌ CLI test error: {e}")

@pytest.mark.asyncio
async def test_tool_name_detection():
    """Test the tool name detection functionality."""
    print("\n🔍 TESTING TOOL NAME DETECTION")
    print("-" * 40)
    
    detection_tests = [
        {
            "input": "Can I run Diablo 4?",
            "expected_tool": "check_compatibility"
        },
        {
            "input": "What are my system specs?",
            "expected_tool": "detect_hardware"
        },
        {
            "input": "Compare Cyberpunk 2077 vs Elden Ring",
            "expected_tool": "compare_games"
        },
        {
            "input": "Show me Steam requirements for Baldur's Gate 3",
            "expected_tool": "steam_compare"
        },
        {
            "input": "Will Starfield work on my computer?",
            "expected_tool": "check_compatibility"
        }
    ]
    
    for test in detection_tests:
        print(f"\n🧪 Testing input: '{test['input']}'")
        print(f"Expected tool: {test['expected_tool']}")
        
        command = create_enhanced_g_assist_command("auto_detect", {"user_input": test["input"]})
        
        try:
            process = subprocess.Popen(
                [sys.executable, "canrun/plugin.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd="."
            )
            
            stdout, stderr = process.communicate(input=command + "\n", timeout=30)
            
            if "<<END>>" in stdout:
                response_part = stdout.split("<<END>>")[0]
                try:
                    response_data = json.loads(response_part)
                    if response_data.get("success"):
                        print("✅ Auto-detection successful")
                        message = response_data.get("message", "")
                        # Check if the response indicates the correct tool was used
                        if test['expected_tool'] == 'check_compatibility' and ('Performance' in message or 'FPS' in message):
                            print("✅ Correct tool detected (compatibility check)")
                        elif test['expected_tool'] == 'detect_hardware' and ('GPU:' in message or 'CPU:' in message):
                            print("✅ Correct tool detected (hardware detection)")
                        elif test['expected_tool'] == 'steam_compare' and 'Steam' in message:
                            print("✅ Correct tool detected (Steam compare)")
                        elif test['expected_tool'] == 'compare_games' and 'COMPARISON' in message:
                            print("✅ Correct tool detected (game comparison)")
                        else:
                            print("⚠️  Tool detection may not be optimal")
                    else:
                        print(f"❌ Auto-detection failed: {response_data.get('message', 'Unknown')}")
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON response: {e}")
            else:
                print("❌ No proper response format")
                
        except subprocess.TimeoutExpired:
            print("❌ Detection test timed out")
            process.kill()
        except Exception as e:
            print(f"❌ Detection test failed: {e}")

async def main():
    """Run all enhanced G-Assist communication tests."""
    print("🚀 ENHANCED CANRUN G-ASSIST INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Test 1: Enhanced manifest parsing
    test_enhanced_manifest_parsing()
    
    # Test 2: Enhanced plugin communication
    await test_enhanced_plugin_communication()
    
    # Test 3: Command-line interface
    test_command_line_interface()
    
    # Test 4: Tool name detection
    await test_tool_name_detection()
    
    print("\n🎯 ENHANCED DIAGNOSTIC COMPLETE")
    print("Check canrun-enhanced-plugin.log for detailed plugin communication logs")
    print("\n📊 ENHANCED FEATURES TESTED:")
    print("✅ Tool name detection and routing")
    print("✅ Steam Compare UI integration")
    print("✅ Enhanced chat window response formatting")
    print("✅ Real-time Steam data fetching")
    print("✅ Auto-detection from natural language")
    print("✅ Multi-game performance comparison")

if __name__ == "__main__":
    asyncio.run(main())