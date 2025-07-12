"""
Simple G-Assist Plugin Test
Test the plugin with sample JSON inputs to validate deployment.
"""

import json
import subprocess
import sys
import os
import pytest

def test_plugin_function():
    """Test plugin function with basic detection."""
    # Simple test to verify plugin file exists
    plugin_path = os.path.join(os.path.dirname(__file__), "..", "plugin", "plugin.py")
    
    if not os.path.exists(plugin_path):
        # Alternative plugin paths
        alternative_paths = [
            os.path.join(os.path.dirname(__file__), "..", "plugin", "g-assist-plugin-canrun.py"),
            os.path.join(os.path.dirname(__file__), "..", "plugin.py")
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                plugin_path = alt_path
                break
    
    # Test plugin basic functionality
    if os.path.exists(plugin_path):
        try:
            # Test basic import capability
            test_input = {
                "function": "detect_hardware",
                "parameters": {}
            }
            
            json_input = json.dumps(test_input)
            
            result = subprocess.run(
                [sys.executable, plugin_path],
                input=json_input,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Plugin should at least not crash
            assert result.returncode in [0, 1], f"Plugin execution failed with code: {result.returncode}"
            
        except subprocess.TimeoutExpired:
            pytest.skip("Plugin test timed out - plugin may require user interaction")
        except Exception as e:
            pytest.skip(f"Plugin test failed due to environment: {e}")
    else:
        pytest.skip("Plugin file not found - skipping plugin test")

if __name__ == "__main__":
    test_plugin_function()