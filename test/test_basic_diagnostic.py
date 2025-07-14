#!/usr/bin/env python3
"""
Basic diagnostic test for G-Assist plugin components.
"""

import json
import os

def test_manifest():
    """Test manifest.json parsing."""
    print('🔍 TESTING MANIFEST.JSON PARSING')
    print('-' * 40)

    try:
        with open('manifest.json', 'r') as f:
            manifest = json.load(f)
        print('✅ Manifest JSON is valid')
        print(f'📋 Plugin name: {manifest.get("name", "UNKNOWN")}')
        print(f'📋 Executable: {manifest.get("executable", "UNKNOWN")}')
        
        functions = manifest.get('functions', [])
        print(f'📋 Functions defined: {len(functions)}')
        
        for func in functions:
            func_name = func.get('name', 'UNKNOWN')
            if func_name == 'check_compatibility':
                tags = func.get('tags', [])
                print(f'   🎯 check_compatibility tags: {tags}')
                
    except Exception as e:
        print(f'❌ Manifest parsing failed: {e}')

def test_executable():
    """Test if executable exists."""
    print()
    print('🔍 TESTING PLUGIN EXECUTABLE')
    print('-' * 40)

    executable_path = './g-assist-plugin-python.exe'
    if os.path.exists(executable_path):
        print(f'✅ Executable found: {executable_path}')
        print(f'📦 File size: {os.path.getsize(executable_path)} bytes')
    else:
        print(f'❌ Executable not found: {executable_path}')

def test_plugin_quick():
    """Test plugin with a simple command line argument."""
    print()
    print('🔍 TESTING PLUGIN QUICK COMMAND')
    print('-' * 40)
    
    import subprocess
    
    try:
        # Test with simple hardware detection command
        result = subprocess.run(
            ['python', 'plugin.py', '--function', 'detect_hardware'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print('✅ Plugin runs successfully with command line args')
            print(f'📤 Output preview: {result.stdout[:200]}...')
        else:
            print(f'❌ Plugin failed: {result.stderr}')
            
    except Exception as e:
        print(f'❌ Plugin test failed: {e}')

if __name__ == "__main__":
    print('🚀 BASIC DIAGNOSTIC TEST')
    print('=' * 50)
    
    test_manifest()
    test_executable()
    test_plugin_quick()
    
    print()
    print('🎯 BASIC DIAGNOSTIC COMPLETE')