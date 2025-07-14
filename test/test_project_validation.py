"""
Project Validation Tests - Test project structure, data files, and configurations
without requiring full build or complex dependencies
"""

import json
import os
import sys
from pathlib import Path
import re

def test_json_data_files():
    """Test that all JSON data files are valid and well-structured"""
    print("\n" + "="*50)
    print("TEST 2: JSON Data File Validation")
    print("="*50)
    
    json_files = [
        "manifest.json",
        "data/game_requirements.json", 
        "data/gpu_hierarchy.json"
    ]
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ {json_file} - Valid JSON ({len(json.dumps(data))} bytes)")
        except Exception as e:
            print(f"✗ {json_file} - Invalid JSON: {e}")
            return False
    
    return True

def test_manifest_structure():
    """Test the G-Assist plugin manifest structure"""
    print("\n" + "="*50)
    print("TEST 3: Plugin Manifest Structure")
    print("="*50)
    
    try:
        with open("manifest.json", 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        # Required fields
        required_fields = ["name", "version", "description", "functions"]
        for field in required_fields:
            if field not in manifest:
                print(f"✗ Missing required field: {field}")
                return False
        
        # Validate functions
        if not isinstance(manifest["functions"], list):
            print("✗ Functions field must be an array")
            return False
        
        expected_functions = [
            "detect_hardware",
            "check_compatibility", 
            "predict_performance",
            "get_optimization_suggestions"
        ]
        
        function_names = [f["name"] for f in manifest["functions"]]
        for func_name in expected_functions:
            if func_name not in function_names:
                print(f"✗ Missing function: {func_name}")
                return False
        
        print("✓ Manifest structure is valid")
        print(f"Plugin: {manifest['name']} v{manifest['version']}")
        print(f"Functions: {len(manifest['functions'])}")
        return True
        
    except Exception as e:
        print(f"✗ Manifest validation failed: {e}")
        return False

def test_game_requirements_data():
    """Test game requirements data structure"""
    print("\n" + "="*50)
    print("TEST 4: Game Requirements Data")
    print("="*50)
    
    try:
        with open("data/game_requirements.json", 'r', encoding='utf-8') as f:
            games_data = json.load(f)
        
        if not isinstance(games_data, dict):
            print("✗ Game requirements must be a dictionary")
            return False
        
        game_count = 0
        
        # Handle both direct game entries and nested "games" structure
        games_to_check = games_data
        if "games" in games_data:
            games_to_check = games_data["games"]
        
        for game_name, game_data in games_to_check.items():
            # Skip comment fields
            if game_name == "comment" or isinstance(game_data, str):
                continue
                
            game_count += 1
            
            # Check required sections
            required_sections = ["minimum", "recommended"]
            for section in required_sections:
                if section not in game_data:
                    print(f"✗ Game '{game_name}' missing section: {section}")
                    return False
                
                # Check required components (using actual field names from data)
                required_components = ["processor", "graphics", "memory", "storage"]
                for component in required_components:
                    if component not in game_data[section]:
                        print(f"✗ Game '{game_name}' {section} missing: {component}")
                        return False
        
        print(f"✓ Game requirements data is valid")
        print(f"Games in database: {game_count}")
        return True
        
    except Exception as e:
        print(f"✗ Game requirements validation failed: {e}")
        return False

def test_gpu_hierarchy_data():
    """Test GPU hierarchy data structure"""
    print("\n" + "="*50)
    print("TEST 5: GPU Hierarchy Data")
    print("="*50)
    
    try:
        with open("data/gpu_hierarchy.json", 'r', encoding='utf-8') as f:
            gpu_data = json.load(f)
        
        if not isinstance(gpu_data, dict):
            print("✗ GPU hierarchy must be a dictionary")
            return False
        
        gpu_count = 0
        tiers = set()
        
        # Handle both direct GPU entries and nested "nvidia" structure
        gpus_to_check = gpu_data
        if "nvidia" in gpu_data:
            gpus_to_check = gpu_data["nvidia"]
        
        for gpu_name, gpu_info in gpus_to_check.items():
            # Skip comment fields
            if gpu_name == "comment" or isinstance(gpu_info, str):
                continue
                
            gpu_count += 1
            
            # Check required fields
            required_fields = ["score", "tier", "memory", "features"]
            for field in required_fields:
                if field not in gpu_info:
                    print(f"✗ GPU '{gpu_name}' missing field: {field}")
                    return False
            
            # Collect tier information
            tiers.add(gpu_info["tier"])
            
            # Validate score is numeric
            if not isinstance(gpu_info["score"], (int, float)):
                print(f"✗ GPU '{gpu_name}' score must be numeric")
                return False
        
        print(f"✓ GPU hierarchy data is valid")
        print(f"GPUs in database: {gpu_count}")
        print(f"Performance tiers: {', '.join(sorted(tiers))}")
        return True
        
    except Exception as e:
        print(f"✗ GPU hierarchy validation failed: {e}")
        return False

def test_python_syntax():
    """Test that all Python files have valid syntax"""
    print("\n" + "="*50)
    print("TEST 6: Python Syntax Validation")
    print("="*50)
    
    python_files = [
        "src/setup.py",
        "src/build.py",
        "plugin.py",
        "src/__init__.py",
        "src/canrun_engine.py",
        "src/hardware_detector.py",
        "src/game_requirements_fetcher.py",
        "src/compatibility_analyzer.py",
        "src/performance_predictor.py"
    ]
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to compile the source code
            compile(content, py_file, 'exec')
            print(f"✓ {py_file} - Valid Python syntax")
            
        except SyntaxError as e:
            print(f"✗ {py_file} - Syntax error: {e}")
            return False
        except Exception as e:
            print(f"✗ {py_file} - Error: {e}")
            return False
    
    return True

def test_requirements_txt():
    """Test requirements.txt file format"""
    print("\n" + "="*50)
    print("TEST 7: Requirements.txt Validation")
    print("="*50)
    
    try:
        with open("requirements.txt", 'r', encoding='utf-8') as f:
            requirements = f.read().strip().split('\n')
        
        valid_requirements = []
        for req in requirements:
            req = req.strip()
            if req and not req.startswith('#'):
                # Basic validation of requirement format
                if '>=' in req or '==' in req or '>' in req or '<' in req:
                    valid_requirements.append(req)
                elif req.replace('-', '').replace('_', '').isalnum():
                    valid_requirements.append(req)
                else:
                    print(f"✗ Invalid requirement format: {req}")
                    return False
        
        print(f"✓ Requirements.txt is valid")
        print(f"Dependencies: {len(valid_requirements)}")
        for req in valid_requirements:
            print(f"  - {req}")
        return True
        
    except Exception as e:
        print(f"✗ Requirements.txt validation failed: {e}")
        return False

def test_documentation_completeness():
    """Test that documentation is complete and references are valid"""
    print("\n" + "="*50)
    print("TEST 8: Documentation Completeness")
    print("="*50)
    
    docs_to_check = [
        "README.md",
        "CONTRIBUTING.md", 
        "CHANGELOG.md"
    ]
    
    for doc in docs_to_check:
        try:
            with open(doc, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for basic structure
            if len(content) < 500:  # Minimum reasonable length
                print(f"✗ {doc} - Too short (less than 500 characters)")
                return False
            
            # Check for code blocks or examples
            if '```' in content or '`' in content:
                print(f"✓ {doc} - Contains code examples")
            else:
                print(f"⚠ {doc} - No code examples found")
            
        except Exception as e:
            print(f"✗ {doc} - Error reading file: {e}")
            return False
    
    return True

def test_command_line_args():
    """Test command-line argument structure"""
    print("\n" + "="*50)
    print("TEST 9: Command-Line Arguments")
    print("="*50)
    
    try:
        with open("plugin.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for argument parsing patterns
        if 'argparse' in content or '--function' in content or 'sys.argv' in content:
            print("✓ Command-line argument parsing found")
        else:
            print("⚠ No command-line argument parsing detected")
        
        # Check for main function
        if 'if __name__ == "__main__"' in content:
            print("✓ Main execution block found")
        else:
            print("⚠ No main execution block found")
        
        return True
        
    except Exception as e:
        print(f"✗ Command-line argument test failed: {e}")
        return False

def run_all_validation_tests():
    """Run all validation tests"""
    print("CanRun Project Validation Tests")
    print("="*50)
    
    tests = [
        test_json_data_files,
        test_manifest_structure,
        test_game_requirements_data,
        test_gpu_hierarchy_data,
        test_python_syntax,
        test_requirements_txt,
        test_documentation_completeness,
        test_command_line_args
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All validation tests passed!")
        print("Project structure and data files are valid.")
        print("The project is ready for deployment.")
        return True
    else:
        print(f"\n❌ {failed} validation test(s) failed.")
        print("Please review the issues above before deployment.")
        return False

if __name__ == "__main__":
    success = run_all_validation_tests()
    sys.exit(0 if success else 1)