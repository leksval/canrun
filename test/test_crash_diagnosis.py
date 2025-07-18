#!/usr/bin/env python3
"""
Comprehensive crash diagnosis for CanRun G-Assist Plugin
"""
import subprocess
import sys
import json
import traceback
import time
import os

def test_basic_execution():
    """Test basic plugin execution"""
    print("🔍 Testing Basic Plugin Execution")
    print("=" * 50)
    
    try:
        # Test CLI mode
        result = subprocess.run(
            [sys.executable, "plugin.py", "--function", "detect_hardware"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd="."
        )
        
        print(f"✅ CLI Mode - Exit Code: {result.returncode}")
        if result.returncode != 0:
            print(f"❌ STDERR: {result.stderr}")
            print(f"❌ STDOUT: {result.stdout}")
        else:
            print("✅ CLI Mode working correctly")
            
    except Exception as e:
        print(f"❌ CLI Mode Exception: {e}")
        traceback.print_exc()

def test_executable_execution():
    """Test compiled executable"""
    print("\n🔍 Testing Compiled Executable")
    print("=" * 50)
    
    try:
        # Test executable CLI mode
        result = subprocess.run(
            ["./plugin.exe", "--function", "detect_hardware"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd="."
        )
        
        print(f"✅ Executable CLI Mode - Exit Code: {result.returncode}")
        if result.returncode != 0:
            print(f"❌ STDERR: {result.stderr}")
            print(f"❌ STDOUT: {result.stdout}")
        else:
            print("✅ Executable CLI Mode working correctly")
            
    except Exception as e:
        print(f"❌ Executable CLI Mode Exception: {e}")
        traceback.print_exc()

def test_g_assist_mode():
    """Test G-Assist mode communication"""
    print("\n🔍 Testing G-Assist Mode Communication")
    print("=" * 50)
    
    try:
        # Test G-Assist mode with Python script
        process = subprocess.Popen(
            [sys.executable, "plugin.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="."
        )
        
        test_command = {"tool_calls": [{"func": "detect_hardware", "params": {}}]}
        stdout, stderr = process.communicate(input=json.dumps(test_command) + "\n", timeout=30)
        
        print(f"✅ G-Assist Mode - Exit Code: {process.returncode}")
        if process.returncode != 0:
            print(f"❌ STDERR: {stderr}")
            print(f"❌ STDOUT: {stdout}")
        else:
            print("✅ G-Assist Mode working correctly")
            if "<<END>>" in stdout:
                print("✅ Proper G-Assist protocol termination found")
            else:
                print("⚠️ Missing <<END>> termination marker")
                
    except Exception as e:
        print(f"❌ G-Assist Mode Exception: {e}")
        traceback.print_exc()

def test_executable_g_assist_mode():
    """Test G-Assist mode with executable"""
    print("\n🔍 Testing Executable G-Assist Mode")
    print("=" * 50)
    
    try:
        # Test G-Assist mode with executable
        process = subprocess.Popen(
            ["./plugin.exe"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="."
        )
        
        test_command = {"tool_calls": [{"func": "detect_hardware", "params": {}}]}
        stdout, stderr = process.communicate(input=json.dumps(test_command) + "\n", timeout=30)
        
        print(f"✅ Executable G-Assist Mode - Exit Code: {process.returncode}")
        if process.returncode != 0:
            print(f"❌ STDERR: {stderr}")
            print(f"❌ STDOUT: {stdout}")
        else:
            print("✅ Executable G-Assist Mode working correctly")
            if "<<END>>" in stdout:
                print("✅ Proper G-Assist protocol termination found")
            else:
                print("⚠️ Missing <<END>> termination marker")
                
    except Exception as e:
        print(f"❌ Executable G-Assist Mode Exception: {e}")
        traceback.print_exc()

def test_game_compatibility():
    """Test game compatibility check"""
    print("\n🔍 Testing Game Compatibility Check")
    print("=" * 50)
    
    try:
        # Test game compatibility
        result = subprocess.run(
            [sys.executable, "plugin.py", "--function", "check_compatibility", "--game", "Diablo 4"],
            capture_output=True,
            text=True,
            timeout=60,  # Longer timeout for Steam API
            cwd="."
        )
        
        print(f"✅ Game Compatibility - Exit Code: {result.returncode}")
        if result.returncode != 0:
            print(f"❌ STDERR: {result.stderr}")
            print(f"❌ STDOUT: {result.stdout}")
        else:
            print("✅ Game Compatibility working correctly")
            # Check for Steam data
            if "STEAM REQUIREMENTS" in result.stdout:
                print("✅ Steam data integration working")
            else:
                print("⚠️ Steam data might be missing")
                
    except Exception as e:
        print(f"❌ Game Compatibility Exception: {e}")
        traceback.print_exc()

def test_memory_usage():
    """Test for memory leaks or excessive usage"""
    print("\n🔍 Testing Memory Usage")
    print("=" * 50)
    
    try:
        import psutil
        
        # Start process and monitor memory
        process = subprocess.Popen(
            [sys.executable, "plugin.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="."
        )
        
        # Monitor memory for a few seconds
        time.sleep(2)
        
        try:
            proc = psutil.Process(process.pid)
            memory_info = proc.memory_info()
            print(f"✅ Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")
            
            if memory_info.rss > 500 * 1024 * 1024:  # 500MB
                print("⚠️ High memory usage detected")
            else:
                print("✅ Memory usage normal")
                
        except psutil.NoSuchProcess:
            print("❌ Process terminated unexpectedly")
            
        # Clean up
        process.terminate()
        process.wait(timeout=5)
        
    except ImportError:
        print("⚠️ psutil not available, skipping memory test")
    except Exception as e:
        print(f"❌ Memory Test Exception: {e}")
        traceback.print_exc()

def check_dependencies():
    """Check for missing dependencies"""
    print("\n🔍 Checking Dependencies")
    print("=" * 50)
    
    required_modules = [
        'pynvml', 'psutil', 'cpuinfo', 'requests', 
        'beautifulsoup4', 'fuzzywuzzy', 'python-Levenshtein'
    ]
    
    for module in required_modules:
        try:
            __import__(module.replace('-', '_'))
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - MISSING")

def main():
    """Run comprehensive crash diagnosis"""
    print("🚨 CanRun G-Assist Plugin Crash Diagnosis")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Plugin File Exists: {os.path.exists('plugin.py')}")
    print(f"Executable Exists: {os.path.exists('plugin.exe')}")
    print("=" * 60)
    
    check_dependencies()
    test_basic_execution()
    test_executable_execution()
    test_g_assist_mode()
    test_executable_g_assist_mode()
    test_game_compatibility()
    test_memory_usage()
    
    print("\n🏁 Crash Diagnosis Complete")
    print("=" * 60)
    print("If any tests failed above, that indicates the source of the crash.")
    print("Please share the specific error messages for further debugging.")

if __name__ == "__main__":
    main()