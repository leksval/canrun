#!/usr/bin/env python3
"""
CanRun Setup Script
Sets up the complete CanRun development and runtime environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

class CanRunSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.is_windows = platform.system() == "Windows"
        self.is_linux = platform.system() == "Linux"
        
    def print_header(self):
        """Print setup header"""
        print("=" * 60)
        print("CanRun Universal Game Compatibility Checker")
        print("Setup and Installation Script")
        print("=" * 60)
        print(f"Platform: {platform.system()}")
        print(f"Python Version: {self.python_version}")
        print(f"Project Root: {self.project_root}")
        print()
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("[1/8] Checking Python version...")
        
        if sys.version_info < (3, 8):
            print("❌ ERROR: Python 3.8 or higher is required")
            print(f"Current version: {sys.version}")
            return False
            
        print(f"✅ Python {self.python_version} detected")
        return True
        
    def check_dependencies(self):
        """Check and install dependencies"""
        print("[2/8] Checking dependencies...")
        
        # Check if pip is available
        try:
            import pip
            print("✅ pip is available")
        except ImportError:
            print("❌ ERROR: pip is not installed")
            return False
            
        # Install requirements
        requirements_file = self.project_root / "plugin" / "requirements.txt"
        if requirements_file.exists():
            print("Installing Python packages...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], check=True)
                print("✅ Dependencies installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ ERROR: Failed to install dependencies: {e}")
                return False
        else:
            print("⚠️  WARNING: requirements.txt not found")
            
        return True
        
    def create_directories(self):
        """Create necessary directories"""
        print("[3/8] Creating directories...")
        
        directories = [
            "logs",
            "cache",
            "config",
            "build",
            "dist"
        ]
        
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created/verified directory: {dir_name}")
            
        return True
        
    def setup_configuration(self):
        """Setup configuration files"""
        print("[4/8] Setting up configuration...")
        
        # Create config file if it doesn't exist
        config_file = self.project_root / "config" / "canrun.conf"
        if not config_file.exists():
            config_content = """[CanRun]
# CanRun Configuration File
version = 1.0.0
debug = false
log_level = info
cache_enabled = true
cache_duration = 3600

[API]
steam_api_enabled = true
steam_api_timeout = 10
max_retries = 3

[Performance]
prediction_accuracy = medium
dlss_detection = true
rtx_detection = true

[Logging]
log_to_file = true
log_file = logs/canrun.log
max_log_size = 10MB
"""
            config_file.write_text(config_content)
            print("✅ Created configuration file")
        else:
            print("✅ Configuration file already exists")
            
        return True
        
    def setup_environment(self):
        """Setup environment variables and paths"""
        print("[5/8] Setting up environment...")
        
        # Create environment setup script
        if self.is_windows:
            env_script = self.project_root / "set_env.bat"
            env_content = f"""@echo off
REM CanRun Environment Setup
set CANRUN_ROOT={self.project_root}
set CANRUN_CONFIG=%CANRUN_ROOT%\\config
set CANRUN_DATA=%CANRUN_ROOT%\\data
set CANRUN_LOGS=%CANRUN_ROOT%\\logs
set CANRUN_CACHE=%CANRUN_ROOT%\\cache
set PYTHONPATH=%CANRUN_ROOT%;%PYTHONPATH%
echo CanRun environment variables set
"""
        else:
            env_script = self.project_root / "set_env.sh"
            env_content = f"""#!/bin/bash
# CanRun Environment Setup
export CANRUN_ROOT={self.project_root}
export CANRUN_CONFIG=$CANRUN_ROOT/config
export CANRUN_DATA=$CANRUN_ROOT/data
export CANRUN_LOGS=$CANRUN_ROOT/logs
export CANRUN_CACHE=$CANRUN_ROOT/cache
export PYTHONPATH=$CANRUN_ROOT:$PYTHONPATH
echo "CanRun environment variables set"
"""
            
        env_script.write_text(env_content)
        if not self.is_windows:
            os.chmod(env_script, 0o755)
            
        print("✅ Environment setup script created")
        return True
        
    def validate_installation(self):
        """Validate the installation"""
        print("[6/8] Validating installation...")
        
        # Check if core modules can be imported
        try:
            sys.path.insert(0, str(self.project_root))
            
            from src.hardware_detector import HardwareDetector
            from src.requirements_fetcher import RequirementsFetcher
            from src.compatibility_analyzer import CompatibilityAnalyzer
            from src.performance_predictor import PerformancePredictor
            from src.canrun_engine import CanRunEngine
            
            print("✅ All core modules imported successfully")
            
            # Check data files
            data_files = [
                "data/game_requirements.json",
                "data/gpu_hierarchy.json"
            ]
            
            for data_file in data_files:
                file_path = self.project_root / data_file
                if file_path.exists():
                    print(f"✅ Data file found: {data_file}")
                else:
                    print(f"⚠️  Data file missing: {data_file}")
                    
        except ImportError as e:
            print(f"❌ ERROR: Failed to import modules: {e}")
            return False
            
        return True
        
    def create_shortcuts(self):
        """Create shortcuts and launcher scripts"""
        print("[7/8] Creating shortcuts...")
        
        # Create launcher script
        if self.is_windows:
            launcher = self.project_root / "run_canrun.bat"
            launcher_content = f"""@echo off
cd /d "{self.project_root}"
call set_env.bat
python plugin/g-assist-plugin-canrun.py %*
"""
        else:
            launcher = self.project_root / "run_canrun.sh"
            launcher_content = f"""#!/bin/bash
cd "{self.project_root}"
source ./set_env.sh
python3 plugin/g-assist-plugin-canrun.py "$@"
"""
            
        launcher.write_text(launcher_content)
        if not self.is_windows:
            os.chmod(launcher, 0o755)
            
        print("✅ Launcher script created")
        
        # Create test runner
        if self.is_windows:
            test_runner = self.project_root / "run_tests.bat"
            test_content = f"""@echo off
cd /d "{self.project_root}"
call set_env.bat
python test/run_all_tests.py
"""
        else:
            test_runner = self.project_root / "run_tests.sh"
            test_content = f"""#!/bin/bash
cd "{self.project_root}"
source ./set_env.sh
python3 test/run_all_tests.py
"""
            
        test_runner.write_text(test_content)
        if not self.is_windows:
            os.chmod(test_runner, 0o755)
            
        print("✅ Test runner created")
        return True
        
    def finalize_setup(self):
        """Finalize the setup process"""
        print("[8/8] Finalizing setup...")
        
        print("✅ CanRun setup completed successfully!")
        print()
        print("Next steps:")
        print("1. Run tests: python test/run_all_tests.py")
        print("2. Try sample queries: python test/sample_queries.py")
        print("3. Use the plugin: python plugin/g-assist-plugin-canrun.py --help")
        print()
        
        if self.is_windows:
            print("Windows users:")
            print("- Run 'set_env.bat' to set environment variables")
            print("- Use 'run_canrun.bat' to launch CanRun")
            print("- Use 'run_tests.bat' to run tests")
        else:
            print("Linux users:")
            print("- Run 'source set_env.sh' to set environment variables")
            print("- Use './run_canrun.sh' to launch CanRun")
            print("- Use './run_tests.sh' to run tests")
            
        return True
        
    def run_setup(self):
        """Run the complete setup process"""
        self.print_header()
        
        steps = [
            self.check_python_version,
            self.check_dependencies,
            self.create_directories,
            self.setup_configuration,
            self.setup_environment,
            self.validate_installation,
            self.create_shortcuts,
            self.finalize_setup
        ]
        
        for step in steps:
            if not step():
                print("❌ Setup failed. Please check the errors above.")
                return False
                
        return True

def main():
    """Main setup function"""
    setup = CanRunSetup()
    success = setup.run_setup()
    
    if success:
        print("\n🎉 CanRun setup completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ CanRun setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()