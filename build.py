#!/usr/bin/env python3
"""
CanRun Build Script
Automated build, package, and distribution script for CanRun
"""

import os
import sys
import subprocess
import shutil
import json
import zipfile
import platform
from pathlib import Path
from datetime import datetime

class CanRunBuilder:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.version = "1.0.0"
        self.build_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.is_windows = platform.system() == "Windows"
        
    def print_header(self):
        """Print build header"""
        print("=" * 60)
        print("CanRun Universal Game Compatibility Checker")
        print("Build and Package Script")
        print("=" * 60)
        print(f"Version: {self.version}")
        print(f"Build Date: {self.build_timestamp}")
        print(f"Platform: {platform.system()}")
        print()
        
    def clean_build_dirs(self):
        """Clean previous build directories"""
        print("[1/10] Cleaning build directories...")
        
        for dir_path in [self.build_dir, self.dist_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"✅ Cleaned {dir_path.name}")
            
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created {dir_path.name}")
            
        return True
        
    def validate_source_code(self):
        """Validate source code before building"""
        print("[2/10] Validating source code...")
        
        # Check if all required files exist
        required_files = [
            "src/hardware_detector.py",
            "src/requirements_fetcher.py",
            "src/compatibility_analyzer.py",
            "src/performance_predictor.py",
            "src/canrun_engine.py",
            "plugin/g-assist-plugin-canrun.py",
            "plugin/manifest.json",
            "data/game_requirements.json",
            "data/gpu_hierarchy.json"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
                
        if missing_files:
            print(f"❌ Missing required files: {', '.join(missing_files)}")
            return False
            
        print("✅ All required files present")
        
        # Try to import modules to check for syntax errors
        try:
            sys.path.insert(0, str(self.project_root))
            
            from src.hardware_detector import HardwareDetector
            from src.requirements_fetcher import RequirementsFetcher
            from src.compatibility_analyzer import CompatibilityAnalyzer
            from src.performance_predictor import PerformancePredictor
            from src.canrun_engine import CanRunEngine
            
            print("✅ All modules imported successfully")
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return False
            
        return True
        
    def build_core_package(self):
        """Build the core CanRun package"""
        print("[3/10] Building core package...")
        
        # Create core package directory
        core_package = self.build_dir / "canrun-core"
        core_package.mkdir(exist_ok=True)
        
        # Copy source files
        src_dest = core_package / "src"
        shutil.copytree(self.project_root / "src", src_dest)
        print("✅ Copied source files")
        
        # Copy data files
        data_dest = core_package / "data"
        shutil.copytree(self.project_root / "data", data_dest)
        print("✅ Copied data files")
        
        # Copy configuration files
        config_dest = core_package / "config"
        config_dest.mkdir(exist_ok=True)
        
        # Create default config
        default_config = {
            "version": self.version,
            "build_date": self.build_timestamp,
            "debug": False,
            "log_level": "info"
        }
        
        with open(config_dest / "default.json", "w") as f:
            json.dump(default_config, f, indent=2)
        print("✅ Created configuration files")
        
        # Create __init__.py files
        (core_package / "__init__.py").write_text("")
        (src_dest / "__init__.py").write_text("")
        print("✅ Created package initialization files")
        
        return True
        
    def build_plugin_package(self):
        """Build the G-Assist plugin package"""
        print("[4/10] Building plugin package...")
        
        # Create plugin package directory
        plugin_package = self.build_dir / "canrun-g-assist-plugin"
        plugin_package.mkdir(exist_ok=True)
        
        # Copy plugin files
        plugin_files = [
            "plugin/g-assist-plugin-canrun.py",
            "plugin/manifest.json",
            "plugin/requirements.txt",
            "plugin/setup.bat",
            "plugin/test.bat",
            "plugin/build.bat"
        ]
        
        for file_path in plugin_files:
            src_file = self.project_root / file_path
            if src_file.exists():
                shutil.copy2(src_file, plugin_package)
                print(f"✅ Copied {file_path}")
        
        # Copy core package as dependency
        core_dest = plugin_package / "canrun-core"
        shutil.copytree(self.build_dir / "canrun-core", core_dest)
        print("✅ Copied core package to plugin")
        
        # Create plugin README
        plugin_readme = plugin_package / "README.md"
        plugin_readme_content = f"""# CanRun G-Assist Plugin v{self.version}

## Installation

1. Run `setup.bat` to install dependencies
2. Run `test.bat` to verify functionality
3. Copy this folder to your G-Assist plugins directory
4. Restart G-Assist to load the plugin

## Usage

The plugin provides the following functions:
- `detect_hardware` - Detect system hardware
- `check_compatibility` - Check game compatibility
- `predict_performance` - Predict game performance
- `get_optimization_suggestions` - Get optimization suggestions

## Requirements

- Python 3.8+
- NVIDIA RTX/GTX GPU (for G-Assist compatibility)
- Windows 10/11

## Build Information

- Version: {self.version}
- Build Date: {self.build_timestamp}
- Platform: {platform.system()}
"""
        plugin_readme.write_text(plugin_readme_content)
        print("✅ Created plugin README")
        
        return True
        
    def build_standalone_package(self):
        """Build standalone application package"""
        print("[5/10] Building standalone package...")
        
        # Create standalone package directory
        standalone_package = self.build_dir / "canrun-standalone"
        standalone_package.mkdir(exist_ok=True)
        
        # Copy core package
        core_dest = standalone_package / "canrun"
        shutil.copytree(self.build_dir / "canrun-core", core_dest)
        print("✅ Copied core package")
        
        # Create standalone launcher
        launcher_content = f"""#!/usr/bin/env python3
'''
CanRun Standalone Application
Version: {self.version}
Build Date: {self.build_timestamp}
'''

import sys
import os
from pathlib import Path

# Add canrun to path
sys.path.insert(0, str(Path(__file__).parent / "canrun"))

from src.canrun_engine import CanRunEngine
import asyncio
import json

async def main():
    print("CanRun Universal Game Compatibility Checker v{self.version}")
    print("=" * 60)
    
    engine = CanRunEngine()
    await engine.initialize()
    
    if len(sys.argv) < 2:
        print("Usage: python canrun_app.py <game_name> [settings] [resolution]")
        print("Example: python canrun_app.py 'Cyberpunk 2077' High 1080p")
        return
    
    game = sys.argv[1]
    settings = sys.argv[2] if len(sys.argv) > 2 else "High"
    resolution = sys.argv[3] if len(sys.argv) > 3 else "1080p"
    
    print(f"Analyzing: {{game}} ({{settings}} @ {{resolution}})")
    print("-" * 60)
    
    result = await engine.analyze_game_compatibility(game, settings, resolution)
    
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("Analysis failed - game not found or system error")

if __name__ == "__main__":
    asyncio.run(main())
"""
        
        launcher_file = standalone_package / "canrun_app.py"
        launcher_file.write_text(launcher_content)
        print("✅ Created standalone launcher")
        
        # Copy requirements
        shutil.copy2(self.project_root / "plugin" / "requirements.txt", standalone_package)
        print("✅ Copied requirements")
        
        return True
        
    def build_test_package(self):
        """Build test package"""
        print("[6/10] Building test package...")
        
        # Create test package directory
        test_package = self.build_dir / "canrun-tests"
        test_package.mkdir(exist_ok=True)
        
        # Copy test files
        test_dest = test_package / "test"
        shutil.copytree(self.project_root / "test", test_dest)
        print("✅ Copied test files")
        
        # Copy core package for testing
        core_dest = test_package / "canrun"
        shutil.copytree(self.build_dir / "canrun-core", core_dest)
        print("✅ Copied core package for testing")
        
        # Create test runner
        test_runner_content = f"""#!/usr/bin/env python3
'''
CanRun Test Runner
Version: {self.version}
Build Date: {self.build_timestamp}
'''

import sys
import os
from pathlib import Path

# Add canrun to path
sys.path.insert(0, str(Path(__file__).parent / "canrun"))

# Run tests
os.chdir(Path(__file__).parent)
sys.path.insert(0, ".")

from test.run_all_tests import main

if __name__ == "__main__":
    main()
"""
        
        test_runner_file = test_package / "run_tests.py"
        test_runner_file.write_text(test_runner_content)
        print("✅ Created test runner")
        
        return True
        
    def build_documentation(self):
        """Build documentation package"""
        print("[7/10] Building documentation...")
        
        # Create docs directory
        docs_package = self.build_dir / "canrun-docs"
        docs_package.mkdir(exist_ok=True)
        
        # Create comprehensive documentation
        docs_content = f"""# CanRun Universal Game Compatibility Checker

Version: {self.version}  
Build Date: {self.build_timestamp}

## Overview

CanRun is a universal game compatibility checker designed specifically for NVIDIA RTX systems with G-Assist integration. It analyzes your system hardware and predicts game performance with high accuracy.

## Features

- **Hardware Detection**: Comprehensive system analysis including CPU, GPU, RAM, and storage
- **Game Compatibility**: Checks if games will run on your system
- **Performance Prediction**: Predicts FPS and performance levels
- **DLSS Support**: Calculates DLSS performance improvements
- **RTX Analysis**: Evaluates RTX feature compatibility and performance impact
- **Optimization Suggestions**: Provides actionable recommendations

## System Requirements

- Windows 10/11 (64-bit)
- NVIDIA RTX/GTX GPU (for G-Assist compatibility)
- Python 3.8 or higher
- 4GB RAM minimum, 8GB recommended
- Internet connection for game data updates

## Installation

### G-Assist Plugin
1. Extract the plugin package
2. Run `setup.bat` to install dependencies
3. Copy the plugin folder to your G-Assist plugins directory
4. Restart G-Assist

### Standalone Application
1. Extract the standalone package
2. Install requirements: `pip install -r requirements.txt`
3. Run: `python canrun_app.py "Game Name" High 1080p`

## Usage

### Command Line
```bash
python canrun_app.py "Cyberpunk 2077" High 1080p
python canrun_app.py "Hogwarts Legacy" Ultra 1440p
```

### G-Assist Integration
- "Can I run Cyberpunk 2077?"
- "What's my performance for Hogwarts Legacy?"
- "Optimize settings for Red Dead Redemption 2"

## API Reference

### Hardware Detection
```python
from canrun.src.hardware_detector import HardwareDetector

detector = HardwareDetector()
hardware = await detector.detect_system_hardware()
```

### Compatibility Analysis
```python
from canrun.src.compatibility_analyzer import CompatibilityAnalyzer

analyzer = CompatibilityAnalyzer()
result = await analyzer.analyze_compatibility(hardware, requirements)
```

### Performance Prediction
```python
from canrun.src.performance_predictor import PerformancePredictor

predictor = PerformancePredictor()
performance = await predictor.predict_performance(hardware, requirements, "1080p", "High")
```

## Configuration

Configuration files are located in the `config` directory:
- `default.json` - Default settings
- `canrun.conf` - User configuration (created during setup)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Hardware Detection Fails**: Run as administrator on Windows
3. **Game Not Found**: Check spelling or update game database
4. **Performance Issues**: Reduce log level in configuration

### Debug Mode
Enable debug mode in configuration:
```json
{{
  "debug": true,
  "log_level": "debug"
}}
```

## Support

For issues and support:
1. Check the troubleshooting section
2. Run tests: `python run_tests.py`
3. Review logs in the `logs` directory

## Technical Details

### Architecture
- **Hardware Detection**: Multi-method detection with fallbacks
- **Requirements Fetching**: Steam API integration with local cache
- **Compatibility Analysis**: Heuristic-based scoring system
- **Performance Prediction**: GPU tier-based calculations with DLSS/RTX support

### Data Sources
- Steam Store API
- Local game requirements database
- GPU performance hierarchy
- Community-contributed data

## License

This software is provided as-is for educational and personal use.

## Version History

### v{self.version} ({self.build_timestamp})
- Initial release
- Core functionality implementation
- G-Assist plugin integration
- Comprehensive hardware detection
- Performance prediction with DLSS/RTX support
"""
        
        docs_file = docs_package / "README.md"
        docs_file.write_text(docs_content)
        print("✅ Created comprehensive documentation")
        
        return True
        
    def create_distribution_packages(self):
        """Create distribution packages"""
        print("[8/10] Creating distribution packages...")
        
        packages = [
            ("canrun-core", "Core Library"),
            ("canrun-g-assist-plugin", "G-Assist Plugin"),
            ("canrun-standalone", "Standalone Application"),
            ("canrun-tests", "Test Suite"),
            ("canrun-docs", "Documentation")
        ]
        
        for package_name, description in packages:
            package_path = self.build_dir / package_name
            if package_path.exists():
                zip_name = f"{package_name}-v{self.version}.zip"
                zip_path = self.dist_dir / zip_name
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(package_path):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(package_path)
                            zipf.write(file_path, arcname)
                
                print(f"✅ Created {zip_name} ({description})")
        
        return True
        
    def create_full_distribution(self):
        """Create complete distribution package"""
        print("[9/10] Creating full distribution...")
        
        # Create complete package
        full_package = self.build_dir / "canrun-complete"
        full_package.mkdir(exist_ok=True)
        
        # Copy all components
        components = [
            ("canrun-core", "canrun"),
            ("canrun-g-assist-plugin", "plugin"),
            ("canrun-tests", "tests"),
            ("canrun-docs", "docs")
        ]
        
        for src_name, dest_name in components:
            src_path = self.build_dir / src_name
            dest_path = full_package / dest_name
            
            if src_path.exists():
                shutil.copytree(src_path, dest_path)
                print(f"✅ Added {dest_name} to complete package")
        
        # Create master setup script
        setup_script = full_package / "setup.py"
        shutil.copy2(self.project_root / "setup.py", setup_script)
        print("✅ Added setup script")
        
        # Create complete package zip
        zip_name = f"canrun-complete-v{self.version}.zip"
        zip_path = self.dist_dir / zip_name
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(full_package):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(full_package)
                    zipf.write(file_path, arcname)
        
        print(f"✅ Created {zip_name} (Complete Distribution)")
        
        return True
        
    def generate_build_report(self):
        """Generate build report"""
        print("[10/10] Generating build report...")
        
        # Calculate package sizes
        package_info = []
        for package_file in self.dist_dir.glob("*.zip"):
            size_mb = package_file.stat().st_size / (1024 * 1024)
            package_info.append({
                "name": package_file.name,
                "size_mb": round(size_mb, 2)
            })
        
        # Create build report
        report = {
            "build_info": {
                "version": self.version,
                "build_date": self.build_timestamp,
                "platform": platform.system(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            },
            "packages": package_info,
            "total_packages": len(package_info),
            "build_status": "SUCCESS"
        }
        
        # Save report
        report_file = self.dist_dir / "build_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print("✅ Build report generated")
        
        # Print summary
        print("\n" + "=" * 60)
        print("BUILD SUMMARY")
        print("=" * 60)
        print(f"Version: {self.version}")
        print(f"Build Date: {self.build_timestamp}")
        print(f"Packages Created: {len(package_info)}")
        print("\nPackage Details:")
        for pkg in package_info:
            print(f"  {pkg['name']}: {pkg['size_mb']} MB")
        
        return True
        
    def build_all(self):
        """Run the complete build process"""
        self.print_header()
        
        build_steps = [
            self.clean_build_dirs,
            self.validate_source_code,
            self.build_core_package,
            self.build_plugin_package,
            self.build_standalone_package,
            self.build_test_package,
            self.build_documentation,
            self.create_distribution_packages,
            self.create_full_distribution,
            self.generate_build_report
        ]
        
        for step in build_steps:
            if not step():
                print("❌ Build failed. Please check the errors above.")
                return False
        
        print("\n🎉 Build completed successfully!")
        print(f"Distribution packages available in: {self.dist_dir}")
        
        return True

def main():
    """Main build function"""
    builder = CanRunBuilder()
    success = builder.build_all()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()