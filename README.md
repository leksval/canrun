<table> <tr> <td width="110" valign="middle"> <img width="100" height="100" alt="canrun_logo" src="https://github.com/user-attachments/assets/239082bd-d5ca-427b-b235-5326299f3104" /> </td> <td valign="middle"> <h1 style="display:inline-block; vertical-align:middle; margin:0; padding:0;">  CanRun - G-Assist Game Compatibility Checker </h1> </td> </tr> </table>
  
  [![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/canrun/canrun)
  [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
  [![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
  [![G-Assist](https://img.shields.io/badge/G--Assist-Compatible-brightgreen.svg)](https://www.nvidia.com/en-us/geforce/technologies/g-assist/)
</div>

## 🚀 Overview

**CanRun** is an RTX/GTX-exclusive G-Assist plugin that instantly tells you if your PC can run any game with an advanced **S-A-B-C-D-F tier system**. Powered by G-Assist's embedded 8B parameter Llama model running locally on RTX GPUs, it delivers intelligent, privacy-protected compatibility analysis with zero data leaving your system.

### ✨ Key Features

- **🎯 RTX/GTX Optimized**: Exclusively designed for RTX/GTX systems with G-Assist integration
- **🎮 CANRUN! Indicator**: Instant visual feedback when your system meets game requirements
- **⭐ S-A-B-C-D-F Tier System**: Advanced performance classification with weighted scoring (GPU 60%, CPU 25%, RAM 15%)
- **🧠 AI-Powered Analysis**: Leverages G-Assist's embedded 8B Llama model for intelligent insights
- **🔒 Privacy-by-Design**: All processing happens locally on your RTX GPU—no data leaves your system
- **🎯 Steam-First Data**: Prioritizes Steam API for most up-to-date game requirements
- **📊 Smart Performance Prediction**: Comprehensive hardware hierarchies with RTX 40/30 series, GTX series, AMD RX series support
- **💡 Intelligent Recommendations**: AI-generated optimization tips, DLSS strategies, upgrade suggestions
- **🏃 Zero Setup**: Drop-in plugin with automatic RTX/GTX validation

## 🏁 Quick Start (For Judges)

**1-Minute Setup:**

```bash
# 1. Clone and enter directory
git clone https://github.com/yourusername/canrun
cd canrun

# 2. Install dependencies with uv (recommended)
uv sync

# Alternative: Install with pip
pip install -r requirements.txt

# 3. Test the advanced performance assessment
uv run python plugin.py --function predict_advanced_performance --game "Call of Duty: Modern Warfare"
```

**Test Commands:**

- "Hey CanRun, can I run Baldur's Gate 3?"
- "What tier performance will I get in Starfield?"
- "Show my CANRUN status for Cyberpunk 2077"
- "What should I upgrade for better gaming performance?"

## 🆕 What's New in v2.0

### Advanced S-A-B-C-D-F Tier System
- **S-Tier (90-100)**: Exceptional - Ultra settings, 4K@60fps+
- **A-Tier (80-89)**: Excellent - High settings, 1440p@60fps
- **B-Tier (70-79)**: Good - High settings, 1080p@60fps
- **C-Tier (60-69)**: Adequate - Medium settings, 1080p@30fps
- **D-Tier (50-59)**: Minimum - Low settings, 720p@30fps
- **F-Tier (0-49)**: Below Minimum - Unable to run acceptably

### 🎮 CANRUN! Status Indicator
```
🎮 **CANRUN!** ✅  # Displayed for D-tier and above
❌ **Cannot Run**   # Displayed for F-tier systems
```

### Steam-First Architecture
- Steam API prioritized for most current game requirements
- Fallback to local cache for offline scenarios
- Real-time game specification updates

## 🧪 Running Tests

**Quick Test Commands:**

```bash
# Run all tests
uv run python -m pytest test/ -v

# Test advanced performance assessment
uv run python plugin.py --function predict_advanced_performance --game "Cyberpunk 2077"

# Run specific test suites
uv run python -m pytest test/test_llm_analysis.py -v          # LLM & G-Assist integration (23 tests)
uv run python -m pytest test/test_steam_api_integration.py -v  # Steam API integration (17 tests)

# Run individual test files
uv run python -m pytest test/test_compatibility_analysis.py -v
uv run python -m pytest test/test_performance_prediction.py -v
uv run python -m pytest test/test_hardware_detection.py -v
uv run python -m pytest test/test_privacy_aware_hardware.py -v
```

**Test Coverage:**
- ✅ **Advanced Performance Assessment**: S-A-B-C-D-F tier system with weighted scoring
- ✅ **LLM Analysis**: 20/20 tests passing - G-Assist integration, privacy protection
- ✅ **Steam API Integration**: 15/15 tests passing - Real-time requirements fetching
- ✅ **Hardware Detection**: Fixed Windows 11, display resolution, NVIDIA driver detection
- ✅ **CANRUN! Indicator**: Visual feedback system for compatibility status

## 🏗️ Building Windows Executable

**For G-Assist Plugin Deployment:**

The plugin includes a Windows executable build process for NVIDIA G-Assist plugin marketplace submission:

```bash
# Build Windows executable with PyInstaller (optimized)
uv run pyinstaller plugin.py --name g-assist-plugin-python --onefile --paths=src --paths=. --hidden-import=canrun_engine --hidden-import=privacy_aware_hardware_detector --hidden-import=service_container --exclude-module=torchvision --exclude-module=torchaudio --exclude-module=torch.utils.tensorboard --exclude-module=torch.testing --exclude-module=torch.hub --exclude-module=pytest --exclude-module=sphinx --exclude-module=pygments --exclude-module=sympy --strip

# Test the built executable
.\dist\g-assist-plugin-python.exe --function detect_hardware
```

**Build Output:**
- Creates `g-assist-plugin-python.exe` in the root directory
- Single-file executable with all dependencies bundled
- Compatible with G-Assist plugin manifest requirements
- Includes all src modules and data files

**What the executable includes:**
- Complete CanRun functionality with S-A-B-C-D-F tier system
- Hardware detection
- Steam API integration with fallback to local cache
- Privacy-aware analysis with local processing
- G-Assist LLM integration support

**Testing Commands:**
```bash
# Hardware detection
.\dist\g-assist-plugin-python.exe --function detect_hardware

# Performance analysis
.\dist\g-assist-plugin-python.exe --function predict_advanced_performance --game "Cyberpunk 2077"

# Compatibility check
.\dist\g-assist-plugin-python.exe --function check_compatibility --game "Baldur's Gate 3"
```


## 📁 Project Structure

```
canrun/
├── plugin.py                   # Main G-Assist Plugin (PRIMARY SUBMISSION)
├── manifest.json              # G-Assist function definitions with LLM integration
├── pyproject.toml             # Modern uv package manager configuration
├── requirements.txt           # Python dependencies
│
├── src/                        # Core modules with advanced tier system
│   ├── canrun_engine.py       # Main compatibility engine with S-A-B-C-D-F integration
│   ├── privacy_aware_hardware_detector.py # Enhanced hardware detection (Windows 11, RTX 4090 support)
│   ├── game_requirements_fetcher.py # Steam-first game requirements with fallbacks
│   ├── compatibility_analyzer.py # Analysis logic with tier classification
│   ├── performance_predictor.py # Advanced S-A-B-C-D-F tier system with weighted algorithms
│   └── rtx_llm_analyzer.py    # G-Assist LLM integration module
│
├── data/                       # Static data files
│   ├── game_requirements.json  # Cached game requirements
│   └── gpu_hierarchy.json     # Comprehensive GPU/CPU performance hierarchies
│
├── test/                       # Comprehensive test suite
│   ├── test_hardware_detection.py
│   ├── test_compatibility_analysis.py
│   ├── test_performance_prediction.py
│   ├── test_llm_analysis.py   # LLM integration tests
│   ├── test_privacy_aware_hardware.py
│   └── test_steam_api_integration.py
│
├── LICENSE                     # Apache 2.0 license
├── README.md                  # This file
└── CHANGELOG.md              # Version history and updates
```

## 🔧 Technical Implementation

### Core Components

**1. Advanced Performance Assessment**
```python
# S-A-B-C-D-F tier system with weighted scoring
- GPU Performance: 60% weight (RTX 3080, GTX 1070, RTX 4060, etc.)
- CPU Performance: 25% weight (Ryzen 7800X3D, i9-13900K, etc.)
- RAM Performance: 15% weight (32GB DDR5, 16GB DDR4, etc.)
- Comprehensive hardware hierarchies with 50+ GPU/CPU models
- Smart system-reserved memory handling (31GB = 32GB installed)
```

**2. Steam-First Requirements Fetching**
```python
# Prioritized data source architecture
- Primary: Steam Store API (real-time, most current)
- Fallback: Local cache (offline support, curated database)
- Privacy-protected data sanitization throughout
- Automatic game ID resolution and requirement parsing
```

**3. CANRUN! Status System**
```python
# Instant compatibility feedback
- Visual indicators: 🎮 CANRUN! ✅ / ❌ Cannot Run
- Tier-based classification (D+ = Can Run, F = Cannot Run)
- Performance expectations with FPS predictions
- Settings recommendations (Ultra/4K, High/1440p, etc.)
```

**4. G-Assist LLM Integration**
```python
# Local AI processing with enhanced context
- G-Assist embedded 8B Llama model
- S-A-B-C-D-F tier-aware analysis
- Hardware-specific optimization recommendations
- Privacy-by-design architecture - all processing local
```

### Advanced Performance Assessment API

#### `predict_advanced_performance(game_name)`

**New Enhanced Performance Function with S-A-B-C-D-F Tier System**

**Returns:**
```json
{
  "tier": "S",
  "tier_description": "Exceptional - Ultra settings, 4K@60fps+",
  "score": 95,
  "expected_fps": 110,
  "recommended_settings": "Ultra/Maximum",
  "recommended_resolution": "4K (3840x2160)",
  "bottlenecks": [],
  "upgrade_suggestions": [],
  "canrun_status": "🎮 **CANRUN!** ✅"
}
```

**G-Assist Usage:**
- "What tier performance can I expect in Cyberpunk 2077?"
- "Show my CANRUN status for Elden Ring"
- "What's my performance tier for Baldur's Gate 3?"

## 🎮 Sample Interactions with New Features

### Advanced Tier System in Action

**Scenario 1: High-End System Assessment**
```
User: "Hey G-Assist, can I run Call of Duty: Modern Warfare?"

CanRun: "🎮 **CANRUN!** ✅

## Performance Tier: S
**Score**: 95/100
**Description**: Exceptional - Ultra settings, 4K@60fps+

## Performance Metrics
- **Expected FPS**: 110
- **Recommended Settings**: Ultra/Maximum
- **Recommended Resolution**: 4K (3840x2160)

## System Analysis
**Bottlenecks**: None detected

🧠 G-Assist AI Analysis:
Your high-end RTX system combination achieves S-tier performance. 
This is flagship-level hardware capable of maximum settings at 4K resolution."
```

**Scenario 2: Mid-Range System Assessment**
```
User: "What tier performance will I get in Cyberpunk 2077?"

CanRun: "🎮 **CANRUN!** ✅

## Performance Tier: B
**Score**: 75/100
**Description**: Good - High settings, 1080p@60fps

## Performance Metrics
- **Expected FPS**: 68
- **Recommended Settings**: High
- **Recommended Resolution**: 1080p (1920x1080)

## System Analysis
**Bottlenecks**: GPU (Minor)

## Upgrade Recommendations
- Consider RTX 3070 or better for A-tier performance
- Enable DLSS Quality mode for 20-30% FPS boost

🧠 AI Analysis: Your system provides solid B-tier gaming experience with room for optimization."
```

### Steam-First Data Integration

**Real-Time Requirements:**
- Fetches latest game requirements directly from Steam Store API
- Automatically updates minimum and recommended specifications
- Parses complex requirement formats with AI assistance
- Provides source attribution (Steam API vs Local Cache)

### Hardware Detection Improvements

**Enhanced System Recognition:**
```
✅ Windows 11 Pro (previously misdetected as Windows 10)
✅ RTX 3080 10GB VRAM (full RTX 30-series support)
✅ Intel Core i7-12700K (latest CPU generation support)
✅ 16GB DDR4-3200 (smart memory calculation)
✅ 2560x1440@165Hz (high-resolution display support)
✅ NVMe SSD Detection (storage type classification)
```

## 🔄 Migration from v1.0

**Automatic Compatibility:**
- All existing G-Assist commands continue to work
- Legacy performance predictions automatically use new tier system
- Enhanced accuracy with no breaking changes
- Improved privacy protection and data handling

**New Commands:**
```bash
# Test new advanced assessment
uv run python plugin.py --function predict_advanced_performance --game "Your Game"

# Check CANRUN status
uv run python plugin.py --function check_compatibility --game "Your Game"
```

## 📊 Performance Tier Benchmarks

### GPU Tier Classifications
- **S-Tier (95-100)**: RTX 4090, RTX 4080, RTX 3090 Ti
- **A-Tier (85-94)**: RTX 4070 Ti, RTX 3080, RX 7900 XTX
- **B-Tier (75-84)**: RTX 4070, RTX 3070, RX 7800 XT
- **C-Tier (65-74)**: RTX 3060 Ti, RTX 2070, RX 6700 XT
- **D-Tier (55-64)**: RTX 3060, GTX 1660 Ti, RX 6600
- **F-Tier (0-54)**: GTX 1050, older hardware

### CPU Tier Classifications
- **S-Tier (95-100)**: Ryzen 7800X3D, Ryzen 9 7950X, i9-13900K
- **A-Tier (85-94)**: Ryzen 7 7700X, i7-13700K, Ryzen 9 5900X
- **B-Tier (75-84)**: Ryzen 5 7600X, i5-13600K, Ryzen 7 5800X
- **C-Tier (65-74)**: Ryzen 5 5600X, i7-11700K, i5-12400
- **D-Tier (55-64)**: Ryzen 5 3600, i5-10600K, older quad-cores
- **F-Tier (0-54)**: Dual-core processors, very old hardware

## 🛠️ Development and Contributing

**Setting up Development Environment:**

```bash
# Clone repository
git clone https://github.com/yourusername/canrun
cd canrun

# Install development dependencies
uv sync --dev

# Run tests to verify setup
uv run python -m pytest test/ -v

# Test new performance assessment
uv run python plugin.py --function predict_advanced_performance --game "Test Game"
```

**Key Development Areas:**
- Advanced tier system enhancements
- Steam API integration improvements
- Hardware detection for new GPU/CPU releases
- G-Assist LLM integration optimization
- Privacy protection strengthening

## 📈 Version History

### v2.0.0 (Current)
- ✅ **S-A-B-C-D-F Tier System**: Complete performance classification overhaul
- ✅ **CANRUN! Indicator**: Visual compatibility status system
- ✅ **Steam-First Architecture**: Prioritized real-time game requirements support
- ✅ **Weighted Scoring Algorithm**: GPU 60%, CPU 25%, RAM 15%
- ✅ **Modern Package Management**: uv integration with pyproject.toml
- ✅ **Comprehensive Hardware Hierarchies**: 50+ GPU/CPU models

### v1.1.0 (Previous)
- Basic performance prediction
- G-Assist integration
- Privacy-aware hardware detection
- Steam API support

## 🎯 Future Roadmap

- **Enhanced Steam Integration**: Game library scanning and batch analysis
- **Expanded Hardware Support**: Intel Arc, future RTX/RX series
- **Advanced LLM Features**: Contextual gaming recommendations
- **Performance Database**: Community-driven benchmark data
- **Real-Time Optimization**: Dynamic settings adjustment

---

**Ready to see if your system can run any game? Try CanRun today!**

For technical support, feature requests, or contributions, visit [GitHub repository](https://github.com/leksval/canrun).
