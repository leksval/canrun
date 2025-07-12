<table> <tr> <td width="110" valign="middle"> <img width="100" height="100" alt="canrun_logo" src="https://github.com/user-attachments/assets/239082bd-d5ca-427b-b235-5326299f3104" /> </td> <td valign="middle"> <h1 style="display:inline-block; vertical-align:middle; margin:0; padding:0;">  CanRun - Universal Game Compatibility Checker </h1> </td> </tr> </table>
  
  [![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/canrun/canrun)
  [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
  [![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
  [![G-Assist](https://img.shields.io/badge/G--Assist-Compatible-brightgreen.svg)](https://www.nvidia.com/en-us/geforce/technologies/g-assist/)
</div>

## 🚀 Overview

**CanRun** is a lightweight G-Assist plugin that instantly tells you if your PC can run any game. No more forum hunting or guesswork—just ask "Can I run Cyberpunk?" and get an immediate, accurate answer with FPS predictions and optimization tips.

### ✨ Key Features

- **🎯 Instant Compatibility Check**: Voice command → immediate answer
- **📊 FPS Prediction**: Heuristic-based performance estimates across quality settings
- **🔍 Multi-Source Data**: Aggregates Steam, community benchmarks, and local data
- **📚 Library Scanner**: Batch-check your entire game collection
- **💡 Smart Recommendations**: Optimal settings, DLSS tips, upgrade suggestions
- **🏃 Zero Setup**: Drop-in plugin, works immediately

## 🏁 Quick Start (For Judges)

**1-Minute Setup:**

```bash
# 1. Clone and enter plugin directory
git clone https://github.com/yourusername/canrun
cd canrun

# 2. Install dependencies
python setup.py

# 3. Test the plugin
python plugin/test.bat
```

**Test Commands:**

- "Hey CanRun, can I run Baldur's Gate 3?"
- "What FPS will I get in Starfield on High?"
- "Scan my Steam library"
- "What should I upgrade for Cyberpunk?"

## 📁 Project Structure

```
canrun/
├── plugin/                      # Main G-Assist Plugin (PRIMARY SUBMISSION)
│   ├── g-assist-plugin-canrun.py # Core plugin entry point
│   ├── manifest.json           # G-Assist function definitions
│   ├── requirements.txt        # Python dependencies
│   ├── setup.bat              # Windows setup script
│   └── test.bat               # Plugin test script
│
├── src/                        # Shared core modules
│   ├── canrun_engine.py       # Main compatibility engine
│   ├── hardware_detector.py   # System spec detection
│   ├── requirements_fetcher.py # Game requirements APIs
│   ├── compatibility_analyzer.py # Analysis logic
│   └── performance_predictor.py # Heuristic-based FPS prediction
│
├── data/                       # Static data files
│   ├── game_requirements.json  # Cached game requirements
│   └── gpu_hierarchy.json     # GPU performance tiers
│
├── test/                       # Test scripts
│   ├── test_hardware_detection.py
│   ├── test_compatibility_analysis.py
│   └── run_all_tests.py
│
├── LICENSE                     # Apache 2.0 license
├── README.md                  # This file
├── setup.py                   # Automated setup script
└── build.py                   # Build automation
```

## 🔧 Technical Implementation

### Core Components

**1. Hardware Detection**
```python
# Automatic system spec detection
- GPU: NVIDIA ML, GPUtil, nvidia-ml-py
- CPU: cpuinfo, psutil
- RAM: psutil
- Storage: psutil
```

**2. Requirements Aggregation**
```python
# Multi-source data fetching
- Steam Store API (official requirements)
- Local cache (offline support)
- Curated database (12+ popular games)
```

**3. Compatibility Analysis**
```python
# Smart analysis engine
- Minimum/Recommended comparison
- Bottleneck identification
- Performance tier calculation
- Feature support (DLSS, RTX, etc.)
```

**4. Performance Prediction**
```python
# Heuristic-based FPS estimation
- Hardware similarity matching
- GPU tier-based calculations
- Resolution/setting scaling
- DLSS performance boost calculation
```

### G-Assist Integration

**Communication Flow:**
```
User Voice/Text → G-Assist SLM → Intent Recognition → CanRun Plugin
     ↓                                                       ↓
  Response ← Markdown Formatting ← Analysis ← Hardware Check
```

## 📚 API Reference

### Plugin Functions

The CanRun plugin provides four main functions accessible through G-Assist:

#### 1. `detect_hardware()`

Detects and returns comprehensive system hardware information.

**Parameters:** None

**Returns:**
```json
{
  "cpu": {
    "name": "Intel Core i7-9700K",
    "cores": 8,
    "threads": 8,
    "base_clock": 3.6,
    "boost_clock": 4.9,
    "generation": "9th Gen",
    "score": 750
  },
  "gpu": {
    "name": "NVIDIA RTX 3080",
    "memory": 10,
    "tier": "High",
    "features": ["DLSS 2", "RTX"],
    "score": 720
  },
  "memory": {
    "total": 16,
    "available": 12,
    "speed": 3200,
    "type": "DDR4"
  },
  "storage": {
    "total": 1000,
    "available": 500,
    "type": "NVMe SSD",
    "speed": 3500
  }
}
```

**G-Assist Usage:**
- "What are my system specs?"
- "Show me my hardware configuration"
- "What's my GPU and CPU?"

#### 2. `check_compatibility(game_name)`

Checks if a specific game is compatible with the current system.

**Parameters:**
- `game_name` (string, required): Name of the game to check

**Returns:**
```json
{
  "compatibility": {
    "compatibility_level": "High",
    "overall_score": 85,
    "bottlenecks": ["Storage"],
    "component_analysis": {
      "cpu": {
        "status": "Excellent",
        "score": 95,
        "meets_minimum": true,
        "meets_recommended": true
      },
      "gpu": {
        "status": "Good",
        "score": 80,
        "meets_minimum": true,
        "meets_recommended": true
      },
      "memory": {
        "status": "Excellent",
        "score": 90,
        "meets_minimum": true,
        "meets_recommended": true
      }
    }
  },
  "requirements": {
    "minimum": {
      "cpu": "Intel Core i5-8400",
      "gpu": "NVIDIA GTX 1060",
      "memory": 12,
      "storage": 70
    },
    "recommended": {
      "cpu": "Intel Core i7-9700K",
      "gpu": "NVIDIA RTX 2070",
      "memory": 16,
      "storage": 70
    }
  }
}
```

**G-Assist Usage:**
- "Can I run Cyberpunk 2077?"
- "Is my system compatible with Hogwarts Legacy?"
- "Check if I can play Elden Ring"

#### 3. `predict_performance(game_name, settings, resolution)`

Predicts game performance for specific graphics settings and resolution.

**Parameters:**
- `game_name` (string, required): Name of the game
- `settings` (string, optional): Graphics settings ("Low", "Medium", "High", "Ultra") - defaults to "High"
- `resolution` (string, optional): Display resolution ("1080p", "1440p", "4K") - defaults to "1080p"

**Returns:**
```json
{
  "performance": {
    "fps": 75,
    "performance_level": "High",
    "stability": "Stable",
    "dlss_boost": 1.4,
    "rtx_features": ["Ray Tracing", "DLSS 2"]
  },
  "settings_analysis": {
    "current_settings": "High",
    "current_resolution": "1080p",
    "recommended_settings": "High",
    "recommended_resolution": "1080p"
  },
  "optimization_notes": [
    "DLSS Quality mode recommended for better performance",
    "RTX features fully supported"
  ]
}
```

**G-Assist Usage:**
- "What FPS can I get in Cyberpunk 2077 on High settings?"
- "How will Hogwarts Legacy perform at 1440p?"
- "Can I run Elden Ring at 4K Ultra?"

#### 4. `get_optimization_suggestions(game_name)`

Provides optimization recommendations for running a specific game.

**Parameters:**
- `game_name` (string, required): Name of the game

**Returns:**
```json
{
  "optimization_suggestions": [
    {
      "type": "Graphics Settings",
      "description": "Enable DLSS Quality mode for better performance",
      "expected_improvement": "20-30% FPS boost",
      "priority": "High"
    },
    {
      "type": "System Settings",
      "description": "Close unnecessary background applications",
      "expected_improvement": "5-10% FPS boost",
      "priority": "Medium"
    },
    {
      "type": "Hardware",
      "description": "Consider upgrading to NVMe SSD for faster loading",
      "expected_improvement": "Reduced loading times",
      "priority": "Low"
    }
  ],
  "current_bottlenecks": ["Storage"],
  "dlss_available": true,
  "rtx_available": true
}
```

**G-Assist Usage:**
- "How can I optimize Cyberpunk 2077?"
- "What settings should I use for best performance in Hogwarts Legacy?"
- "Give me optimization tips for Elden Ring"

### Command Line Interface

The plugin can also be used directly via command line for testing:

```bash
# Detect hardware
python plugin/g-assist-plugin-canrun.py --function detect_hardware

# Check compatibility
python plugin/g-assist-plugin-canrun.py --function check_compatibility --game "Cyberpunk 2077"

# Predict performance
python plugin/g-assist-plugin-canrun.py --function predict_performance --game "Hogwarts Legacy" --settings "Ultra" --resolution "1440p"

# Get optimization suggestions
python plugin/g-assist-plugin-canrun.py --function get_optimization_suggestions --game "Elden Ring"
```

### Natural Language Examples

Users can interact with CanRun through G-Assist using natural language:

**Hardware Detection:**
- "What's my GPU?"
- "Show system specs"
- "What CPU do I have?"

**Compatibility Checking:**
- "Can I play [game name]?"
- "Is [game name] compatible?"
- "Will [game name] run on my system?"

**Performance Prediction:**
- "What FPS in [game name]?"
- "How will [game name] perform?"
- "Can I run [game name] at 4K?"

**Optimization:**
- "How to optimize [game name]?"
- "Best settings for [game name]?"
- "Improve performance in [game name]?"

### Error Handling

**Common Error Responses:**
```json
{
  "error": "Game not found",
  "message": "The specified game is not in our database",
  "available_games": ["Cyberpunk 2077", "Hogwarts Legacy", "..."]
}
```

```json
{
  "error": "Hardware detection failed",
  "message": "Could not detect system hardware",
  "fallback_detected": true
}
```

```json
{
  "error": "Invalid parameters",
  "message": "Settings must be one of: Low, Medium, High, Ultra",
  "provided": "Maximum"
}
```

## 📋 Requirements

### System Requirements
- Windows 10/11
- NVIDIA RTX/GTX GPU (for G-Assist)
- Python 3.8+ (for development)
- 4GB RAM minimum, 8GB recommended

### Python Dependencies
```txt
# Core
psutil>=5.9.0          # System hardware detection
GPUtil>=1.4.0          # GPU detection and monitoring
nvidia-ml-py>=12.535.108  # NVIDIA GPU management library
py-cpuinfo>=9.0.0      # Detailed CPU information
requests>=2.31.0       # HTTP requests for Steam API
pynvml>=11.5.0         # Enhanced NVIDIA GPU monitoring
cpuinfo>=9.0.0         # Additional CPU architecture detection
```

## 🚀 Advanced Usage

### Building from Source
```bash
# Development mode
python plugin/g-assist-plugin-canrun.py --help

# Production build
python build.py
```

### API Examples
```python
# Direct Python usage
from src.canrun_engine import CanRunEngine

engine = CanRunEngine()
await engine.initialize()
result = await engine.analyze_game_compatibility("Cyberpunk 2077", "High", "1080p")
print(f"Compatibility: {result['compatibility']['compatibility_level']}")
print(f"Expected FPS: {result['performance']['fps']}")
```

## 🎯 Why CanRun Wins

1. **Solves Universal Problem**: Every gamer asks "Can I run it?"
2. **Zero Friction**: No sign-ups, no ads, instant answers
3. **Technical Excellence**: Multi-source data, heuristic predictions
4. **Perfect G-Assist Fit**: Natural language, local processing
5. **Extensible**: Easy to add new games/sources

## 📝 Development Notes

### For Hackathon Judges
- Primary submission is `/plugin` folder - drop and run
- No Docker/complex setup required
- Source code in `/src` shows technical depth
- Comprehensive test suite in `/test`

### Supported Games
- Cyberpunk 2077
- Hogwarts Legacy
- Call of Duty: Modern Warfare II
- Red Dead Redemption 2
- Spider-Man Remastered
- Forza Horizon 5
- Starfield
- Diablo IV
- Resident Evil 4
- The Witcher 3
- Apex Legends
- Valorant

## 🧪 Testing

### Run All Tests
```bash
python test/run_all_tests.py
```

### Sample Queries
```bash
python test/sample_queries.py
```

## 🤝 Contributing

This project is open source! Contributions welcome:
- Add new game sources
- Improve performance predictions
- Enhance hardware detection
- Add international support

## 📄 License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) file.

```
Copyright 2025 CanRun Project

```

## 🙏 Acknowledgments

- NVIDIA G-Assist team for the platform
- Steam for official API access
- Hardware detection library contributors
- All beta testers

<div align="center">
  
**Built for NVIDIA G-Assist Plug-in Hackathon 2025**

*Because every gamer deserves to know - Can I Run It?*

#AIonRTXHackathon

</div>
