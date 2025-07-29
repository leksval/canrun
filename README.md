---
title: CanRun – tells you if your PC can run any game with an advanced **S-A-B-C-D-F tier system
emoji: 🎮
colorFrom: red
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - mcp-server
  - hardvare-check
  - gradio
  - python
  - mathematics
  - llm-tools
---
<table> <tr> <td width="110" valign="middle"> <img width="100" height="100" alt="canrun_logo" src="https://github.com/user-attachments/assets/239082bd-d5ca-427b-b235-5326299f3104" /> </td> <td valign="middle"> <h1 style="display:inline-block; vertical-align:middle; margin:0; padding:0;">  CanRun - System Spec Game Compatibility Checker </h1> </td> </tr> </table>
  
  [![Version](https://img.shields.io/badge/version-8.0.0-blue.svg)](https://github.com/canrun/canrun)
  [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
  [![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
  [![G-Assist](https://img.shields.io/badge/G--Assist-Official%20Protocol%20Verified-brightgreen.svg)](https://www.nvidia.com/en-us/geforce/technologies/g-assist/)
  [![Steam API](https://img.shields.io/badge/Steam%20API-Integrated-blue.svg)](https://steamcommunity.com/dev)
  [![MCP Server](https://img.shields.io/badge/MCP%20Server-Enabled-brightgreen.svg)](https://developer.nvidia.com/mcp)

## 🚀 Overview

**CanRun** is an RTX/GTX-exclusive G-Assist plugin that instantly tells you if your PC can run any game with an advanced **S-A-B-C-D-F tier system** and enhanced Steam integration.

## ✨ Key Features

- **🎯 RTX/GTX Optimized**: Exclusively designed for RTX/GTX systems with G-Assist integration
- **🎮 CANRUN! Indicator**: Instant visual feedback when your system meets game requirements
- **⭐ S-A-B-C-D-F Tier System**: Advanced performance classification with weighted scoring (GPU 60%, CPU 25%, RAM 15%)
- **🧠 AI-Powered Analysis**: Leverages G-Assist's embedded 8B Llama model for intelligent insights
- **🔒 Privacy-by-Design**: All processing happens locally on your RTX GPU—no data leaves your system
- **🎯 Steam-First Data**: Prioritizes Steam API for most up-to-date game requirements
- **🎯 Intelligent Game Matching**: Advanced fuzzy matching handles game name variations
- **📊 Smart Performance Prediction**: Comprehensive hardware hierarchies
- **💡 Intelligent Recommendations**: AI-generated optimization tips, DLSS strategies, upgrade suggestions
- **🏃 Zero Setup**: Drop-in plugin with automatic RTX/GTX validation
- **🤖 MCP Server**: Official Model Context Protocol (MCP) server for G-Assist integration
- **📱 User-Friendly Display**: Optimized formatting with verdict at bottom for scrollable windows
- **🔄 JSON-Based Configuration**: All benchmark data externalized for easy maintenance
- **📊 Accurate Memory Reporting**: Shows available RAM/VRAM for precise calculations


![R3](https://github.com/user-attachments/assets/79a68ba5-a4d9-493c-bea5-d455edadf627)

---
## Live Demo, download and test locally for real hardvare evaluation as MCP server
[LINK](https://huggingface.co/spaces/grasant/canrun)
![HF Space Demo](image.png)

---

```bash
# Test the transformation - RTX 3080 + Intel i7-12700K system
uv run python plugin.py --function check_compatibility --game "Diablo 4"

# Result: CanRun Analysis: Diablo 4 - Tier A - EXCELLENT
# Score: 92/100 (Previously: 49/100)
# Steam API: ✅ Working (ID: 2344520)
# Performance Tier: A (Previously: F)
```



## 📦 G-Assist Plugin Installation

#### **Ready-to-Use Plugin**
The G-Assist plugin is now available with the correct structure:
```
└── canrun/                          # Plugin directory for G-Assist
    ├── g-assist-plugin-canrun.exe   # Main executable (FIXED)
    ├── manifest.json                # Plugin manifest (FIXED)
    └── plugin.py                    # Source code
```

### **🚀 INSTALLATION INSTRUCTIONS**

**Step 1: Copy Plugin to G-Assist Directory**
```bash
# Copy the entire canrun/canrun/ directory to G-Assist plugins folder:
# %USERPROFILE%\AppData\Local\NVIDIA Corporation\NVIDIA App\plugins\
# or
# %PROGRAMDATA%\NVIDIA Corporation\nvtopps\rise\plugins\
```

**Step 2: Restart G-Assist**
1. Close NVIDIA App / G-Assist completely
2. Restart NVIDIA App
3. Enable G-Assist if not already enabled

**Step 3: Test Plugin**
Try these commands:
- "canrun diablo"
- "can I run Cyberpunk 2077?"
- "/canrun elden ring"

  
## 🏁 Quick Start (cmd verion)

**1-Minute Setup & Verification:**

```bash
# 1. Clone and enter directory
git clone https://github.com/leksval/canrun
cd canrun

# 2. Install dependencies with uv (recommended)
uv sync

# 3. Test the official G-Assist protocol
python test/test_official_g_assist_protocol.py
# Expected: All tests PASSING with official protocol

# 4. Test enhanced G-Assist plugin
uv run python plugin.py --function check_compatibility --game "Diablo 4" --show-steam
# Expected: Enhanced G-Assist response with Steam Compare UI

# 5. Test natural language auto-detection
uv run python plugin.py --function auto_detect --input "Can I run Elden Ring?"
# Expected: Automatic routing to compatibility check
```


## 🤖 MCP Server Functionality (NEW!)

CanRun now includes a full-featured **Model Context Protocol (MCP) server** that allows G-Assist to directly integrate with the CanRun compatibility engine. This provides seamless AI-assisted game compatibility checking through the official NVIDIA MCP standard.

### MCP Tools and Capabilities

The MCP server exposes the following tools to G-Assist:

- **check_game_compatibility**: Analyze if a specific game can run on the current system
  - Input: Game name (e.g., "Diablo 4")
  - Output: Detailed compatibility analysis with performance tier

- **detect_hardware**: Provides comprehensive hardware detection for gaming systems
  - Output: Detailed hardware specifications focused on gaming performance

### Running the MCP Server

```bash
# Start the MCP server with auto port discovery
python app.py

# The server will be available at:
# http://localhost:xxxx (where xxxx is an available port)
```

### G-Assist MCP Integration

G-Assist can automatically discover and use the CanRun MCP server when both are running. This enables advanced conversational interactions like:

- "G-Assist, ask CanRun if I can play Starfield"
- "G-Assist, check if my system meets Diablo 4 requirements"
- "G-Assist, what's my gaming hardware like?"

## 🧪 Running Tests

**Primary Test Command (Recommended):**

```bash
# Run all tests with pytest
uv run python -m pytest test/ -v

# Test official G-Assist protocol specifically
python test/test_official_g_assist_protocol.py

# Test enhanced G-Assist communication
uv run python test/test_enhanced_g_assist_communication.py
```

**Test Coverage:**
- ✅ **Advanced Performance Assessment**: S-A-B-C-D-F tier system with weighted scoring
- ✅ **LLM Analysis**: 20/20 tests passing - G-Assist integration, privacy protection
- ✅ **Steam API Integration**: 15/15 tests passing - Real-time requirements fetching
- ✅ **Hardware Detection**: Fixed Windows 11, display resolution, NVIDIA driver detection
- ✅ **MCP Server**: Verified Model Context Protocol implementation

## 📁 Project Structure

```
canrun/
├── plugin.py                   # Main G-Assist Plugin (PRIMARY SUBMISSION)
├── app.py                      # Gradio UI and MCP Server implementation
├── manifest.json              # G-Assist function definitions with LLM integration
├── pyproject.toml             # Modern uv package manager configuration
├── requirements.txt           # Python dependencies
│
├── src/                        # Core modules with advanced tier system
│   ├── canrun_engine.py       # Main compatibility engine with S-A-B-C-D-F integration
│   ├── privacy_aware_hardware_detector.py # Enhanced hardware detection
│   ├── game_requirements_fetcher.py # Steam-first game requirements with fallbacks
│   ├── compatibility_analyzer.py # Analysis logic with tier classification
│   ├── dynamic_performance_predictor.py # Advanced S-A-B-C-D-F tier system
│   └── rtx_llm_analyzer.py    # G-Assist LLM integration module
│
├── data/                       # Static data files
│   ├── game_requirements.json  # Cached game requirements
│   └── gpu_hierarchy.json     # Comprehensive GPU/CPU performance hierarchies
│
├── test/                       # Comprehensive test suite
├── LICENSE                     # Apache 2.0 license
├── README.md                  # This file
└── CHANGELOG.md              # Version history and updates
```

## 🔧 Technical Implementation

### Core Components

**1. Official G-Assist Protocol**
```python
# Official NVIDIA G-Assist communication protocol
- Input: {"tool_calls": [{"func": "function_name", "params": {...}}]}
- Output: {"success": true, "message": "..."}<<END>>
- Communication: Standard stdin/stdout (verified working)
- Mode Detection: stdin.isatty() check for G-Assist environment
```

**2. Advanced Performance Assessment**
```python
# S-A-B-C-D-F tier system with weighted scoring
- GPU Performance: 60% weight (RTX 3080, RTX 3070, GTX 1660 Ti, etc.)
- CPU Performance: 25% weight (Intel i7-12700K, Ryzen 5 5600X, etc.)
- RAM Performance: 15% weight (16GB DDR4, 32GB DDR4, 8GB DDR4, etc.)
- Comprehensive hardware hierarchies with 50+ GPU/CPU models
```

**3. Steam-First Requirements Fetching**
```python
# Prioritized data source architecture
- Primary: Steam Store API (real-time, most current)
- Fallback: Local cache (offline support, curated database)
- Privacy-protected data sanitization throughout
- Automatic game ID resolution and requirement parsing
```

**4. MCP Server Implementation**
```python
# Model Context Protocol (MCP) server integration
- Uses Gradio for both UI and MCP server
- Async function support for real-time analysis
- Exposes game compatibility and hardware detection tools
- G-Assist direct integration capability
```


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

# Test official G-Assist protocol
python test/test_official_g_assist_protocol.py
```

**Rebuilding the Executable:**
```bash
# Rebuild the G-Assist plugin executable (required after code changes)
cd canrun
uv run python -m PyInstaller g-assist-plugin-canrun.spec

# The executable will be created at:
# canrun/dist/g-assist-plugin-canrun.exe

# Deploy to G-Assist:
copy "dist\g-assist-plugin-canrun.exe" "C:\ProgramData\NVIDIA Corporation\nvtopps\rise\plugins\canrun\g-assist-plugin-canrun.exe"

# This follows the official NVIDIA G-Assist naming convention: g-assist-plugin-<name>.exe
# This includes all dependencies and data files and can be used by G-Assist

# Alternative manual build command:
# pyinstaller --onefile --name g-assist-plugin-canrun --distpath . --add-data "src;src" --add-data "data;data" --add-data "config.json;." plugin.py
```

---


**Ready to see if your system can run any game? CanRun delivers A-tier performance analysis with official G-Assist protocol support!**

For technical support, feature requests, or contributions, visit [GitHub repository](https://github.com/leksval/canrun).
#AIonRTXHackathon
