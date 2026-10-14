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
  
  [![Version](https://img.shields.io/badge/version-10.0-blue.svg)](https://github.com/canrun/canrun)
  [![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
  [![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
  [![G-Assist](https://img.shields.io/badge/G--Assist-Official%20Protocol%20Verified-brightgreen.svg)](https://www.nvidia.com/en-us/geforce/technologies/g-assist/)
  [![Steam API](https://img.shields.io/badge/Steam%20API-Integrated-blue.svg)](https://steamcommunity.com/dev)
  [![MCP Server](https://img.shields.io/badge/MCP%20Server-Enabled-brightgreen.svg)](https://developer.nvidia.com/mcp)

## 🚀 Overview

**CanRun** is an RTX/GTX-exclusive G-Assist plugin that instantly tells you if your PC can run any game with an advanced **S-A-B-C-D-F tier system** and enhanced Steam integration.

AVAILABLE on official Nvidia g-assist mod marketplace:  [mod.io/canrun](https://mod.io/g/g-assist/m/canrun2#description)

![R3](https://github.com/user-attachments/assets/79a68ba5-a4d9-493c-bea5-d455edadf627)

## ✨ Key Features

- **🎯 RTX/GTX Optimized**: Exclusively designed for RTX/GTX systems with G-Assist integration
- **🎮 CANRUN! Indicator**: Instant visual feedback when your system meets game requirements
 
- **⭐ S-A-B-C-D-F Tier System**: Advanced performance classification with weighted scoring
- 
- **🧠 AI-Powered Analysis**: G-Assist routes natural language to the plugin. The S-A-B-C-D-F tier still comes from local hardware scoring
- **📈 Frame Rates**: A Steam player sample is used when it has at least 30 reports. Otherwise a similar-PC estimate is used for games in the measured table. The range is wide enough that it is not presented as an exact FPS. The S-A-B-C-D-F tier still comes from local hardware scoring, not from that FPS number.
- **🔒 Local Matching**: GPU and CPU names stay on this PC. The frame-rate request sends only the Steam app id
- **🎯 Steam-First Data**: Game requirements come from the Steam Store API
- **🎯 Intelligent Game Matching**: Names are matched against Windows games from the last 30 days and the current 100 bestsellers, then the Steam store. A close or split title still lands on the published game.
- **📊 Smart Performance Prediction**: Comprehensive hardware hierarchies
- **💡 Intelligent Recommendations**: AI-generated optimization tips
- 
- **🤖 MCP Server**: Official Model Context Protocol (MCP) server for G-Assist integration
- **🔄 JSON-Based Configuration**: All benchmark data externalized for easy maintenance
- **🏃 Protocol V2 Plugin**: `plugin.py` on G-Assist SDK 3.1.0. Commands keep answering pings while a Steam lookup runs


## Recent changes

Version 10.0 adds measured frame rates and an honest range around them.

- **Steam player FPS.** `GetGameFrameRateStats` supplies the number when a hardware cluster has at least 30 reports. The line is labeled `Steam players`. The band is the measured average ± one standard deviation, and never narrower than ±8%.
- **Similar-PC estimate.** When that sample is missing and the game is in `data/training_benchmarks.json`, FPS comes from a per-game, per-resolution curve: `log(fps) = a + b * log(passmark)`. The line says `estimated from a similar PC, native, ray tracing off`, and the band is ±20%. Games that are not in that table, and have no Steam sample, show FPS as Unknown.
- **Display class.** If the monitor has substantially more pixels than the class used for the estimate (for example 3072×1728 treated as 1440p), the display line says which class the FPS number uses.
- **Local matching.** GPU and CPU names stay on this PC. The Steam request sends only the app id. Set `STEAM_WEBAPI_KEY` or `canrun_config.steam_webapi_key`. The key is removed from logs.
- **G-Assist SDK 3.1.0.** Protocol V2, JSON-RPC 2.0, length-prefixed messages. `manifest.json` uses `protocol_version` `"2.0"` and `executable` `plugin.py`.
- **Hardening.** The Gradio app binds to `127.0.0.1`. Game names are stripped of control characters and capped at 120 characters. Cache files cannot leave the cache directory, and symlinks and files over 256 KB are refused. Steam app ids must be numeric.
- **Estimate notice.** Every result starts with a notice that the check is an estimate and not a guarantee that the game will run or that the frame rate will match.
- **Recent releases and bestsellers.** A saved list of Windows games from the last 30 days plus the current 100 bestsellers is checked before the store search, and kept for 6 hours. Refreshing that list does not delay the answer. A typed name that is the same title, or a split of one long word in it, uses that game so the check is scored instead of coming back unknown. "Dawn Walker" matches "The Blood of Dawnwalker" when that game is in the list. A short word such as "within" does not attach to a longer title. The local Steam library is still checked first. If the list has no similar title, store search retries the joined spelling.

## 📦 G-Assist Plugin Installation

<<<<<<< HEAD
CanRun follows the current [NVIDIA G-Assist](https://github.com/NVIDIA/G-Assist) plugin contract: Protocol V2, JSON-RPC 2.0, and `gassist_sdk` 3.1.0. G-Assist launches `plugin.py` directly. The SDK lives in `libs/gassist_sdk` and is added to `PYTHONPATH` by the engine.
=======


Following [Official NVIDIA G-Assist Plugin Standards](https://github.com/NVIDIA/G-Assist), CanRun uses the correct plugin architecture:
>>>>>>> 0cc7b3582232700fa6c247a8085649b51afe5e8f

```
plugins/
└── canrun/                          # Plugin directory name = invocation name
    ├── plugin.py                    # Protocol V2 entry point
    ├── manifest.json                # protocol_version 2.0
    ├── config.json                  # Settings
    ├── requirements.txt             # Runtime packages installed into libs/
    ├── data/                        # Benchmark data
    └── libs/
        └── gassist_sdk/             # Vendored SDK 3.1.0
```

Install location:

```
%PROGRAMDATA%\NVIDIA Corporation\nvtopps\rise\plugins\canrun
```

### **🚀 AUTOMATIC INSTALLATION (RECOMMENDED)**

**Run the installation script as Administrator:**

```bash
cd canrun
install_g_assist_plugin.bat
```

The installer will:
- Copy the plugin sources, manifest, data, and vendored SDK
- Install `requirements.txt` into `libs/`
- Verify `plugin.py`, `manifest.json`, and `libs/gassist_sdk`

**Requirements:**
- Administrator privileges (required to write under `%PROGRAMDATA%`)
- Python 3.12+ on PATH
- NVIDIA G-Assist installed

### **📋 MANUAL INSTALLATION (ALTERNATIVE)**

**Step 1: Create the plugin directory**

```
mkdir "%PROGRAMDATA%\NVIDIA Corporation\nvtopps\rise\plugins\canrun"
```

**Step 2: Copy Plugin Files**

```bash
xcopy /E /I "canrun" "%PROGRAMDATA%\NVIDIA Corporation\nvtopps\rise\plugins\canrun"
pip install -r "%PROGRAMDATA%\NVIDIA Corporation\nvtopps\rise\plugins\canrun\requirements.txt" -t "%PROGRAMDATA%\NVIDIA Corporation\nvtopps\rise\plugins\canrun\libs"
```

**Step 3: Restart G-Assist**
1. Close NVIDIA App / G-Assist completely
2. Restart NVIDIA App
3. Enable G-Assist if it is not already enabled

**Step 4: Test Plugin**
Try these commands:
- "can I run elden ring"
- "canrun cyberpunk 2077"

### **Plugin Architecture Notes**
- **Directory Name**: Must match the plugin invocation name (`canrun`)
- **Executable**: `plugin.py` (`manifest.json` field `executable`)
- **Protocol**: `protocol_version` `"2.0"`. Messages are JSON-RPC 2.0 with a 4-byte big-endian length prefix
- **Persistent**: `true`, so the engine can ping the plugin while a Steam lookup runs
- **CLI Mode**: `python plugin.py canrun "cyberpunk 2077" --json`

### **🔧 Troubleshooting**

If the plugin is not detected by G-Assist:

1. **Verify Installation Path**: Run the installation script again
2. **Check File Permissions**: Confirm `plugin.py`, `manifest.json`, and `libs/gassist_sdk` were copied
3. **Restart G-Assist**: Completely close and restart NVIDIA App/G-Assist
4. **Check Logs**: Review `%USERPROFILE%\canrun_plugin.log` and `gassist_sdk.log` in the plugin directory
5. **Test CLI Mode**: Run `python plugin.py canrun "cyberpunk 2077" --json` from the plugin directory

A Steam-backed FPS line looks like `93-139 FPS (Steam players)` or `93-139 FPS (estimated from a similar PC, native, ray tracing off)`. A heavier monitor adds `FPS estimate uses 1440p` on the display line. When neither Steam nor the measured table has the game, FPS is Unknown.

### Steam Web API key

Frame-rate stats require a key from [Steam Web API](https://steamcommunity.com/dev). Put it in the environment or in `config.json`:

```json
"canrun_config": {
  "steam_webapi_key": "your-key-here"
}
```

```bash
set STEAM_WEBAPI_KEY=your-key-here
```

Leave the key out of git. An empty `steam_webapi_key` is fine: requirements still load, and a game in the measured table can still show a similar-PC estimate. Without a key and without a table row, FPS is Unknown.

## Installation and Setup

### Prerequisites
- Python 3.12+ (NVIDIA G-Assist requirement)
- Windows 10/11 (for hardware detection)
- NVIDIA GPU recommended for full G-Assist integration
- **uv package manager** for the development install below

### Install uv Package Manager
If you don't have `uv` installed:
```bash
# Install uv package manager
pip install uv
```

### Option 1: G-Assist plugin (Protocol V2)

```bash
# Clone repository
git clone https://github.com/leksval/canrun
cd canrun

# Install the plugin for G-Assist (Administrator)
install_g_assist_plugin.bat

# Protocol check and CLI
uv run python -m pytest test/test_official_g_assist_protocol.py test/test_g_assist_communication.py -v
uv run python plugin.py initialize --json
uv run python plugin.py canrun "cyberpunk 2077" --json
```


### Option 2: Full ML Pipeline Development

```bash
# Clone repository
git clone https://github.com/leksval/canrun
cd canrun

# Install full development environment with ML training capabilities using pyproject
uv sync --all-extras

# Run comprehensive test suite
uv run pytest test/ -v

# Train unified stacking ensemble ML model
uv run python src/train_unified_ml_model.py

# Test plugin functionality
uv run python plugin.py canrun "cyberpunk 2077" --json
```

### G-Assist MCP Integration

G-Assist can automatically discover and use the CanRun MCP server when both are running. This enables advanced conversational interactions like:

- "G-Assist, ask CanRun if I can play Starfield"
- "G-Assist, check if my system meets Diablo 4 requirements"
- "G-Assist, what's my gaming hardware like?"

## 🧪 Running Tests

**Primary Test Command (Recommended):**

```bash
# Navigate to canrun directory first
cd canrun

# Run all tests with pytest using uv
uv run python -m pytest test/ -v

# Protocol V2, Steam FPS cases, and security edge cases
uv run python -m pytest test/test_official_g_assist_protocol.py test/test_g_assist_communication.py test/test_steam_fps_cases.py test/test_edge_cases.py -v
```

## 🤖 MCP Server Functionality

CanRun now includes a full-featured **Model Context Protocol (MCP) server** that allows G-Assist to directly integrate with the CanRun compatibility engine. This provides seamless AI-assisted game compatibility checking through the official NVIDIA MCP standard.

### MCP Tools and Capabilities

The MCP server exposes the following tools to G-Assist:

- **check_game_compatibility**: Analyze if a specific game can run on the current system
  - Input: Game name (e.g., "Cyberpunk 2077")
  - Output: Detailed compatibility analysis with performance tier

- **detect_hardware**: Provides comprehensive hardware detection for gaming systems
  - Output: Detailed hardware specifications focused on gaming performance

### Running the MCP Server

```bash
# Start the local UI. It binds to 127.0.0.1 only.
python app.py
```


## 🧠 ML Pipeline

The stacking ensemble below is kept for training experiments. The FPS number shown in G-Assist comes from Steam samples and the similar-PC curve described above, not from this model.

### **Final Production Accuracy:**
- **Mean Absolute Error (MAE):** 13.79 FPS
- **R² Score:** 0.9558 (95.58% variance explained)
- **RMSE:** 11.01 FPS

### **Calibrated Model Performance:**
- **Calibrated MAE:** 3.07 FPS
- **Calibrated R²:** 0.9941 (99.41% variance explained)

### **Model Details:**

#### **Training Configuration:**
- **Model Type:** Unified_Stacking_Ensemble_2025
- **Base Models:** 7 ensemble models (XGBoost GPU, LightGBM GPU, CatBoost, Random Forest, etc.)
- **Training Samples:** 143
- **Test Samples:** 36
- **Training Time:** 81.3 seconds
- **Features:** 80 engineered features

#### **Quality Indicators:**
- **Overfitting Gap:** 0.0085 (very low - indicates good generalization)
- **Domain Correction Factor:** 0.88 (accounts for real-world variance)
- **GPU Acceleration:** CatBoost GPU enabled

### **Training Pipeline**
```bash
# Retrain with latest benchmarks
cd canrun
uv run python src/train_unified_ml_model_2025.py

# Expected output:
# ✅ Unified_Stacking_Ensemble_2025 v7.0_2025_Edition
# ✅ 99.42% R² after calibration (SOTA performance)
# ✅ Model saved to canrun/src/ml_fps_model.json
```

### **Training Data**
- **1,848 benchmarks** across 14 games, 44 GPUs (GTX 960 → RTX 5090)
- **Multi-resolution**: 1080p, 1440p, 4K validated data
- **Real-world sources**: Verified through web research + curated database
- **Feature engineering**: Polynomial interactions, PassMark scores, VRAM ratios


---

## 🚀 **GPU-Accelerated Training**

For developers wanting to retrain the ML model with CUDA acceleration:

### **CUDA Setup (Recommended for RTX/GTX users)**

```bash
# Step 1: Install CUDA-enabled PyTorch (for RTX with CUDA 13.0)
cd canrun
uv run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Step 2: Install XGBoost with GPU support
uv add xgboost

# Step 3: Optional - Install cuML for maximum GPU acceleration
# conda install -c rapidsai -c conda-forge cuml

# Step 4: Run GPU-accelerated training
uv run python src/train_fps_predictor_ml_model.py
```


### **XGBoost CUDA Configuration (Verified Working):**
```python
# Modern XGBoost syntax (recommended)
XGBRegressor(device='cuda', tree_method='hist', n_estimators=500)

# Console output confirms: "XGBoost is running on: cuda:0"
```
### **Expected GPU Benefits:**
- **XGBoost GPU**: `device='cuda'` with `tree_method='hist'` - **3-5x faster** tree training
- **cuML**: GPU-accelerated RandomForest - **10-50x faster** on RTX cards
- **PyTorch CUDA**: Tensor operations on GPU for enhanced feature engineering
- **Automatic Fallback**: CPU training if CUDA unavailable

# Features

**1. Advanced Performance Assessment**
# S-A-B-C-D-F tier system with weighted scoring
- GPU Performance: 60% weight (RTX 3080, RTX 3070, GTX 1660 Ti, etc.)
- CPU Performance: 25% weight (Intel i7-12700K, Ryzen 5 5600X, etc.)
- RAM Performance: 15% weight (16GB DDR4, 32GB DDR4, 8GB DDR4, etc.)
- Comprehensive hardware hierarchies with 50+ GPU/CPU models

**2. Steam-First Requirements Fetching**
# Prioritized data source architecture
- Primary: Steam Store API (real-time, most current)
- Fallback: Local cache (offline support, curated database)
- Privacy-protected data sanitization throughout
- Automatic game ID resolution and requirement parsing
- Better title matching uses a saved list of Windows games from the last 30 days and the current 100 bestsellers. A similar or split name in that list is used, so the check is scored instead of left unknown. Refreshing the list does not delay the answer. The list is cached for 6 hours. The local Steam library is checked first.

**3. Frame-Rate Prediction**

# Steam sample, then a similar PC, then Unknown
- Steam players: measured average ± one standard deviation, at least ±8%, when the cluster has 30 or more reports
- Similar PC: per-game curve at 1080p, 1440p, or 4K, shown as ±20% and labeled native with ray tracing off
- Unknown when the game is missing from both Steam and `data/training_benchmarks.json`
- The display line names the resolution class when the monitor is heavier than that class
- S-A-B-C-D-F tier stays the local weighted hardware score

**4. MCP Server Implementation**
# Model Context Protocol (MCP) server integration
- Uses Gradio for both UI and MCP server, bound to 127.0.0.1
- Async function support for real-time analysis
- Exposes game compatibility and hardware detection tools
- G-Assist direct integration capability

---
## Live Demo, download and test locally for real hardvare evaluation as MCP server
[LINK](https://huggingface.co/spaces/grasant/canrun)
![HF Space Demo](image.png)

---


**Ready to see if your system can run any game? CanRun delivers A-tier performance analysis with official G-Assist protocol support!**

For technical support, feature requests, or contributions, visit [GitHub repository](https://github.com/leksval/canrun).
#AIonRTXHackathon
