# CanRun G-Assist Plugin

Quick setup guide for the CanRun G-Assist plugin.

## Installation

1. Copy the entire `plugin` folder to: `%APPDATA%\NVIDIA\G-Assist\plugins\canrun\`
2. Install dependencies: `pip install -r requirements.txt`
3. Build executable: `python build.py`

## Usage

- "Hey CanRun, can I run Baldur's Gate 3?"
- "What FPS will I get in Starfield on High?"
- "Scan my Steam library"
- "What should I upgrade for Cyberpunk?"

## Files

- `plugin.py`: Main G-Assist plugin entry point
- `manifest.json`: G-Assist function definitions
- `requirements.txt`: Python dependencies
- `build.py`: Build script