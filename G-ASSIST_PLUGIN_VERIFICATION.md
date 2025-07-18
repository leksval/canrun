# CanRun G-Assist Plugin - Verification & Testing Guide

## 🎯 Installation Verification Checklist

### ✅ Pre-Installation Requirements
- [ ] NVIDIA GeForce Experience is installed
- [ ] G-Assist feature is enabled in GeForce Experience
- [ ] You have Administrator privileges
- [ ] All required files are present:
  - [ ] `canrun-g-assist-plugin.exe`
  - [ ] `manifest.json`
  - [ ] `config.json`
  - [ ] `data/` directory (optional)
  - [ ] `src/` directory (optional)

### ✅ Installation Process
1. **Run the installer as Administrator:**
   ```cmd
   Right-click install_plugin.bat → "Run as administrator"
   ```

2. **Verify installation directory:**
   ```
   Expected location: C:\ProgramData\NVIDIA Corporation\nvtopps\rise\plugins\canrun\
   ```

3. **Check installed files:**
   ```
   plugins/canrun/
   ├── canrun-g-assist-plugin.exe
   ├── manifest.json
   ├── config.json
   ├── data/ (if present)
   └── src/ (if present)
   ```

### ✅ Plugin Testing
1. **Test plugin functionality:**
   ```cmd
   cd "C:\ProgramData\NVIDIA Corporation\nvtopps\rise\plugins\canrun"
   canrun-g-assist-plugin.exe --function detect_hardware
   ```
   
   **Expected output:** Clean JSON response with your system specifications

2. **Test compatibility check:**
   ```cmd
   canrun-g-assist-plugin.exe --function check_compatibility --game "Diablo 4"
   ```
   
   **Expected output:** JSON response with compatibility analysis

### ✅ G-Assist Integration Testing
1. **Restart NVIDIA GeForce Experience**
2. **Press Alt+G to activate G-Assist**
3. **Test voice commands:**
   - "Hey canrun, can I run Diablo 4?"
   - "Hey canrun, what are my system specs?"
   - "Hey canrun, can I run Cyberpunk 2077?"

## 🔍 Troubleshooting Guide

### Plugin Not Detected by G-Assist

**Check installation directory:**
```cmd
dir "C:\ProgramData\NVIDIA Corporation\nvtopps\rise\plugins\canrun"
```

**Verify executable name matches manifest:**
- Manifest declares: `canrun-g-assist-plugin.exe`
- File must exist with exact name

**Check plugin logs:**
```
Location: %USERPROFILE%\canrun-enhanced-plugin.log
```

### Plugin Discovery Logging

The plugin now includes enhanced discovery logging. Check the log file for:
- Plugin startup messages
- Working directory information
- Command line arguments
- Initialization status

### Common Issues & Solutions

1. **"G-Assist plugins directory not found"**
   - Ensure NVIDIA GeForce Experience is installed
   - Verify G-Assist is enabled
   - Run installer as Administrator

2. **"Failed to copy files"**
   - Run installer as Administrator
   - Check disk space
   - Verify antivirus isn't blocking

3. **Plugin works in terminal but not in G-Assist**
   - Restart GeForce Experience
   - Check plugin directory structure
   - Verify executable name matches manifest

## 🎮 Voice Command Examples

Once properly installed, you can use these voice commands:

### Compatibility Checks
- "Hey canrun, can I run [Game Name]?"
- "Hey canrun, will [Game Name] work on my system?"
- "Hey canrun, check compatibility for [Game Name]"

### Hardware Information
- "Hey canrun, what are my system specs?"
- "Hey canrun, show me my hardware"
- "Hey canrun, detect my hardware"

### Game Comparisons
- "Hey canrun, compare [Game 1] and [Game 2]"
- "Hey canrun, which runs better, [Game 1] or [Game 2]?"

### Steam Integration
- "Hey canrun, show Steam requirements for [Game Name]"
- "Hey canrun, Steam compare [Game Name]"

## 📊 Expected Response Format

G-Assist will display responses in a formatted chat window with:
- ✅/❌ Compatibility status
- 🎯 Performance analysis
- 💻 System specifications
- 🎮 Gaming performance tier
- 💡 Optimization tips
- 📊 Steam requirements comparison (when available)

## 🔧 Advanced Debugging

### Enable Debug Mode
Edit `config.json`:
```json
{
  "DEBUG_MODE": true,
  "LOG_LEVEL": "DEBUG"
}
```

### Manual Plugin Testing
```cmd
# Test auto-detection
canrun-g-assist-plugin.exe --function auto_detect --input "Can I run Elden Ring?"

# Test Steam comparison
canrun-g-assist-plugin.exe --function steam_compare --game "Cyberpunk 2077"

# Test game comparison
canrun-g-assist-plugin.exe --function compare_games --games "Diablo 4" "Cyberpunk 2077"
```

## 📝 Installation Log Analysis

The installer provides detailed feedback:
- ✅ Success indicators for each step
- ⚠️ Warnings for non-critical issues
- ❌ Errors that prevent installation
- 📁 Directory structure visualization
- 🧪 Plugin functionality test results

## 🎯 Success Criteria

Your plugin is correctly installed and working when:
1. Installation completes without errors
2. Plugin test returns valid JSON (no argument parser errors)
3. G-Assist recognizes "canrun" voice commands
4. Responses appear in G-Assist chat window
5. Plugin logs show successful initialization

## 🚀 What We Fixed

### Critical Issues Resolved:
1. **✅ Executable Name Mismatch**: Removed old `g-assist-plugin-python.exe`, using correct `canrun-g-assist-plugin.exe`
2. **✅ Installation Directory**: Fixed to use proper G-Assist path: `%PROGRAMDATA%\NVIDIA Corporation\nvtopps\rise\plugins\canrun\`
3. **✅ Directory Structure**: Proper plugin structure with `canrun` as invocation name
4. **✅ Response Termination**: Correct `<<END>>` marker (was already implemented)
5. **✅ Argument Parser Errors**: Fixed duplicate error messages in command line mode
6. **✅ Enhanced Logging**: Added plugin discovery logging for better debugging

### Installation Script Improvements:
- G-Assist directory validation
- Administrator privilege checking
- Comprehensive error handling
- File existence verification
- Plugin functionality testing

---

**Note:** If you encounter any issues, check the plugin log file at `%USERPROFILE%\canrun-enhanced-plugin.log` for detailed diagnostic information.