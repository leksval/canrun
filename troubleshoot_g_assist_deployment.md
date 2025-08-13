# G-Assist Plugin Deployment Troubleshooting

## Issue: CanRun Plugin Not Appearing in G-Assist List

Based on your screenshot showing other plugins (corsair, google, logiled, msi, nanoleaf) but missing CanRun, this indicates a deployment issue.

## Step 1: Verify File Deployment

Check if files were actually copied to the correct location:

```cmd
# Open Command Prompt as Administrator and run:
dir "C:\Program Files\NVIDIA Corporation\NVIDIA app\plugins\canrun"
```

**Expected Output:**
```
manifest.json
plugin.py
g-assist-plugin-canrun.exe
```

If directory doesn't exist or files are missing, the deployment failed.

## Step 2: Verify File Permissions

Ensure files have correct permissions:

```cmd
# Check file permissions
icacls "C:\Program Files\NVIDIA Corporation\NVIDIA app\plugins\canrun\*"
```

Files should be readable by all users.

## Step 3: Test Plugin Executable

Verify the executable works:

```cmd
# Navigate to plugin directory
cd "C:\Program Files\NVIDIA Corporation\NVIDIA app\plugins\canrun"

# Test executable
g-assist-plugin-canrun.exe canrun "Cyberpunk 2077"
```

**Expected:** Should return game compatibility analysis.
**If Error:** Executable is corrupted or dependencies missing.

## Step 4: Validate Manifest JSON

Check manifest syntax:

```cmd
# In plugin directory, test JSON validity
type manifest.json
```

Common issues:
- Missing closing braces
- Invalid JSON syntax
- Incorrect executable name reference

## Step 5: Check NVIDIA App Plugin Discovery

Restart NVIDIA App completely:

```cmd
# Kill all NVIDIA processes
taskkill /f /im "NVIDIA app.exe"
taskkill /f /im "nvcontainer.exe"

# Wait 5 seconds, then restart NVIDIA App
start "" "C:\Program Files\NVIDIA Corporation\NVIDIA app\nvidia_app.exe"
```

## Step 6: Manual Deployment Verification

If automated deployment failed, manually copy files:

1. **Open File Explorer as Administrator:**
   - Win+R → `explorer` → Ctrl+Shift+Enter → Yes

2. **Navigate to:** `C:\Program Files\NVIDIA Corporation\NVIDIA app\plugins\`

3. **Create canrun directory** if it doesn't exist

4. **Copy these 3 files** from `d:\projects\canrun\canrun\`:
   - `manifest.json`
   - `plugin.py` 
   - `g-assist-plugin-canrun.exe`

## Step 7: Check G-Assist Logs

Look for plugin loading errors:

```cmd
# Check NVIDIA App logs
dir "%LOCALAPPDATA%\NVIDIA Corporation\NVIDIA app\logs"
```

Look for recent log files mentioning plugin errors.

## Step 8: Alternative Plugin Directory

Some installations use different paths. Try:

```cmd
# Check alternative locations
dir "C:\Users\%USERNAME%\AppData\Local\NVIDIA Corporation\NVIDIA app\plugins"
dir "C:\ProgramData\NVIDIA Corporation\NVIDIA app\plugins"
```

## Step 9: Verify G-Assist Version

Ensure you have G-Assist enabled:

1. Open NVIDIA App
2. Go to Settings
3. Verify G-Assist is enabled
4. Check for NVIDIA App updates

## Common Solutions

### Solution 1: Re-run Deployment Script as Administrator
```cmd
# Right-click deploy_plugin.bat → "Run as administrator"
cd d:\projects\canrun\canrun
deploy_plugin.bat
```

### Solution 2: Manual File Copy
If deployment script fails, manually copy the 3 required files to the plugin directory.

### Solution 3: Check Executable Dependencies
The plugin executable might be missing required DLLs:

```cmd
# Test in canrun development directory first
cd d:\projects\canrun\canrun
g-assist-plugin-canrun.exe canrun "test game"
```

If this works but deployed version doesn't, it's a deployment issue.

## Next Steps

Please run Step 1 first and report:
1. Does the canrun directory exist?
2. Are all 3 files present?
3. What are the file sizes?

This will help identify if it's a deployment issue or a plugin detection problem.