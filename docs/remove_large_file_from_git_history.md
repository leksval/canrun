# Removing Large Files from Git History

## Problem
Your repository contains `data/fps_model_calibrated.pkl` (116.48 MB) which exceeds GitHub's 100MB file size limit. This file is preventing you from pushing to GitHub.

## Solution Options

### Option 1: Using BFG Repo-Cleaner (Recommended - Easier and Faster)

BFG is a simpler, faster alternative to git filter-branch specifically designed for removing large files from Git history.

#### Step 1: Install BFG
```bash
# Download BFG jar file from https://rtyley.github.io/bfg-repo-cleaner/
# Or if you have Java installed, download directly:
curl -O https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar
```

#### Step 2: Clean the repository
```bash
# First, make a backup of your repository
cd ..
cp -r canrun canrun-backup

# Go back to your repository
cd canrun

# Remove files larger than 100M from history
java -jar ../bfg-1.14.0.jar --strip-blobs-bigger-than 100M

# Or specifically remove the problematic file
java -jar ../bfg-1.14.0.jar --delete-files fps_model_calibrated.pkl

# Clean up the repository
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

### Option 2: Using git filter-branch (Built-in but Slower)

This method uses Git's built-in tools but is slower and more complex.

#### Remove the specific file from all commits
```bash
# Remove the file from all commits
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch data/fps_model_calibrated.pkl" \
  --prune-empty --tag-name-filter cat -- --all

# Clean up references
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d

# Garbage collect
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Option 3: Using git filter-repo (Modern Alternative)

This is a modern replacement for filter-branch, faster and safer.

#### Step 1: Install git-filter-repo
```bash
pip install git-filter-repo
```

#### Step 2: Remove the file
```bash
git filter-repo --invert-paths --path data/fps_model_calibrated.pkl
```

## After Cleaning: Force Push to GitHub

**WARNING**: This will rewrite history. Make sure all team members are aware!

```bash
# Force push to overwrite the remote repository
git push origin main --force

# If you have other branches, push them too
git push origin --all --force

# Push tags if you have any
git push origin --tags --force
```

## Update .gitignore

Add these lines to your `.gitignore` file to prevent future issues:

```gitignore
# Machine Learning Models and Large Files
*.pkl
*.pickle
*.joblib
*.h5
*.hdf5
*.model
*.pth
*.pt
*.ckpt

# Specific data files
data/fps_model_calibrated.pkl
data/fps_calibrator.pkl

# Any file larger than 95MB (safety margin for GitHub's 100MB limit)
*.exe
!g-assist-plugin-canrun.exe  # Keep your specific exe if needed
```

## Alternative: Use Git LFS for Large Files

If you need to version control large files, use Git Large File Storage (LFS):

### Setup Git LFS
```bash
# Install Git LFS (one-time setup)
git lfs install

# Track large file types
git lfs track "*.pkl"
git lfs track "*.joblib"
git lfs track "*.model"

# Add .gitattributes to track these patterns
git add .gitattributes
git commit -m "Configure Git LFS for large model files"
```

### Migrate existing large files to LFS
```bash
git lfs migrate import --include="*.pkl" --everything
```

## Verification Steps

After completing the cleanup:

1. **Check repository size**:
   ```bash
   git count-objects -vH
   ```

2. **Verify the file is gone from history**:
   ```bash
   git log --all --full-history -- data/fps_model_calibrated.pkl
   # Should return nothing
   ```

3. **Test push**:
   ```bash
   git push origin main --dry-run
   ```

## Prevention Tips

1. **Use smaller model formats**: Consider using compressed formats or model quantization
2. **Store models externally**: Use cloud storage (S3, Google Drive) and download during setup
3. **Use Git LFS**: For files that must be versioned but are large
4. **Pre-commit hooks**: Add size checks to prevent large file commits

## Model Storage Alternatives

Instead of storing large models in Git, consider:

1. **Cloud Storage with Version Control**:
   ```python
   # In your code, download models on first run
   import urllib.request
   import os
   
   MODEL_URL = "https://your-storage.com/models/fps_model_calibrated.pkl"
   MODEL_PATH = "data/fps_model_calibrated.pkl"
   
   if not os.path.exists(MODEL_PATH):
       print("Downloading model...")
       urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
   ```

2. **Use Model Registry Services**:
   - MLflow Model Registry
   - DVC (Data Version Control)
   - Weights & Biases
   - Neptune.ai

3. **Compress Models**:
   ```python
   import joblib
   import gzip
   
   # Save compressed
   with gzip.open('data/fps_model_calibrated.pkl.gz', 'wb') as f:
       joblib.dump(model, f)
   
   # Load compressed
   with gzip.open('data/fps_model_calibrated.pkl.gz', 'rb') as f:
       model = joblib.load(f)
   ```

## Troubleshooting

### Error: "cannot create a new backup"
```bash
# Remove the backup refs
rm -rf .git/refs/original/
```

### Error: "remote rejected"
Make sure you have the correct permissions to force push. You may need to temporarily disable branch protection rules on GitHub.

### Still seeing the file in history
Make sure to clean all references:
```bash
git reflog expire --expire-unreachable=now --all
git gc --prune=now
```

## Quick Command Summary

For immediate resolution, run these commands in order:

```bash
# Backup your repo first!
cd ..
cp -r canrun canrun-backup
cd canrun

# Remove the file from history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch data/fps_model_calibrated.pkl" \
  --prune-empty --tag-name-filter cat -- --all

# Clean up
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push
git push origin main --force
```

## Next Steps

1. Update your code to handle missing model files gracefully
2. Implement a model download mechanism
3. Update documentation for other developers
4. Consider using Git LFS or external storage for future large files