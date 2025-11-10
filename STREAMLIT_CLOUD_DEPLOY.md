# 🚀 Streamlit Cloud Deployment Guide

This guide explains how to deploy this 13F Analysis Dashboard to Streamlit Cloud.

## 📋 Prerequisites

1. **GitHub account** with the repository
2. **Streamlit Cloud account** (free tier available)
3. **Git LFS installed** on your local machine (for handling large files)

## 🔧 Setup Steps

### Step 1: Install Git LFS

```bash
# macOS
brew install git-lfs

# Linux
sudo apt-get install git-lfs

# Windows
# Download from: https://git-lfs.github.com/
```

### Step 2: Initialize Git LFS in your repository

```bash
# Navigate to your repository
cd /path/to/13F_analysis

# Initialize Git LFS
git lfs install

# Track parquet files with Git LFS
git lfs track "processed_data/*.parquet"
```

### Step 3: Commit the .gitattributes file

```bash
git add .gitattributes
git commit -m "Add Git LFS tracking for parquet files"
```

### Step 4: Add and commit your data files

```bash
# Add the processed data files (they will be tracked by LFS)
git add processed_data/*.parquet
git add processed_data/*.json
git commit -m "Add processed data files (tracked with Git LFS)"

# Push to GitHub
git push origin main
```

**Note:** The first push with LFS files may take longer as files are uploaded to LFS storage.

### Step 5: Verify files are tracked by LFS

```bash
# Check which files are tracked by LFS
git lfs ls-files

# You should see your parquet files listed
```

### Step 6: Deploy to Streamlit Cloud

1. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**
2. **Click "New app"**
3. **Select your repository** from GitHub
4. **Configure the app:**
   - **Main file path:** `streamlit_dashboard.py`
   - **Python version:** `3.10` (or higher)
   - **Dependencies file:** `requirements.txt` (or `pyproject.toml`)

5. **Click "Deploy"**

Streamlit Cloud will:
- Install dependencies from `requirements.txt` or `pyproject.toml`
- Clone your repository (including LFS files)
- Run `streamlit run streamlit_dashboard.py`

## ✅ Verification

After deployment, your dashboard should:
- Load data from `processed_data/*.parquet` files
- Display manager listings
- Show portfolio analytics
- Allow quarter-over-quarter comparisons

## 🔍 Troubleshooting

### Issue: "File not found" errors

**Solution:** Verify that parquet files are tracked by Git LFS:
```bash
git lfs ls-files
```

If files are not listed, add them:
```bash
git lfs track "processed_data/*.parquet"
git add processed_data/*.parquet
git commit -m "Track parquet files with LFS"
git push
```

### Issue: Large file upload fails

**Solution:** Ensure Git LFS is properly configured:
```bash
git lfs install
git lfs track "processed_data/*.parquet"
```

### Issue: Streamlit Cloud can't find dependencies

**Solution:** Ensure `requirements.txt` exists and includes all dependencies:
```bash
# Verify requirements.txt exists
cat requirements.txt

# If missing, create it from pyproject.toml
```

## 📊 File Size Limits

- **GitHub LFS Free Tier:** 1 GB storage, 1 GB/month bandwidth
- **GitHub File Size Limit:** 100 MB per file (files over 100 MB must use LFS)
- **Your largest file:** `combined_holdings.parquet` (~267 MB) - **requires LFS**

## 🎯 Alternative: External Storage (Optional)

If you prefer not to use Git LFS, you can:

1. **Store data in cloud storage** (S3, GCS, Azure Blob)
2. **Update the dashboard** to load data from URLs
3. **Use Streamlit secrets** to store credentials

See `STREAMLIT_EXTERNAL_STORAGE.md` for details (if needed).

## 📝 Notes

- **Raw data files** (`data/` directory) are NOT included in the repository (too large)
- **Only processed data** (`processed_data/*.parquet`) is tracked with Git LFS
- **Streamlit Cloud automatically handles Git LFS** - no additional configuration needed
- **Data files are cached** in the dashboard (1 hour TTL) for performance

## 🔄 Updating Data

When you update the processed data:

```bash
# 1. Regenerate processed data
python scripts/preprocess_13f_data.py

# 2. Add new/updated parquet files
git add processed_data/*.parquet

# 3. Commit and push
git commit -m "Update processed data"
git push origin main

# 4. Streamlit Cloud will automatically redeploy
```

