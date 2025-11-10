# 📝 Deployment Notes

## Why `pyproject.toml` was renamed to `pyproject.toml.local`

Streamlit Cloud automatically detects `pyproject.toml` and uses Poetry to install dependencies. However, when Poetry tries to install the current project as a package, it fails because this is a Streamlit app, not a Python package.

**Solution:** Rename `pyproject.toml` to `pyproject.toml.local` so that:
- Streamlit Cloud uses `requirements.txt` for dependency installation
- Local development can still use Poetry (if desired) by referencing `pyproject.toml.local`

## For Streamlit Cloud Deployment

Streamlit Cloud will now:
1. Detect `requirements.txt` (since `pyproject.toml` is not present)
2. Install dependencies using `pip install -r requirements.txt`
3. Run the Streamlit app without trying to install it as a package

## For Local Development

### Option 1: Use pip (Recommended)
```bash
pip install -r requirements.txt
```

### Option 2: Use Poetry (if you prefer)
```bash
# Rename the file back temporarily
mv pyproject.toml.local pyproject.toml

# Install dependencies
poetry install --no-root

# Or install the project in editable mode
poetry install
```

### Option 3: Use conda (Current setup)
```bash
conda activate cme-13f-analysis
pip install -r requirements.txt
```

## Files

- `requirements.txt` - Used by Streamlit Cloud and pip
- `pyproject.toml.local` - Available for local Poetry usage (optional)
- `requirements_streamlit.txt` - Alternative requirements file (not used by Streamlit Cloud)

