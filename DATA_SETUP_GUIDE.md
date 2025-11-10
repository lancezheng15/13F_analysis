# 📊 Data Setup Guide

## 🚨 Data Files Not in Git

**Data files are NOT included** (too large for GitHub). You need to provide your own data.

## 📁 Required Data Structure

Your project should have this structure:

```
cme-13f-analysis/
├── data/                          # Raw 13F data (NOT in git)
│   ├── 01JUN2025-31AUG2025_form13f/
│   │   ├── COVERPAGE.tsv
│   │   ├── INFOTABLE.tsv
│   │   ├── OTHERMANAGER.tsv
│   │   ├── OTHERMANAGER2.tsv
│   │   ├── SIGNATURE.tsv
│   │   ├── SUBMISSION.tsv
│   │   ├── SUMMARYPAGE.tsv
│   │   ├── FORM13F_metadata.json
│   │   └── FORM13F_readme.htm
│   └── [other_periods]/           # Additional quarters
├── processed_data/                # Processed data (NOT in git)
│   ├── combined_holdings.parquet
│   ├── manager_summary.parquet
│   ├── security_summary.parquet
│   └── sector_summary.parquet
├── scripts/                       # Processing scripts (in git)
├── notebooks/                     # Jupyter notebooks (in git)
├── streamlit_dashboard.py         # Streamlit dashboard (in git)
└── README.md                      # Project documentation (in git)
```

## 🔧 Setup Instructions

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd cme-13f-analysis
```

### Step 2: Set Up Environment
```bash
# Create conda environment
conda create -n cme-13f-analysis python=3.10
conda activate cme-13f-analysis

# Install dependencies
pip install -e .
```

### Step 3: Add Your Data Files

#### Option A: If you have raw 13F data
1. **Create the data directory:**
   ```bash
   mkdir -p data/01JUN2025-31AUG2025_form13f
   ```

2. **Add your TSV files** to the appropriate directory:
   - `COVERPAGE.tsv`
   - `INFOTABLE.tsv`
   - `OTHERMANAGER.tsv`
   - `OTHERMANAGER2.tsv`
   - `SIGNATURE.tsv`
   - `SUBMISSION.tsv`
   - `SUMMARYPAGE.tsv`
   - `FORM13F_metadata.json`
   - `FORM13F_readme.htm`

3. **Run preprocessing:**
   ```bash
   python scripts/preprocess_13f_data.py
   ```

#### Option B: If you have processed data
1. **Create the processed_data directory:**
   ```bash
   mkdir processed_data
   ```

2. **Add your processed files:**
   - `combined_holdings.parquet`
   - `manager_summary.parquet`
   - `security_summary.parquet`
   - `sector_summary.parquet`

### Step 4: Verify Setup
```bash
# Test data loading
python -c "
import pandas as pd
holdings = pd.read_parquet('processed_data/combined_holdings.parquet')
print(f'✅ Data loaded: {len(holdings):,} records')
"
```

### Step 5: Launch Dashboard
```bash
# Install dashboard dependencies
pip install streamlit plotly

# Launch the dashboard
streamlit run streamlit_dashboard.py
```

## 📊 Data Sources

### SEC 13F Filings
- **Source**: [SEC EDGAR Database](https://www.sec.gov/edgar/searchedgar/companysearch)
- **Format**: TSV files with quarterly data
- **Size**: ~1-2 GB per quarter
- **Update Frequency**: Quarterly (45 days after quarter end)

### Data Processing
The preprocessing script (`scripts/preprocess_13f_data.py`) converts raw TSV files into:
- **Combined Holdings**: All holdings across all managers and quarters
- **Manager Summary**: Aggregated statistics per manager
- **Security Summary**: Aggregated statistics per security
- **Sector Summary**: Aggregated statistics per sector

## 🔍 Data Validation

### Expected Data Sizes
- **Holdings**: 4-5 million records
- **Managers**: 8,000-10,000 unique managers
- **Securities**: 200,000+ unique securities
- **Quarters**: 2+ quarters for comparison

### Data Quality Checks
```bash
# Check data completeness
python -c "
import pandas as pd
holdings = pd.read_parquet('processed_data/combined_holdings.parquet')
print(f'Holdings: {len(holdings):,}')
print(f'Managers: {holdings[\"manager_name\"].nunique():,}')
print(f'Securities: {holdings[\"cusip\"].nunique():,}')
print(f'Quarters: {holdings[\"data_period\"].nunique()}')
"
```

## 🚀 Quick Start (If You Have Data)

If you already have the processed data files:

1. **Clone the repo:**
   ```bash
   git clone <your-repo-url>
   cd cme-13f-analysis
   ```

2. **Set up environment:**
   ```bash
   conda create -n cme-13f-analysis python=3.10
   conda activate cme-13f-analysis
   pip install -e .
   pip install streamlit plotly
   ```

3. **Add your data:**
   ```bash
   # Copy your processed data files to:
   # processed_data/combined_holdings.parquet
   # processed_data/manager_summary.parquet
   # processed_data/security_summary.parquet
   # processed_data/sector_summary.parquet
   ```

4. **Launch dashboard:**
   ```bash
   streamlit run streamlit_dashboard.py
   ```

## 📝 Data Notes

### File Formats
- **Raw Data**: TSV (Tab-Separated Values)
- **Processed Data**: Parquet (for fast loading)
- **Metadata**: JSON

### Data Periods
The data uses custom period naming:
- Format: `01JUN2025-31AUG2025_form13f`
- Represents: June 1 - August 31, 2025 (Q2 2025)

### Memory Requirements
- **Raw Data**: 2-4 GB
- **Processed Data**: 1-2 GB
- **RAM Usage**: 4-8 GB recommended

## 🆘 Troubleshooting

### Common Issues

1. **"No data found" error:**
   - Check that `processed_data/` directory exists
   - Verify parquet files are present
   - Check file permissions

2. **Memory errors:**
   - Close other applications
   - Use smaller data subsets for testing
   - Consider using a machine with more RAM

3. **Import errors:**
   - Ensure conda environment is activated
   - Install all dependencies: `pip install -e .`
   - Check Python version (3.10+ recommended)

### Getting Help
- Check the main README.md for project overview
- Review the preprocessing script for data requirements
- Test with the demo script: `python demo_dashboard.py`

## 📋 Data Checklist

Before running the dashboard, ensure you have:

- [ ] `processed_data/combined_holdings.parquet` (4M+ records)
- [ ] `processed_data/manager_summary.parquet` (8K+ managers)
- [ ] `processed_data/security_summary.parquet` (200K+ securities)
- [ ] `processed_data/sector_summary.parquet` (sector data)
- [ ] Conda environment activated
- [ ] All dependencies installed
- [ ] At least 4GB RAM available

## 🔄 Data Updates

To add new quarterly data:

1. **Add new raw data** to `data/[new_period]/`
2. **Run preprocessing** to include new period
3. **Restart dashboard** to see new data

The dashboard will automatically detect new quarters and make them available for comparison.
