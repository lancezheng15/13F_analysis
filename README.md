# 📊 CME 13F Portfolio Analysis Dashboard

Real-time Streamlit dashboard for monitoring investment manager portfolio changes between quarters, built with Streamlit and Plotly.

## 🚨 Data Setup Required

**Data files are NOT included** (too large for GitHub). See [DATA_SETUP_GUIDE.md](DATA_SETUP_GUIDE.md) for setup instructions.

## 🚀 Quick Start

```bash
# 1. Clone and setup environment
git clone <repo-url>
cd cme-13f-analysis
conda create -n cme-13f-analysis python=3.10
conda activate cme-13f-analysis

# 2. Install dependencies
pip install -e .
pip install streamlit plotly

# 3. Add your data files to processed_data/
# 4. Launch dashboard
streamlit run streamlit_dashboard.py
```

## 📊 Features

- **Real-time portfolio monitoring** with auto-refresh
- **Interactive manager search** (8,079+ managers)
- **Quarter-over-quarter comparison**
- **New position detection** and tracking
- **Visual analytics** with interactive charts

## 📁 Files

- `streamlit_dashboard.py` - Main Streamlit dashboard
- `demo_dashboard.py` - Demo script
- `check_data_setup.py` - Data validation
- `notebooks/` - Jupyter analysis notebooks
- `scripts/` - Data processing scripts
- `DATA_SETUP_GUIDE.md` - Setup instructions

## 🔧 Usage

```bash
# Check data setup
python check_data_setup.py

# Run demo
python demo_dashboard.py

# Launch dashboard
streamlit run streamlit_dashboard.py
```
