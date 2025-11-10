#!/bin/bash

echo "🚀 Launching 13F Portfolio Dashboard..."

# Activate conda environment (non-interactive shells)
eval "$(conda shell.bash hook)" || { echo "❌ Run 'conda init bash' then restart your shell."; exit 1; }
conda activate cme-13f-analysis || { echo "❌ Failed to activate 'cme-13f-analysis' env."; exit 1; }

# Launch dashboard
echo "🌐 Starting dashboard at http://localhost:8501"
streamlit run streamlit_dashboard.py
