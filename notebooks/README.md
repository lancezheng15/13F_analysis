# 13F Analysis Notebooks

This directory contains comprehensive Jupyter notebooks for analyzing 13F filings data. Each notebook focuses on different aspects of the analysis and provides interactive exploration capabilities.

## Notebook Overview

### 📊 Core Analysis Notebooks

1. **[01_data_exploration.ipynb](01_data_exploration.ipynb)**
   - **Purpose**: Initial data exploration and quality assessment
   - **Key Features**: Data loading, quality checks, basic statistics, time period analysis
   - **Best For**: Getting familiar with the dataset and understanding data structure
   - **Runtime**: ~5-10 minutes

2. **[02_investment_manager_analysis.ipynb](02_investment_manager_analysis.ipynb)**
   - **Purpose**: Deep dive into investment manager behavior and characteristics
   - **Key Features**: Manager concentration, portfolio characteristics, behavior patterns
   - **Best For**: Understanding institutional investment patterns and manager strategies
   - **Runtime**: ~10-15 minutes

3. **[03_security_holdings_analysis.ipynb](03_security_holdings_analysis.ipynb)**
   - **Purpose**: Analysis of individual securities and holdings patterns
   - **Key Features**: Top securities, holder concentration, voting authority analysis
   - **Best For**: Understanding which securities are most popular and how they're held
   - **Runtime**: ~10-15 minutes

4. **[04_market_intelligence.ipynb](04_market_intelligence.ipynb)**
   - **Purpose**: Market intelligence and futures-focused insights
   - **Key Features**: Sector analysis, institutional sentiment, risk assessment
   - **Best For**: Generating actionable market insights and identifying trends
   - **Runtime**: ~15-20 minutes

5. **[05_advanced_visualizations.ipynb](05_advanced_visualizations.ipynb)**
   - **Purpose**: Interactive visualizations and dashboards
   - **Key Features**: Plotly charts, network analysis, interactive filtering
   - **Best For**: Creating presentations and exploring data interactively
   - **Runtime**: ~20-25 minutes

### 🛠️ Utility Notebooks

6. **[00_setup_and_config.ipynb](00_setup_and_config.ipynb)**
   - **Purpose**: Environment setup and configuration
   - **Key Features**: Library imports, data loading functions, configuration settings
   - **Best For**: Setting up your analysis environment

7. **[99_export_and_reporting.ipynb](99_export_and_reporting.ipynb)**
   - **Purpose**: Export results and generate reports
   - **Key Features**: Data export, report generation, summary statistics
   - **Best For**: Finalizing analysis and creating deliverables

## Getting Started

### Prerequisites

1. **Environment Setup**: Ensure you have the conda environment activated:
   ```bash
   conda activate cme-13f-analysis
   ```

2. **Data Availability**: Make sure processed data exists in `../processed_data/`:
   - `combined_holdings.parquet`
   - `manager_summary.parquet`
   - `security_summary.parquet`
   - `sector_summary.parquet`

### Running the Notebooks

1. **Start Jupyter Lab**:
   ```bash
   jupyter lab
   ```

2. **Recommended Order**:
   - Start with `01_data_exploration.ipynb` to understand the data
   - Run `02_investment_manager_analysis.ipynb` for manager insights
   - Continue with `03_security_holdings_analysis.ipynb` for security analysis
   - Use `04_market_intelligence.ipynb` for market insights
   - Finish with `05_advanced_visualizations.ipynb` for interactive exploration

3. **Individual Analysis**: Each notebook can be run independently, but they build upon each other for comprehensive analysis.

## Key Features

### 📈 Analysis Capabilities

- **Data Quality Assessment**: Comprehensive data quality checks and validation
- **Manager Analysis**: Investment manager behavior, concentration, and performance metrics
- **Security Analysis**: Individual security popularity, holder patterns, and voting authority
- **Market Intelligence**: Sector trends, institutional sentiment, and futures implications
- **Interactive Visualizations**: Plotly charts, network analysis, and interactive dashboards

### 🎯 Use Cases

- **Investment Research**: Understanding institutional investment patterns
- **Market Analysis**: Identifying trends and sector preferences
- **Risk Assessment**: Analyzing concentration and diversification patterns
- **Futures Trading**: Gaining insights for futures market strategies
- **Academic Research**: Studying institutional investment behavior

### 📊 Output Formats

- **Interactive Charts**: Plotly visualizations for web-based exploration
- **Static Plots**: Matplotlib/Seaborn charts for reports and presentations
- **Data Exports**: CSV and Parquet files for further analysis
- **Summary Reports**: Automated insights and key findings

## Customization

### Configuration

Each notebook includes configuration sections where you can:
- Adjust analysis parameters
- Modify visualization settings
- Change date ranges and filters
- Customize output formats

### Extending Analysis

The notebooks are designed to be modular and extensible:
- Add new analysis sections
- Create custom visualizations
- Implement additional metrics
- Integrate with external data sources

## Performance Tips

### Memory Management

- **Large Datasets**: Use chunked processing for very large datasets
- **Memory Optimization**: Clear variables when not needed
- **Efficient Operations**: Use vectorized operations where possible

### Runtime Optimization

- **Parallel Processing**: Use multiprocessing for CPU-intensive operations
- **Caching**: Cache intermediate results for repeated analysis
- **Selective Loading**: Load only required data columns

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce dataset size or use chunked processing
2. **Missing Data**: Check data file paths and preprocessing completion
3. **Import Errors**: Ensure all dependencies are installed
4. **Visualization Issues**: Update Plotly and related libraries

### Getting Help

- Check the preprocessing logs in `../processed_data/`
- Review the configuration files in `../scripts/`
- Ensure the conda environment is properly activated

## Output and Results

### Generated Files

The notebooks generate various outputs:
- **Charts and Visualizations**: Saved as HTML and PNG files
- **Data Exports**: CSV and Parquet files with analysis results
- **Summary Reports**: Markdown and HTML reports
- **Interactive Dashboards**: Standalone HTML files

### Sharing Results

- **Interactive Dashboards**: Share HTML files for interactive exploration
- **Static Reports**: Export to PDF or share as HTML
- **Data Exports**: Share processed datasets for further analysis
- **Code Sharing**: Export notebooks as Python scripts

## Best Practices

### Analysis Workflow

1. **Start Small**: Begin with data exploration before complex analysis
2. **Validate Results**: Cross-check findings across multiple notebooks
3. **Document Insights**: Add notes and interpretations to your analysis
4. **Iterate**: Refine analysis based on initial findings

### Code Organization

- **Modular Functions**: Create reusable functions for common operations
- **Clear Documentation**: Add comments and markdown explanations
- **Version Control**: Track changes and maintain analysis history
- **Reproducibility**: Ensure analysis can be reproduced with different data

## Next Steps

After completing the analysis notebooks:

1. **Custom Analysis**: Create specialized notebooks for specific research questions
2. **Automation**: Develop automated reporting pipelines
3. **Integration**: Connect with real-time data sources
4. **Deployment**: Deploy interactive dashboards for ongoing monitoring

## Support

For questions or issues:
- Review the preprocessing documentation in `../scripts/README.md`
- Check the main project README in `../README.md`
- Ensure all dependencies are properly installed
- Verify data preprocessing completed successfully
