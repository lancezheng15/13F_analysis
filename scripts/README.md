# 13F Data Preprocessing Scripts

This directory contains scripts for preprocessing and analyzing Form 13F filing data.

## Files

- `preprocess_13f_data.py` - Main preprocessing script
- `test_preprocessing.py` - Test script to validate preprocessing functionality
- `config.py` - Configuration settings and constants
- `README.md` - This file

## Quick Start

### 1. Install Dependencies

```bash
# From the project root
uv pip install -e .
```

### 2. Test the Preprocessing

```bash
# Run tests to ensure everything works
python scripts/test_preprocessing.py
```

### 3. Run Full Preprocessing

```bash
# Process all available data periods
python scripts/preprocess_13f_data.py
```

## What the Preprocessing Does

### Data Loading
- Automatically discovers all data periods in the `data/` directory
- Loads all TSV files for each period (INFOTABLE, COVERPAGE, SUMMARYPAGE, etc.)
- Handles large datasets efficiently with progress tracking

### Data Cleaning
- Standardizes column names across all tables
- Cleans and validates CUSIP codes, issuer names, and security classes
- Converts data types (dates, numbers) appropriately
- Removes invalid or duplicate records
- Creates derived columns (security types, voting authority totals)

### Data Merging
- Combines holdings data with manager information
- Merges with summary statistics
- Adds period and processing metadata

### Aggregation
- Creates manager-level summaries (total holdings, unique securities, etc.)
- Creates security-level summaries (total market value, holder count, etc.)
- Creates sector-level summaries (by security type)

### Export
- Saves processed data in multiple formats (Parquet, CSV, Excel)
- Generates comprehensive processing summary
- Creates data quality reports

## Output Files

The preprocessing creates several output files in the `processed_data/` directory:

- `combined_holdings.parquet` - All holdings data from all periods
- `manager_summary.parquet` - Aggregated data by manager
- `security_summary.parquet` - Aggregated data by security
- `sector_summary.parquet` - Aggregated data by security type
- `processing_summary.json` - Processing metadata and statistics

## Configuration

Edit `config.py` to customize:
- Data cleaning parameters
- Security type categorization rules
- Export formats
- Performance settings
- Data quality thresholds

## Data Quality Features

The preprocessing includes several data quality checks:
- Completeness validation
- Duplicate detection
- CUSIP validation
- Outlier detection
- Data consistency checks

## Performance

The script is designed to handle large datasets efficiently:
- Processes data in chunks for memory efficiency
- Uses progress bars for long-running operations
- Supports parallel processing (configurable)
- Optimized for datasets with millions of records

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `chunk_size` in config.py
2. **File not found**: Ensure data directories follow the expected naming pattern
3. **Encoding issues**: The script handles UTF-8 encoding automatically
4. **Date parsing errors**: Check date formats in the raw data

### Logs

All processing activities are logged to:
- Console output (INFO level and above)
- `preprocessing.log` file (detailed logging)

## Next Steps

After preprocessing, you can:
1. Load the processed data into Jupyter notebooks for analysis
2. Create visualizations and reports
3. Build predictive models
4. Generate market intelligence insights

## Support

For issues or questions:
1. Check the logs for error messages
2. Run the test script to identify problems
3. Review the configuration settings
4. Check data file formats and structure

