"""
Configuration file for 13F data preprocessing.

This file contains configuration settings and constants used throughout
the preprocessing pipeline.
"""

# Data processing settings
MAX_QUARTERS_TO_PROCESS = 3  # Number of most recent quarters to process for dashboard

# File patterns
TSV_FILES = {
    'INFOTABLE.tsv': 'holdings',
    'COVERPAGE.tsv': 'coverpage', 
    'SUMMARYPAGE.tsv': 'summary',
    'SUBMISSION.tsv': 'submission',
    'SIGNATURE.tsv': 'signature',
    'OTHERMANAGER.tsv': 'other_managers',
    'OTHERMANAGER2.tsv': 'other_managers2'
}

# Column mappings for standardization
COLUMN_MAPPINGS = {
    'INFOTABLE': {
        'NAMEOFISSUER': 'issuer_name',
        'TITLEOFCLASS': 'security_class',
        'CUSIP': 'cusip',
        'FIGI': 'figi',
        'VALUE': 'market_value',
        'SSHPRNAMT': 'shares_or_principal',
        'SSHPRNAMTTYPE': 'amount_type',
        'PUTCALL': 'put_call',
        'INVESTMENTDISCRETION': 'investment_discretion',
        'OTHERMANAGER': 'other_manager',
        'VOTING_AUTH_SOLE': 'voting_sole',
        'VOTING_AUTH_SHARED': 'voting_shared',
        'VOTING_AUTH_NONE': 'voting_none'
    },
    'COVERPAGE': {
        'FILINGMANAGER_NAME': 'manager_name',
        'REPORTCALENDARORQUARTER': 'report_date',
        'REPORTTYPE': 'report_type',
        'ISAMENDMENT': 'is_amendment',
        'AMENDMENTNO': 'amendment_number',
        'AMENDMENTTYPE': 'amendment_type'
    },
    'SUMMARYPAGE': {
        'TABLEENTRYTOTAL': 'total_entries',
        'TABLEVALUETOTAL': 'total_value',
        'OTHERINCLUDEDMANAGERSCOUNT': 'other_managers_count'
    }
}

# Data cleaning parameters
CLEANING_CONFIG = {
    'min_market_value': 0,  # Minimum market value to keep
    'max_market_value': 1e12,  # Maximum market value (sanity check)
    'min_shares': 0,  # Minimum shares to keep
    'max_cusip_length': 9,  # Maximum CUSIP length
    'min_cusip_length': 6,  # Minimum CUSIP length
    'remove_duplicates': True,  # Remove duplicate records
    'fill_missing_dates': True,  # Fill missing dates with reasonable defaults
}

# Security type categorization rules
SECURITY_TYPE_RULES = {
    'COMMON_STOCK': ['COM', 'COMMON', 'ORD', 'SHARES', 'SHS'],
    'PREFERRED_STOCK': ['PREF', 'PREFERRED', 'PFD'],
    'DEBT': ['BOND', 'NOTE', 'DEBT', 'DEBENTURE'],
    'OPTION': ['OPTION', 'PUT', 'CALL', 'WARRANT'],
    'FUND': ['ETF', 'FUND', 'TRUST', 'MUTUAL', 'INDEX'],
    'WARRANT': ['WARRANT', 'WTS'],
    'UNIT': ['UNIT', 'UNITS'],
    'RIGHTS': ['RIGHTS', 'RTS']
}

# Export formats
EXPORT_FORMATS = ['parquet', 'csv', 'excel']

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'file': 'preprocessing.log'
}

# Performance settings
PERFORMANCE_CONFIG = {
    'chunk_size': 100000,  # Process data in chunks of this size
    'use_multiprocessing': False,  # Enable multiprocessing for large datasets
    'max_workers': 4,  # Maximum number of worker processes
    'memory_efficient': True,  # Use memory-efficient processing
}

# Data quality thresholds
QUALITY_THRESHOLDS = {
    'min_completeness': 0.8,  # Minimum data completeness ratio
    'max_duplicate_ratio': 0.05,  # Maximum allowed duplicate ratio
    'min_valid_cusips': 0.9,  # Minimum ratio of valid CUSIPs
    'max_outlier_ratio': 0.01,  # Maximum ratio of outliers
}

# Analysis-specific settings
ANALYSIS_CONFIG = {
    'top_n_managers': 100,  # Number of top managers to focus on
    'top_n_securities': 1000,  # Number of top securities to focus on
    'min_holding_value': 1000000,  # Minimum holding value for analysis ($1M)
    'sector_mapping_file': None,  # Path to sector mapping file (if available)
}

# Output file naming conventions
OUTPUT_NAMING = {
    'combined_holdings': 'combined_holdings',
    'manager_summary': 'manager_summary',
    'security_summary': 'security_summary',
    'sector_summary': 'sector_summary',
    'data_quality_report': 'data_quality_report',
    'processing_summary': 'processing_summary'
}

