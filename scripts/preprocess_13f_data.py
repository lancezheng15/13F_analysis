#!/usr/bin/env python3
"""
13F Filings Data Preprocessing Script

This script processes raw 13F filing data from multiple time periods,
cleans and standardizes the data, and creates analysis-ready datasets.

Author: CME 13F Analysis Team
Date: 2025
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
from collections import defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Form13FPreprocessor:
    """
    A comprehensive preprocessor for Form 13F filing data.
    
    This class handles:
    - Loading data from multiple time periods
    - Data cleaning and standardization
    - Merging related tables
    - Creating analysis-ready datasets
    - Exporting processed data in multiple formats
    """
    
    def __init__(self, data_dir: str = "data", output_dir: str = "processed_data"):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Directory containing raw 13F data
            output_dir: Directory to save processed data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.raw_data = {}
        self.processed_data = {}
        
        # Column mappings for standardization
        self.column_mappings = {
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
    
    def discover_data_periods(self):
        """Discover the most recent quarters of data periods based on configuration."""
        from config import MAX_QUARTERS_TO_PROCESS
        
        periods = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and 'form13f' in item.name.lower():
                periods.append(item.name)
        
        # Sort periods by date (extract start date from period name)
        def extract_date(period_name):
            try:
                # Extract start date part (e.g., '01JUN2025' from '01JUN2025-31AUG2025_form13f')
                start_part = period_name.split('-')[0]
                return pd.to_datetime(start_part, format='%d%b%Y')
            except:
                return pd.Timestamp.min
        
        sorted_periods = sorted(periods, key=extract_date)
        recent_periods = sorted_periods[-MAX_QUARTERS_TO_PROCESS:] if len(sorted_periods) >= MAX_QUARTERS_TO_PROCESS else sorted_periods
        
        logger.info(f"Using most recent {len(recent_periods)} quarters: {recent_periods}")
        return recent_periods
    
    def load_period_data(self, period: str):
        """
        Load all TSV files for a specific time period.
        
        Args:
            period: Directory name for the time period
        """
        period_path = self.data_dir / period
        logger.info(f"Loading data for period: {period}")
        
        period_data = {}
        
        # Define file mappings
        file_mappings = {
            'INFOTABLE.tsv': 'holdings',
            'COVERPAGE.tsv': 'coverpage',
            'SUMMARYPAGE.tsv': 'summary',
            'SUBMISSION.tsv': 'submission',
            'SIGNATURE.tsv': 'signature',
            'OTHERMANAGER.tsv': 'other_managers',
            'OTHERMANAGER2.tsv': 'other_managers2'
        }
        
        for filename, key in file_mappings.items():
            filepath = period_path / filename
            if filepath.exists():
                try:
                    logger.info(f"Loading {filename}...")
                    df = pd.read_csv(filepath, sep='\t', low_memory=False)
                    period_data[key] = df
                    logger.info(f"Loaded {len(df):,} records from {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
            else:
                logger.warning(f"File not found: {filename}")
        
        self.raw_data[period] = period_data
        return period_data
    
    def clean_holdings_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize holdings data.
        
        Args:
            df: Raw holdings dataframe
            
        Returns:
            Cleaned holdings dataframe
        """
        logger.info("Cleaning holdings data...")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Standardize column names
        if 'NAMEOFISSUER' in df_clean.columns:
            df_clean = df_clean.rename(columns=self.column_mappings['INFOTABLE'])
        
        # Clean issuer names
        if 'issuer_name' in df_clean.columns:
            df_clean['issuer_name'] = df_clean['issuer_name'].str.strip().str.upper()
            df_clean['issuer_name_clean'] = df_clean['issuer_name'].str.replace(r'[^\w\s]', '', regex=True)
        
        # Clean security class
        if 'security_class' in df_clean.columns:
            df_clean['security_class'] = df_clean['security_class'].str.strip().str.upper()
        
        # Clean CUSIP (remove any non-alphanumeric characters)
        if 'cusip' in df_clean.columns:
            df_clean['cusip'] = df_clean['cusip'].str.replace(r'[^A-Z0-9]', '', regex=True)
        
        # Convert numeric columns
        numeric_columns = ['market_value', 'shares_or_principal', 'voting_sole', 'voting_shared', 'voting_none']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Handle missing values in critical columns
        df_clean = df_clean.dropna(subset=['issuer_name', 'cusip', 'market_value'])
        
        # Create derived columns
        df_clean['total_voting_authority'] = (
            df_clean['voting_sole'].fillna(0) + 
            df_clean['voting_shared'].fillna(0) + 
            df_clean['voting_none'].fillna(0)
        )
        
        # Create security type categories
        if 'security_class' in df_clean.columns:
            df_clean['security_type'] = df_clean['security_class'].apply(self._categorize_security_type)
        
        # Add data quality flags
        df_clean['has_figi'] = df_clean['figi'].notna()
        df_clean['is_option'] = df_clean['put_call'].notna()
        df_clean['has_other_manager'] = df_clean['other_manager'].notna()
        
        logger.info(f"Cleaned holdings data: {len(df_clean):,} records")
        return df_clean
    
    def _categorize_security_type(self, security_class: str) -> str:
        """Categorize security class into broader types."""
        if pd.isna(security_class):
            return 'UNKNOWN'
        
        security_class = security_class.upper()
        
        if any(word in security_class for word in ['COM', 'COMMON', 'ORD']):
            return 'COMMON_STOCK'
        elif any(word in security_class for word in ['PREF', 'PREFERRED']):
            return 'PREFERRED_STOCK'
        elif any(word in security_class for word in ['BOND', 'NOTE', 'DEBT']):
            return 'DEBT'
        elif any(word in security_class for word in ['OPTION', 'PUT', 'CALL']):
            return 'OPTION'
        elif any(word in security_class for word in ['ETF', 'FUND', 'TRUST']):
            return 'FUND'
        else:
            return 'OTHER'
    
    def clean_coverpage_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize coverpage data."""
        logger.info("Cleaning coverpage data...")
        
        df_clean = df.copy()
        
        # Standardize column names
        if 'FILINGMANAGER_NAME' in df_clean.columns:
            df_clean = df_clean.rename(columns=self.column_mappings['COVERPAGE'])
        
        # Clean manager names
        if 'manager_name' in df_clean.columns:
            df_clean['manager_name'] = df_clean['manager_name'].str.strip()
            df_clean['manager_name_clean'] = df_clean['manager_name'].str.replace(r'[^\w\s]', '', regex=True)
        
        # Convert dates
        date_columns = ['report_date']
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        # Clean report types
        if 'report_type' in df_clean.columns:
            df_clean['report_type'] = df_clean['report_type'].str.strip().str.upper()
        
        # Handle amendments
        if 'is_amendment' in df_clean.columns:
            df_clean['is_amendment'] = df_clean['is_amendment'].fillna('N')
            df_clean['is_amendment_bool'] = df_clean['is_amendment'] == 'Y'
        
        logger.info(f"Cleaned coverpage data: {len(df_clean):,} records")
        return df_clean
    
    def merge_period_data(self, period: str):
        """
        Merge all tables for a specific period into a comprehensive dataset.
        
        Args:
            period: Time period to process
        """
        logger.info(f"Merging data for period: {period}")
        
        if period not in self.raw_data:
            logger.error(f"No data loaded for period: {period}")
            return None
        
        period_data = self.raw_data[period]
        
        # Start with holdings data as the base
        if 'holdings' not in period_data:
            logger.error(f"No holdings data for period: {period}")
            return None
        
        # Clean holdings data
        holdings_clean = self.clean_holdings_data(period_data['holdings'])
        
        # Merge with coverpage data
        if 'coverpage' in period_data:
            coverpage_clean = self.clean_coverpage_data(period_data['coverpage'])
            
            # Merge on ACCESSION_NUMBER
            merged_df = holdings_clean.merge(
                coverpage_clean[['ACCESSION_NUMBER', 'manager_name', 'report_date', 'report_type', 'is_amendment_bool']],
                on='ACCESSION_NUMBER',
                how='left'
            )
        else:
            merged_df = holdings_clean.copy()
            merged_df['manager_name'] = None
            merged_df['report_date'] = None
            merged_df['report_type'] = None
            merged_df['is_amendment_bool'] = False
        
        # Merge with summary data
        if 'summary' in period_data:
            summary_clean = period_data['summary'].copy()
            if 'TABLEENTRYTOTAL' in summary_clean.columns:
                summary_clean = summary_clean.rename(columns=self.column_mappings['SUMMARYPAGE'])
            
            merged_df = merged_df.merge(
                summary_clean[['ACCESSION_NUMBER', 'total_entries', 'total_value', 'other_managers_count']],
                on='ACCESSION_NUMBER',
                how='left'
            )
        
        # Add period information
        merged_df['data_period'] = period
        
        # Add processing timestamp
        merged_df['processed_at'] = datetime.now()
        
        logger.info(f"Merged data for {period}: {len(merged_df):,} records")
        return merged_df
    
    def create_aggregated_datasets(self, all_periods_data: pd.DataFrame):
        """
        Create aggregated datasets for analysis.
        
        Args:
            all_periods_data: Combined data from all periods
        """
        logger.info("Creating aggregated datasets...")
        
        # Manager-level aggregations
        manager_summary = all_periods_data.groupby(['manager_name', 'data_period']).agg({
            'market_value': ['sum', 'count', 'mean'],
            'issuer_name': 'nunique',
            'cusip': 'nunique',
            'total_voting_authority': 'sum',
            'is_amendment_bool': 'sum'
        }).round(2)
        
        manager_summary.columns = [
            'total_market_value', 'total_holdings', 'avg_holding_value',
            'unique_issuers', 'unique_securities', 'total_voting_shares', 'amendment_count'
        ]
        manager_summary = manager_summary.reset_index()
        
        # Security-level aggregations
        security_summary = all_periods_data.groupby(['issuer_name', 'cusip', 'data_period']).agg({
            'market_value': ['sum', 'count'],
            'shares_or_principal': 'sum',
            'manager_name': 'nunique',
            'voting_sole': 'sum',
            'voting_shared': 'sum',
            'voting_none': 'sum'
        }).round(2)
        
        security_summary.columns = [
            'total_market_value', 'holding_count', 'total_shares',
            'unique_managers', 'total_voting_sole', 'total_voting_shared', 'total_voting_none'
        ]
        security_summary = security_summary.reset_index()
        
        # Sector-level aggregations (if we had sector data)
        sector_summary = all_periods_data.groupby(['security_type', 'data_period']).agg({
            'market_value': ['sum', 'count'],
            'issuer_name': 'nunique',
            'manager_name': 'nunique'
        }).round(2)
        
        sector_summary.columns = [
            'total_market_value', 'total_holdings', 'unique_issuers', 'unique_managers'
        ]
        sector_summary = sector_summary.reset_index()
        
        self.processed_data['manager_summary'] = manager_summary
        self.processed_data['security_summary'] = security_summary
        self.processed_data['sector_summary'] = sector_summary
        
        logger.info("Created aggregated datasets")
    
    def export_processed_data(self, format: str = 'parquet'):
        """
        Export processed data in specified format.
        
        Args:
            format: Export format ('parquet', 'csv', 'excel')
        """
        logger.info(f"Exporting processed data in {format} format...")
        
        for name, df in self.processed_data.items():
            if format == 'parquet':
                output_path = self.output_dir / f"{name}.parquet"
                df.to_parquet(output_path, index=False)
            elif format == 'csv':
                output_path = self.output_dir / f"{name}.csv"
                df.to_csv(output_path, index=False)
            elif format == 'excel':
                output_path = self.output_dir / f"{name}.xlsx"
                df.to_excel(output_path, index=False)
            
            logger.info(f"Exported {name}: {len(df):,} records to {output_path}")
    
    def generate_data_summary(self):
        """Generate a comprehensive summary of the processed data."""
        logger.info("Generating data summary...")
        
        summary = {
            'processing_timestamp': datetime.now().isoformat(),
            'data_periods': list(self.raw_data.keys()),
            'total_records': sum(len(data.get('holdings', [])) for data in self.raw_data.values()),
            'datasets_created': list(self.processed_data.keys())
        }
        
        # Add dataset-specific summaries
        for name, df in self.processed_data.items():
            summary[f'{name}_records'] = len(df)
            summary[f'{name}_columns'] = list(df.columns)
        
        # Save summary
        summary_path = self.output_dir / 'processing_summary.json'
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Data summary saved to {summary_path}")
        return summary
    
    def process_cme_products(self):
        """
        Process CME products data - load cleaned data and filter top products per asset class.
        
        Returns:
            DataFrame with filtered CME products
        """
        logger.info("Processing CME products data...")
        
        try:
            # Load the cleaned CME products data
            cme_file = 'data/cme_products_cleaned.csv'
            if not os.path.exists(cme_file):
                logger.warning(f"CME products file not found: {cme_file}")
                return pd.DataFrame()
            
            cme_df = pd.read_csv(cme_file)
            logger.info(f"Loaded {len(cme_df)} CME products")
            
            # Filter top 10 products per asset class by Open Interest
            logger.info("Filtering top 10 products per asset class by Open Interest...")
            top_products = cme_df.groupby('Asset Class').apply(
                lambda x: x.nlargest(10, 'Open Interest')
            ).reset_index(drop=True)
            
            logger.info(f"Filtered to {len(top_products)} top products across {top_products['Asset Class'].nunique()} asset classes")
            
            # Store the processed data
            self.processed_data['cme_products_filtered'] = top_products
            
            # Export filtered CME products
            top_products.to_csv(f'{self.output_dir}/cme_products_filtered.csv', index=False)
            top_products.to_parquet(f'{self.output_dir}/cme_products_filtered.parquet', index=False)
            
            return top_products
            
        except Exception as e:
            logger.error(f"Error processing CME products: {e}")
            return pd.DataFrame()
    
    def enhance_holdings_with_cme_flags(self, holdings_df):
        """
        Enhance holdings data with CME product flags using fast combined issuer+security_type matching.
        
        Args:
            holdings_df: The holdings DataFrame to enhance
            
        Returns:
            Enhanced DataFrame with CME flags
        """
        logger.info("Enhancing holdings data with CME product flags using fast matching...")
        
        if 'cme_products_filtered' not in self.processed_data:
            logger.warning("No CME products data available for matching")
            return holdings_df
        
        cme_products = self.processed_data['cme_products_filtered']
        
        # Initialize CME flags
        holdings_df['is_cme_product'] = False
        holdings_df['cme_product_name'] = ''
        holdings_df['cme_asset_class'] = ''
        holdings_df['cme_product_group'] = ''
        holdings_df['cme_category'] = ''
        holdings_df['cme_sub_category'] = ''
        holdings_df['cme_open_interest'] = 0
        holdings_df['cme_volume'] = 0
        
        # Filter out stock-related securities first (CME doesn't trade stocks)
        logger.info("Filtering out stock securities for CME matching...")
        stock_security_types = ['COM', 'CMN', 'CL A', 'CL B', 'CL C', 'CL D', 'CL E', 'CL F', 'CL G', 'CL H', 'CL I', 'CL J', 'CL K', 'CL L', 'CL M', 'CL N', 'CL O', 'CL P', 'CL Q', 'CL R', 'CL S', 'CL T', 'CL U', 'CL V', 'CL W', 'CL X', 'CL Y', 'CL Z']
        stock_keywords = ['CORP', 'CORPORATION', 'INC', 'INCORPORATED', 'LTD', 'LIMITED', 'COMPANY', 'CO', 'GROUP', 'HOLDINGS', 'ENTERPRISES', 'INTERNATIONAL', 'GLOBAL', 'WORLDWIDE', 'SYSTEMS', 'TECHNOLOGIES', 'SOLUTIONS', 'SERVICES', 'MANAGEMENT', 'PARTNERS', 'VENTURES', 'CAPITAL', 'INVESTMENTS', 'FINANCIAL', 'BANK', 'BANCORP', 'TRUST', 'REIT', 'REAL ESTATE']

        non_stock_mask = ~(
            holdings_df['security_type'].isin(stock_security_types) |
            holdings_df['issuer_name'].str.contains('|'.join(stock_keywords), case=False, na=False)
        )
        
        df_non_stock = holdings_df[non_stock_mask].copy()
        logger.info(f"Filtered to {len(df_non_stock):,} non-stock holdings for CME matching")
        
        # Create combined issuer+security_type combinations for fast matching
        logger.info("Creating combined issuer+security_type combinations...")
        df_non_stock['combined_key'] = df_non_stock['issuer_name'] + ' | ' + df_non_stock['security_type']
        
        # Get unique combinations with aggregated data
        unique_combinations = df_non_stock.groupby('combined_key').agg({
            'issuer_name': 'first',
            'security_type': 'first',
            'market_value': 'sum',
            'manager_name': 'count'
        }).reset_index()
        
        unique_combinations.columns = ['combined_key', 'issuer_name', 'security_type', 'total_market_value', 'holding_count']
        logger.info(f"Created {len(unique_combinations):,} unique combinations")
        
        # Create CME product group patterns - use product group name for strict matching
        cme_patterns = {}
        for _, row in cme_products.iterrows():
            product_group = row['cme_product_group']
            
            # Create matching patterns from product group name (primary) and key parts of product name
            # The product group itself must appear in the 13F data for a match
            product_group_upper = str(product_group).upper().strip()
            
            # Extract key identifying parts from product group for matching
            # Remove common suffixes like "Futures", "Options", etc. for matching
            group_for_matching = product_group_upper
            common_suffixes = [' FUTURES', ' OPTIONS', ' FUTURE', ' OPTION']
            for suffix in common_suffixes:
                if group_for_matching.endswith(suffix):
                    group_for_matching = group_for_matching[:-len(suffix)].strip()
            
            if product_group not in cme_patterns:
                cme_patterns[product_group] = {
                    'group_name': product_group_upper,
                    'group_for_matching': group_for_matching,
                    'product_info': row,
                    'match_length': len(group_for_matching)  # For ranking matches
                }
            else:
                # Keep the product info with highest Open Interest
                if row['Open Interest'] > cme_patterns[product_group]['product_info']['Open Interest']:
                    cme_patterns[product_group]['product_info'] = row
        
        logger.info(f"Created {len(cme_patterns)} CME product group patterns for strict matching")
        
        # Fast matching using strict product group name matching
        matched_combinations = {}
        
        for cme_group, pattern_data in cme_patterns.items():
            group_for_matching = pattern_data['group_for_matching']
            product_info = pattern_data['product_info']
            
            matches = []
            for _, row in unique_combinations.iterrows():
                combined_text = str(row['combined_key']).upper()
                issuer_text = str(row['issuer_name']).upper()
                
                # Strict matching with word boundaries to avoid false positives
                # Check if the group name appears as a whole word/phrase (not as substring within another word)
                # Example: "GOLD" should NOT match "GOLDMAN", but should match "GOLD TRUST"
                
                # Escape special regex characters in the group name and match as whole word
                escaped_group = re.escape(group_for_matching)
                # Match as whole word (word boundary before and after)
                pattern = r'\b' + escaped_group + r'\b'
                
                if re.search(pattern, combined_text):
                    matches.append({
                        'combined_key': row['combined_key'],
                        'issuer_name': row['issuer_name'],
                        'security_type': row['security_type'],
                        'total_market_value': row['total_market_value'],
                        'holding_count': row['holding_count'],
                        'match_group': cme_group,
                        'match_length': pattern_data['match_length']  # For deduplication
                    })
            
            if matches:
                matched_combinations[cme_group] = {
                    'matches': matches,
                    'product_info': product_info
                }
                logger.info(f"Matched {len(matches)} combinations to {cme_group}")
        
        # Handle duplicate matches - one holding can only match to one CME group
        # Keep the match with the longest (most specific) product group name
        logger.info("Deduplicating matches to ensure one-to-one matching...")
        all_matches_flat = []
        for cme_group, match_data in matched_combinations.items():
            for match in match_data['matches']:
                all_matches_flat.append(match)
        
        # Group by combined_key and keep only the best match (longest match_length)
        matches_by_key = defaultdict(list)
        for match in all_matches_flat:
            matches_by_key[match['combined_key']].append(match)
        
        # Keep only the best match for each holding
        deduplicated_matches = {}
        for combined_key, match_list in matches_by_key.items():
            # Sort by match_length (descending) to get most specific match
            best_match = max(match_list, key=lambda x: x['match_length'])
            deduplicated_matches[combined_key] = best_match
        
        # Rebuild matched_combinations with deduplicated matches
        matched_combinations_dedup = {}
        for combined_key, match in deduplicated_matches.items():
            cme_group = match['match_group']
            if cme_group not in matched_combinations_dedup:
                matched_combinations_dedup[cme_group] = {
                    'matches': [],
                    'product_info': matched_combinations[cme_group]['product_info']
                }
            # Reconstruct match dict without the match_group and match_length fields
            matched_combinations_dedup[cme_group]['matches'].append({
                'combined_key': match['combined_key'],
                'issuer_name': match['issuer_name'],
                'security_type': match['security_type'],
                'total_market_value': match['total_market_value'],
                'holding_count': match['holding_count']
            })
        
        matched_combinations = matched_combinations_dedup
        total_matches = len(deduplicated_matches)
        logger.info(f"After deduplication: {total_matches:,} unique matches across {len(matched_combinations)} CME groups")
        
        # Apply CME flags to original holdings data using pandas merge (fastest approach)
        logger.info("Applying CME flags to holdings data using pandas merge...")
        
        # Create a DataFrame with all CME matches for merging
        cme_matches_df = []
        for cme_group, match_data in matched_combinations.items():
            product_info = match_data['product_info']
            matches = match_data['matches']
            
            for match in matches:
                cme_matches_df.append({
                    'issuer_name': match['issuer_name'],
                    'security_type': match['security_type'],
                    'is_cme_product': True,
                    'cme_product_name': product_info['Product Name'],
                    'cme_asset_class': product_info['Asset Class'],
                    'cme_product_group': product_info['cme_product_group'],
                    'cme_category': product_info['Category'],
                    'cme_sub_category': product_info['Sub-Category'],
                    'cme_open_interest': product_info['Open Interest'],
                    'cme_volume': product_info['Volume']
                })
        
        if cme_matches_df:
            cme_df = pd.DataFrame(cme_matches_df)
            
            # Merge with holdings data to apply CME flags
            holdings_df = holdings_df.merge(
                cme_df, 
                on=['issuer_name', 'security_type'], 
                how='left',
                suffixes=('', '_cme')
            )
            
            # Update CME flags where matches were found
            cme_mask = holdings_df['is_cme_product_cme'].notna()
            holdings_df.loc[cme_mask, 'is_cme_product'] = holdings_df.loc[cme_mask, 'is_cme_product_cme']
            holdings_df.loc[cme_mask, 'cme_product_name'] = holdings_df.loc[cme_mask, 'cme_product_name_cme']
            holdings_df.loc[cme_mask, 'cme_asset_class'] = holdings_df.loc[cme_mask, 'cme_asset_class_cme']
            holdings_df.loc[cme_mask, 'cme_product_group'] = holdings_df.loc[cme_mask, 'cme_product_group_cme']
            holdings_df.loc[cme_mask, 'cme_category'] = holdings_df.loc[cme_mask, 'cme_category_cme']
            holdings_df.loc[cme_mask, 'cme_sub_category'] = holdings_df.loc[cme_mask, 'cme_sub_category_cme']
            holdings_df.loc[cme_mask, 'cme_open_interest'] = holdings_df.loc[cme_mask, 'cme_open_interest_cme']
            holdings_df.loc[cme_mask, 'cme_volume'] = holdings_df.loc[cme_mask, 'cme_volume_cme']
            
            # Clean up merge columns
            cme_columns = [col for col in holdings_df.columns if col.endswith('_cme')]
            holdings_df.drop(cme_columns, axis=1, inplace=True)
        
        # Clean up temporary columns
        if 'combined_key' in holdings_df.columns:
            holdings_df.drop('combined_key', axis=1, inplace=True)
        
        total_cme_holdings = holdings_df['is_cme_product'].sum()
        logger.info(f"Total CME product matches: {total_cme_holdings:,} out of {len(holdings_df):,} holdings")
        logger.info(f"CME groups found: {len(matched_combinations)}")
        
        return holdings_df
    
    def run_full_preprocessing(self):
        """Run the complete preprocessing pipeline."""
        logger.info("Starting full preprocessing pipeline...")
        
        # Discover and load all data periods
        periods = self.discover_data_periods()
        logger.info(f"Discovered {len(periods)} data periods: {periods}")
        
        all_periods_data = []
        
        for period in periods:
            # Load data for this period
            self.load_period_data(period)
            
            # Merge and clean data for this period
            merged_data = self.merge_period_data(period)
            if merged_data is not None:
                all_periods_data.append(merged_data)
        
        # Combine all periods
        if all_periods_data:
            combined_data = pd.concat(all_periods_data, ignore_index=True)
            
            # Process CME products first
            self.process_cme_products()
            
            # Enhance holdings data with CME flags
            combined_data = self.enhance_holdings_with_cme_flags(combined_data)
            
            self.processed_data['combined_holdings'] = combined_data
            
            # Create aggregated datasets
            self.create_aggregated_datasets(combined_data)
            
            # Export data
            self.export_processed_data('parquet')
            self.export_processed_data('csv')
            
            # Generate summary
            summary = self.generate_data_summary()
            
            logger.info("Preprocessing completed successfully!")
            return summary
        else:
            logger.error("No data was successfully processed!")
            return None

def main():
    """Main execution function."""
    # Initialize preprocessor
    preprocessor = Form13FPreprocessor()
    
    # Run full preprocessing
    summary = preprocessor.run_full_preprocessing()
    
    if summary:
        print("\n" + "="*50)
        print("PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Processing completed at: {summary['processing_timestamp']}")
        print(f"Data periods processed: {summary['data_periods']}")
        print(f"Total records processed: {summary['total_records']:,}")
        print(f"Datasets created: {summary['datasets_created']}")
        print("\nOutput files saved to: processed_data/")
        print("="*50)
    else:
        print("Preprocessing failed. Check logs for details.")

if __name__ == "__main__":
    main()

