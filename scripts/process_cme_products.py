#!/usr/bin/env python3
"""
CME Products Processing Script

This script combines CME products cleaning and grouped summary generation.
It groups related CME products together by adding a cme_product_group column
using keyword matching, filters out inactive products, and creates a grouped
summary by Asset Class and product group.

Author: CME 13F Analysis Team
Date: 2025
"""

import pandas as pd
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cme_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CMEProductsProcessor:
    """
    A comprehensive processor for CME products data that handles cleaning,
    grouping, and summary generation.
    """
    
    def __init__(self, input_file: str = "data/all_cme_products.csv", 
                 cleaned_output: str = "data/cme_products_cleaned.csv",
                 summary_output: str = "data/cme_products_grouped_summary.csv"):
        """
        Initialize the processor.
        
        Args:
            input_file: Path to the input CME products CSV
            cleaned_output: Path to save the cleaned CSV
            summary_output: Path to save the grouped summary CSV
        """
        self.input_file = input_file
        self.cleaned_output = cleaned_output
        self.summary_output = summary_output
        
        # Define keyword mappings for product grouping
        self.product_groups = {
            # Interest Rate Products
            'SOFR': ['SOFR'],
            'T-Note': ['T-Note', 'Treasury Note'],
            'Treasury Bond': ['Treasury Bond', 'T-Bond'],
            'Federal Funds': ['Federal Funds', 'Fed Funds'],
            'Eurodollar': ['Eurodollar'],
            
            # Energy Products
            'Natural Gas': ['Natural Gas', 'Henry Hub'],
            'Crude Oil': ['Crude Oil', 'WTI', 'Brent'],
            'Gasoline': ['Gasoline', 'RBOB'],
            'Heating Oil': ['Heating Oil'],
            'ULSD': ['ULSD', 'NY Harbor ULSD'],
            'Propane': ['Propane'],
            'Ethanol': ['Ethanol'],
            
            # Equity Products
            'S&P 500': ['S&P 500', 'S&P', 'E-mini S&P', 'SPDR'],
            'Nasdaq': ['Nasdaq', 'NQ', 'E-mini Nasdaq'],
            'Dow Jones': ['Dow', 'DJIA', 'Dow Jones'],
            'Russell': ['Russell', 'Russell 2000'],
            
            # Agriculture Products
            'Corn': ['Corn'],
            'Soybean': ['Soybean'],
            'Wheat': ['Wheat'],
            'Cotton': ['Cotton'],
            'Live Cattle': ['Live Cattle'],
            'Lean Hogs': ['Lean Hogs', 'Lean Hog'],
            'Feeder Cattle': ['Feeder Cattle'],
            'Sugar': ['Sugar'],
            'Coffee': ['Coffee'],
            'Cocoa': ['Cocoa'],
            'Orange Juice': ['Orange Juice'],
            
            # Metals Products
            'Gold': ['Gold'],
            'Silver': ['Silver'],
            'Copper': ['Copper'],
            'Platinum': ['Platinum'],
            'Zinc': ['Zinc'],
            'Aluminum': ['Aluminum'],
            'Palladium': ['Palladium'],
            
            # Cryptocurrency Products
            'Bitcoin': ['Bitcoin', 'BTC'],
            'Ethereum': ['Ethereum', 'ETH'],
            'XRP': ['XRP'],
            'Solana': ['Solana', 'SOL'],
            'Litecoin': ['Litecoin', 'LTC'],
            'Dogecoin': ['Dogecoin', 'DOGE'],
            
            # Foreign Exchange Products
            'Euro FX': ['Euro FX', 'EUR'],
            'British Pound': ['British Pound', 'GBP'],
            'Japanese Yen': ['Japanese Yen', 'JPY'],
            'Canadian Dollar': ['Canadian Dollar', 'CAD'],
            'Australian Dollar': ['Australian Dollar', 'AUD'],
            'Swiss Franc': ['Swiss Franc', 'CHF'],
            'Mexican Peso': ['Mexican Peso', 'MXN'],
            'Brazilian Real': ['Brazilian Real', 'BRL'],
            'South African Rand': ['South African Rand', 'ZAR'],
            'Russian Ruble': ['Russian Ruble', 'RUB'],
            'Chinese Yuan': ['Chinese Yuan', 'CNY', 'CNH'],
            'New Zealand Dollar': ['New Zealand Dollar', 'NZD'],
            'Korean Won': ['Korean Won', 'KRW'],
            'Israeli Shekel': ['Israeli Shekel', 'ILS'],
            
            # Weather Products
            'HDD': ['HDD', 'Heating Degree Days'],
            'CDD': ['CDD', 'Cooling Degree Days'],
            
            # Additional Energy Products
            'PJM': ['PJM', 'PJM Western Hub'],
            'Carbon': ['Carbon', 'Carbon Allowance', 'CBL California Carbon'],
            'LNG': ['LNG', 'LNG Freight', 'LNG Japan/Korea Marker'],
            'Fuel Oil': ['Fuel Oil', 'Fuel Oil Barges'],
            
            # Additional Agriculture Products
            'UMBS': ['UMBS', 'UMBS TBA', '30-Year UMBS TBA'],
            'KC HRW': ['KC HRW', 'Kansas City HRW'],
            
            # Additional Products
            'Bloomberg Commodity': ['Bloomberg Commodity', 'Bloomberg Commodity Index'],
            'Butter': ['Butter', 'Spot Butter'],
        }
    
    def assign_product_group(self, product_name: str) -> str:
        """
        Assign a product group based on keyword matching.
        
        Args:
            product_name: The product name to analyze
            
        Returns:
            The assigned product group or original name if no match
        """
        if pd.isna(product_name) or product_name == '':
            return 'Unknown'
        
        product_name_upper = str(product_name).upper()
        
        # Check each group's keywords
        for group_name, keywords in self.product_groups.items():
            for keyword in keywords:
                if keyword.upper() in product_name_upper:
                    return group_name
        
        # If no match found, return original name
        return product_name
    
    def clean_products(self) -> pd.DataFrame:
        """
        Clean and group the CME products data.
        
        Returns:
            DataFrame with added cme_product_group column
        """
        logger.info(f"Loading CME products from {self.input_file}")
        
        # Load the data
        try:
            df = pd.read_csv(self.input_file)
            logger.info(f"Loaded {len(df)} products")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
        
        # Filter out products with zero Open Interest
        logger.info("Filtering out products with zero Open Interest...")
        initial_count = len(df)
        df = df[df['Open Interest'] > 0]
        filtered_count = len(df)
        logger.info(f"Removed {initial_count - filtered_count} products with zero Open Interest")
        logger.info(f"Remaining products: {filtered_count}")
        
        # Add cme_product_group column
        logger.info("Assigning product groups...")
        df['cme_product_group'] = df['Product Name'].apply(self.assign_product_group)
        
        # Log statistics
        self._log_grouping_stats(df)
        
        return df
    
    def create_grouped_summary(self, cleaned_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create grouped summary by Asset Class and cme_product_group.
        
        Args:
            cleaned_df: The cleaned DataFrame
            
        Returns:
            DataFrame with grouped summary data
        """
        logger.info("Creating grouped summary...")
        
        # Group by Asset Class and cme_product_group, sum Open Interest and Volume
        grouped_data = cleaned_df.groupby(['Asset Class', 'cme_product_group']).agg({
            'Open Interest': 'sum',
            'Volume': 'sum',
            'Product Name': 'count'
        }).reset_index()
        
        # Rename columns for clarity
        grouped_data.columns = ['Asset Class', 'CME Product Group', 'Total Open Interest', 'Total Volume', 'Product Count']
        
        # Sort by Total Open Interest descending
        grouped_data = grouped_data.sort_values('Total Open Interest', ascending=False)
        
        logger.info(f"Created {len(grouped_data)} grouped combinations")
        
        return grouped_data
    
    def _log_grouping_stats(self, df: pd.DataFrame):
        """Log grouping statistics."""
        total_products = len(df)
        unique_groups = df['cme_product_group'].nunique()
        
        logger.info(f"Total products processed: {total_products:,}")
        logger.info(f"Unique product groups created: {unique_groups:,}")
        
        # Top 10 groups by product count
        top_groups = df['cme_product_group'].value_counts().head(10)
        logger.info("Top 10 product groups by count:")
        for group, count in top_groups.items():
            logger.info(f"  {group}: {count:,} products")
        
        # Products that weren't grouped (kept original names)
        original_names = df[df['cme_product_group'] == df['Product Name']]
        logger.info(f"Products kept with original names: {len(original_names):,}")
        
        if len(original_names) > 0:
            logger.info("Sample ungrouped products:")
            for name in original_names['Product Name'].head(5):
                logger.info(f"  - {name}")
    
    def save_cleaned_data(self, df: pd.DataFrame):
        """
        Save the cleaned data to CSV.
        
        Args:
            df: The cleaned DataFrame
        """
        logger.info(f"Saving cleaned data to {self.cleaned_output}")
        
        try:
            # Ensure output directory exists
            Path(self.cleaned_output).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df.to_csv(self.cleaned_output, index=False)
            logger.info(f"Successfully saved {len(df)} products to {self.cleaned_output}")
            
        except Exception as e:
            logger.error(f"Error saving cleaned data: {e}")
    
    def save_grouped_summary(self, grouped_df: pd.DataFrame):
        """
        Save the grouped summary to CSV.
        
        Args:
            grouped_df: The grouped DataFrame
        """
        logger.info(f"Saving grouped summary to {self.summary_output}")
        
        try:
            # Ensure output directory exists
            Path(self.summary_output).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            grouped_df.to_csv(self.summary_output, index=False)
            logger.info(f"Successfully saved grouped summary to {self.summary_output}")
            
        except Exception as e:
            logger.error(f"Error saving grouped summary: {e}")
    
    def print_summary_report(self, cleaned_df: pd.DataFrame, grouped_df: pd.DataFrame):
        """
        Print a comprehensive summary report.
        
        Args:
            cleaned_df: The cleaned DataFrame
            grouped_df: The grouped DataFrame
        """
        print("\n" + "="*80)
        print("CME PRODUCTS PROCESSING SUMMARY REPORT")
        print("="*80)
        print(f"Input file: {self.input_file}")
        print(f"Cleaned output: {self.cleaned_output}")
        print(f"Summary output: {self.summary_output}")
        print()
        
        # Cleaning summary
        print("CLEANING RESULTS:")
        print("-" * 50)
        print(f"Total products processed: {len(cleaned_df):,}")
        print(f"Unique product groups: {cleaned_df['cme_product_group'].nunique():,}")
        print(f"Top 5 groups by count:")
        for group, count in cleaned_df['cme_product_group'].value_counts().head(5).items():
            print(f"  {group}: {count:,} products")
        print()
        
        # Grouped summary
        print("GROUPED SUMMARY RESULTS:")
        print("-" * 50)
        print(f"Total combinations: {len(grouped_df):,}")
        print()
        
        # Top 20 by Open Interest
        print("Top 20 combinations by Open Interest:")
        print("-" * 80)
        print(f"{'Open Interest':>15} | {'Asset Class':<15} | {'Product Group':<20} | {'Count':>3}")
        print("-" * 80)
        for _, row in grouped_df.head(20).iterrows():
            print(f"{row['Total Open Interest']:>15,} | {row['Asset Class']:<15} | {row['CME Product Group']:<20} | {row['Product Count']:>3}")
        
        print()
        
        # Summary by Asset Class
        print("Summary by Asset Class:")
        print("-" * 50)
        asset_summary = grouped_df.groupby('Asset Class').agg({
            'Total Open Interest': 'sum',
            'Total Volume': 'sum',
            'Product Count': 'sum',
            'CME Product Group': 'count'
        }).sort_values('Total Open Interest', ascending=False)
        
        asset_summary.columns = ['Total Open Interest', 'Total Volume', 'Total Products', 'Product Groups']
        
        for asset_class, row in asset_summary.iterrows():
            print(f"{asset_class:<20} | {row['Total Open Interest']:>15,} | {row['Product Groups']:>3} groups | {row['Total Products']:>4} products")
        
        print("="*80)
    
    def run_processing(self):
        """Run the complete processing pipeline."""
        logger.info("Starting CME products processing pipeline...")
        
        # Step 1: Clean the products
        cleaned_df = self.clean_products()
        
        if cleaned_df.empty:
            logger.error("Cleaning failed - no data processed")
            return False
        
        # Step 2: Create grouped summary
        grouped_df = self.create_grouped_summary(cleaned_df)
        
        if grouped_df.empty:
            logger.error("Grouped summary creation failed")
            return False
        
        # Step 3: Save outputs
        self.save_cleaned_data(cleaned_df)
        self.save_grouped_summary(grouped_df)
        
        # Step 4: Print comprehensive report
        self.print_summary_report(cleaned_df, grouped_df)
        
        return True

def main():
    """Main function to run the processing script."""
    processor = CMEProductsProcessor()
    success = processor.run_processing()
    
    if success:
        print("\n✅ CME products processing completed successfully!")
        print(f"📁 Check the cleaned data: {processor.cleaned_output}")
        print(f"📁 Check the grouped summary: {processor.summary_output}")
    else:
        print("\n❌ CME products processing failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
