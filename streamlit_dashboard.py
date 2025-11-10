#!/usr/bin/env python3
"""
 Dashboard for 13F Portfolio Monitoring
Real-time tracking of investment manager portfolio changes between quarters
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import warnings
import os
import sys
warnings.filterwarnings('ignore')

class PortfolioStreamlitDashboard:
    def __init__(self):
        self.holdings_data = None
        self.managers_data = None
        self.securities_data = None
        self.cme_products_data = None
        # Don't load data immediately - use lazy loading
        
    @st.cache_data(ttl=3600, show_spinner="Loading holdings data...")  # Cache for 1 hour
    def load_holdings_data(_self):
        """Load the processed 13F holdings data with caching"""
        file_path = 'processed_data/combined_holdings.parquet'
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return pd.DataFrame()
            
            # Check file size (warn if very large)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            if file_size > 500:
                st.warning(f"⚠️ Large file detected ({file_size:.1f} MB). Loading may take a while...")
            
            # Load data with error handling
            try:
                data = pd.read_parquet(file_path, engine='pyarrow')
            except Exception as parquet_error:
                # Try alternative engine
                try:
                    data = pd.read_parquet(file_path, engine='fastparquet')
                except Exception:
                    st.error(f"❌ Failed to read parquet file: {parquet_error}")
                    return pd.DataFrame()
            
            # Check if dataframe is empty
            if data.empty:
                return data
            
            # Add quarter information - handle the custom date format
            def parse_data_period(period_str):
                """Parse the custom data period format like '01JUN2025-31AUG2025_form13f'"""
                try:
                    if pd.isna(period_str):
                        return pd.NaT
                    # Extract the start date part
                    start_part = str(period_str).split('-')[0]  # '01JUN2025'
                    # Parse the date
                    return pd.to_datetime(start_part, format='%d%b%Y', errors='coerce')
                except:
                    return pd.NaT
            
            # Only process if data_period column exists
            if 'data_period' in data.columns:
                data['quarter'] = data['data_period'].apply(parse_data_period).dt.to_period('Q')
            
            return data
        except MemoryError:
            st.error("❌ Out of memory while loading holdings data. The file may be too large for this environment.")
            return pd.DataFrame()
        except Exception as e:
            # Don't show error in cached function to avoid spam
            # Error will be handled in the calling function
            import traceback
            error_msg = f"Error loading holdings: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # Print to console for debugging
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600, show_spinner="Loading managers data...")  # Cache for 1 hour
    def load_managers_data(_self):
        """Load the processed managers data with caching"""
        file_path = 'processed_data/manager_summary.parquet'
        try:
            if not os.path.exists(file_path):
                return pd.DataFrame()
            try:
                return pd.read_parquet(file_path, engine='pyarrow')
            except:
                return pd.read_parquet(file_path, engine='fastparquet')
        except Exception as e:
            print(f"Error loading managers data: {e}")  # Log to console
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600, show_spinner="Loading securities data...")  # Cache for 1 hour
    def load_securities_data(_self):
        """Load the processed securities data with caching"""
        file_path = 'processed_data/security_summary.parquet'
        try:
            if not os.path.exists(file_path):
                return pd.DataFrame()
            try:
                return pd.read_parquet(file_path, engine='pyarrow')
            except:
                return pd.read_parquet(file_path, engine='fastparquet')
        except Exception as e:
            print(f"Error loading securities data: {e}")  # Log to console
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600, show_spinner="Loading CME products data...")  # Cache for 1 hour
    def load_cme_products_data(_self):
        """Load the filtered CME products data with caching"""
        file_path = 'processed_data/cme_products_filtered.parquet'
        try:
            if not os.path.exists(file_path):
                return pd.DataFrame()
            try:
                return pd.read_parquet(file_path, engine='pyarrow')
            except:
                return pd.read_parquet(file_path, engine='fastparquet')
        except Exception as e:
            print(f"Error loading CME products data: {e}")  # Log to console
            return pd.DataFrame()
    
    def get_holdings_data(self):
        """Get holdings data with lazy loading"""
        if self.holdings_data is None:
            try:
                self.holdings_data = self.load_holdings_data()
                if self.holdings_data.empty:
                    st.error("❌ **Failed to load holdings data**")
                    st.info("💡 **Possible causes:**\n"
                           "- Data file is missing or not properly committed\n"
                           "- Git LFS files may not have been downloaded\n"
                           "- File may be corrupted or too large for this environment")
            except Exception as e:
                st.error(f"❌ **Error loading holdings data:** {e}")
                import traceback
                with st.expander("🔍 Show detailed error"):
                    st.code(traceback.format_exc())
                self.holdings_data = pd.DataFrame()
        return self.holdings_data
    
    def get_managers_data(self):
        """Get managers data with lazy loading"""
        if self.managers_data is None:
            try:
                self.managers_data = self.load_managers_data()
            except Exception as e:
                st.error(f"❌ Error loading managers data: {e}")
                self.managers_data = pd.DataFrame()
        return self.managers_data
    
    def get_securities_data(self):
        """Get securities data with lazy loading"""
        if self.securities_data is None:
            try:
                self.securities_data = self.load_securities_data()
            except Exception as e:
                st.error(f"❌ Error loading securities data: {e}")
                self.securities_data = pd.DataFrame()
        return self.securities_data
    
    def get_cme_products_data(self):
        """Get CME products data with lazy loading"""
        if self.cme_products_data is None:
            try:
                self.cme_products_data = self.load_cme_products_data()
            except Exception as e:
                st.error(f"❌ Error loading CME products data: {e}")
                self.cme_products_data = pd.DataFrame()
        return self.cme_products_data
    
    def _get_descriptive_security_class(self, security_class):
        """Convert security class codes to more descriptive names"""
        if pd.isna(security_class) or security_class == '':
            return 'Unknown'
        
        security_class = str(security_class).upper().strip()
        
        # Common mappings
        mappings = {
            'COM': 'Common Stock',
            'COMMON STOCK': 'Common Stock', 
            'COMMON': 'Common Stock',
            'CMN': 'Common Stock',
            'CL A': 'Class A Stock',
            'CL B': 'Class B Stock',
            'CL C': 'Class C Stock',
            'COM CL A': 'Common Class A',
            'COM NEW': 'Common Stock (New)',
            'STOCK': 'Stock',
            'SHS': 'Shares',
            'ORD SHS': 'Ordinary Shares',
            'ETF': 'Exchange Traded Fund',
            'PFD': 'Preferred Stock',
            'PREFERRED': 'Preferred Stock',
            'PREFERRED STOCK': 'Preferred Stock',
            'DEBT': 'Debt Security',
            'BOND': 'Bond',
            'NOTE': 'Note',
            'OPTION': 'Option',
            'WARRANT': 'Warrant',
            'UNIT': 'Unit',
            'TRUST': 'Trust',
            'FUND': 'Fund',
            'MUTUAL FUND': 'Mutual Fund',
            'REIT': 'Real Estate Investment Trust',
            'ADR': 'American Depositary Receipt',
            'GDR': 'Global Depositary Receipt'
        }
        
        # Check for exact matches first
        if security_class in mappings:
            return mappings[security_class]
        
        # Check for partial matches
        for key, value in mappings.items():
            if key in security_class:
                return value
        
        # If it contains common stock indicators
        if any(indicator in security_class for indicator in ['COM', 'COMMON', 'STOCK']):
            return 'Common Stock'
        
        # If it contains preferred indicators
        if any(indicator in security_class for indicator in ['PFD', 'PREFERRED']):
            return 'Preferred Stock'
        
        # If it contains debt indicators
        if any(indicator in security_class for indicator in ['DEBT', 'BOND', 'NOTE']):
            return 'Debt Security'
        
        # If it contains option indicators
        if any(indicator in security_class for indicator in ['OPTION', 'CALL', 'PUT']):
            return 'Option'
        
        # If it contains fund indicators
        if any(indicator in security_class for indicator in ['ETF', 'FUND', 'TRUST']):
            return 'Fund/Trust'
        
        # Default fallback
        return security_class
    
    def get_available_quarters(self):
        """Get list of available quarters in the data"""
        holdings_data = self.get_holdings_data()
        if not holdings_data.empty:
            quarters = sorted(holdings_data['quarter'].unique(), reverse=True)
            return [str(q) for q in quarters]
        return []
    
    def get_managers_list(self):
        """Get list of all managers"""
        managers_data = self.get_managers_data()
        if not managers_data.empty:
            return sorted(managers_data['manager_name'].unique())
        return []
    
    def calculate_portfolio_changes(self, manager_name, current_quarter, previous_quarter):
        """Calculate portfolio changes between quarters for a specific manager"""
        try:
            holdings_data = self.get_holdings_data()
            # Get current quarter holdings
            current_holdings = holdings_data[
                (holdings_data['manager_name'] == manager_name) & 
                (holdings_data['quarter'] == current_quarter)
            ].copy()
            
            
            # Get previous quarter holdings
            previous_holdings = holdings_data[
                (holdings_data['manager_name'] == manager_name) & 
                (holdings_data['quarter'] == previous_quarter)
            ].copy()
            
            if current_holdings.empty:
                return None, "No data for current quarter"
            
            # Calculate total portfolio value for each quarter
            current_total = current_holdings['market_value'].sum()
            previous_total = previous_holdings['market_value'].sum() if not previous_holdings.empty else 0
            
            # Create comparison dataframe
            comparison_data = []
            
            # Process current holdings
            for _, holding in current_holdings.iterrows():
                cusip = holding['cusip']
                issuer = holding['issuer_name']
                current_value = holding['market_value']
                current_pct = (current_value / current_total) * 100
                
                # Find corresponding previous holding
                prev_holding = previous_holdings[previous_holdings['cusip'] == cusip]
                
                if not prev_holding.empty:
                    prev_value = prev_holding['market_value'].iloc[0]
                    prev_pct = (prev_value / previous_total) * 100 if previous_total > 0 else 0
                    change_type = "Existing"
                else:
                    prev_value = 0
                    prev_pct = 0
                    change_type = "New Position"
                
                # Calculate changes
                value_change = current_value - prev_value
                pct_change = current_pct - prev_pct
                pct_change_pct = (value_change / prev_value * 100) if prev_value > 0 else float('inf')
                
                comparison_data.append({
                    'cusip': cusip,
                    'issuer_name': issuer,
                    'security_class': self._get_descriptive_security_class(holding.get('security_class', '')),
                    'security_type': holding['security_type'],
                    'current_value': current_value,
                    'previous_value': prev_value,
                    'value_change': value_change,
                    'current_pct': current_pct,
                    'previous_pct': prev_pct,
                    'pct_change': pct_change,
                    'pct_change_pct': pct_change_pct,
                    'change_type': change_type
                })
            
            # Add positions that were sold (existed in previous but not current)
            for _, holding in previous_holdings.iterrows():
                cusip = holding['cusip']
                if cusip not in current_holdings['cusip'].values:
                    prev_value = holding['market_value']
                    prev_pct = (prev_value / previous_total) * 100 if previous_total > 0 else 0
                    
                comparison_data.append({
                    'cusip': cusip,
                    'issuer_name': holding['issuer_name'],
                    'security_class': self._get_descriptive_security_class(holding.get('security_class', '')),
                    'security_type': holding['security_type'],
                        'current_value': 0,
                        'previous_value': prev_value,
                        'value_change': -prev_value,
                        'current_pct': 0,
                        'previous_pct': prev_pct,
                        'pct_change': -prev_pct,
                        'pct_change_pct': -100,
                        'change_type': "Sold Position"
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('current_value', ascending=False)
            
            return comparison_df, None
            
        except Exception as e:
            return None, f"Error calculating changes: {e}"
    
    def create_portfolio_summary(self, comparison_df, manager_name, current_quarter, previous_quarter):
        """Create portfolio summary metrics"""
        if comparison_df is None or comparison_df.empty:
            return {}
        
        total_current = comparison_df['current_value'].sum()
        total_previous = comparison_df['previous_value'].sum()
        total_change = total_current - total_previous
        
        new_positions = comparison_df[comparison_df['change_type'] == 'New Position']
        sold_positions = comparison_df[comparison_df['change_type'] == 'Sold Position']
        existing_positions = comparison_df[comparison_df['change_type'] == 'Existing']
        
        summary = {
            'manager_name': manager_name,
            'current_quarter': str(current_quarter),
            'previous_quarter': str(previous_quarter),
            'total_current_value': total_current,
            'total_previous_value': total_previous,
            'total_change': total_change,
            'total_change_pct': (total_change / total_previous * 100) if total_previous > 0 else 0,
            'num_positions_current': len(comparison_df[comparison_df['current_value'] > 0]),
            'num_positions_previous': len(comparison_df[comparison_df['previous_value'] > 0]),
            'num_new_positions': len(new_positions),
            'num_sold_positions': len(sold_positions),
            'new_positions_value': new_positions['current_value'].sum(),
            'sold_positions_value': sold_positions['previous_value'].sum(),
            'top_new_position': new_positions['issuer_name'].iloc[0] if not new_positions.empty else None,
            'largest_increase': existing_positions.loc[existing_positions['value_change'].idxmax(), 'issuer_name'] if not existing_positions.empty else None,
            'largest_decrease': existing_positions.loc[existing_positions['value_change'].idxmin(), 'issuer_name'] if not existing_positions.empty else None
        }
        
        return summary

    def render_cme_products_tab(self, selected_manager, current_quarter, previous_quarter):
        """Render the CME Products tab with individual manager portfolio analysis."""
        
        # Get CME-enhanced holdings data
        holdings_data = self.get_holdings_data()
        cme_products_data = self.get_cme_products_data()
        
        if holdings_data.empty or cme_products_data.empty:
            st.warning("CME products data not available. Please run preprocessing first.")
            return
        
        # Filter for CME-related holdings AND the selected current quarter
        cme_holdings = holdings_data[
            (holdings_data['is_cme_product'] == True) & 
            (holdings_data['quarter'] == current_quarter)
        ].copy()
        
        if cme_holdings.empty:
            st.info(f"No CME-related holdings found for quarter {current_quarter}.")
            return
        
        if not selected_manager:
            st.info("Please select a manager from the sidebar to analyze their CME holdings.")
            return
        
        # Check if selected manager has CME holdings in the current quarter
        manager_cme_holdings = cme_holdings[cme_holdings['manager_name'] == selected_manager].copy()
        
        if manager_cme_holdings.empty:
            st.info(f"No CME holdings found for {selected_manager} in quarter {current_quarter}. This manager may not have any CME-related positions in this quarter.")
            return
        
        # Calculate quarter-over-quarter changes for CME holdings
        cme_changes = self._calculate_cme_quarter_changes(manager_cme_holdings, current_quarter, previous_quarter)
        
        # Render analysis for the selected manager with changes
        self._render_individual_manager_cme_analysis(manager_cme_holdings, selected_manager, cme_changes)

    def _calculate_cme_quarter_changes(self, manager_cme_holdings, current_quarter, previous_quarter):
        """Calculate quarter-over-quarter changes for CME holdings."""
        holdings_data = self.get_holdings_data()
        
        # Get current quarter CME holdings
        current_cme = holdings_data[
            (holdings_data['manager_name'] == manager_cme_holdings.iloc[0]['manager_name']) & 
            (holdings_data['quarter'] == current_quarter) &
            (holdings_data['is_cme_product'] == True)
        ].copy()
        
        # Get previous quarter CME holdings
        previous_cme = holdings_data[
            (holdings_data['manager_name'] == manager_cme_holdings.iloc[0]['manager_name']) & 
            (holdings_data['quarter'] == previous_quarter) &
            (holdings_data['is_cme_product'] == True)
        ].copy()
        
        # Calculate changes by product group
        changes_data = []
        
        # Get all unique product groups from both quarters
        all_groups = set(current_cme['cme_product_group'].unique()) | set(previous_cme['cme_product_group'].unique())
        
        for group in all_groups:
            current_group = current_cme[current_cme['cme_product_group'] == group]
            previous_group = previous_cme[previous_cme['cme_product_group'] == group]
            
            current_value = current_group['market_value'].sum() if not current_group.empty else 0
            previous_value = previous_group['market_value'].sum() if not previous_group.empty else 0
            value_change = current_value - previous_value
            
            current_count = len(current_group)
            previous_count = len(previous_group)
            count_change = current_count - previous_count
            
            changes_data.append({
                'cme_product_group': group,
                'current_value': current_value,
                'previous_value': previous_value,
                'value_change': value_change,
                'value_change_pct': (value_change / previous_value * 100) if previous_value > 0 else float('inf'),
                'current_count': current_count,
                'previous_count': previous_count,
                'count_change': count_change,
                'change_type': 'New' if previous_value == 0 else ('Sold' if current_value == 0 else 'Existing')
            })
        
        return pd.DataFrame(changes_data).sort_values('current_value', ascending=False)

    def _render_individual_manager_cme_analysis(self, manager_cme_holdings, manager_name, cme_changes):
        """Render detailed CME analysis for a specific manager."""
        
        # Summary metrics with changes
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_cme_value = manager_cme_holdings['market_value'].sum()
            total_previous_value = cme_changes['previous_value'].sum()
            total_change = total_cme_value - total_previous_value
            st.metric(
                "Total CME Value",
                f"${total_cme_value:,.0f}",
                f"${total_change:+,.0f} ({total_change/total_previous_value*100:+.1f}%)" if total_previous_value > 0 else f"${total_change:+,.0f}",
                help="Total market value in CME products"
            )
        
        with col2:
            num_cme_positions = len(manager_cme_holdings)
            previous_positions = cme_changes['previous_count'].sum()
            position_change = num_cme_positions - previous_positions
            st.metric(
                "CME Positions",
                f"{num_cme_positions:,}",
                f"{position_change:+d}",
                help="Number of CME product holdings"
            )
        
        with col3:
            unique_groups = manager_cme_holdings['cme_product_group'].nunique()
            previous_groups = len(cme_changes[cme_changes['previous_value'] > 0])
            group_change = unique_groups - previous_groups
            st.metric(
                "Product Groups",
                f"{unique_groups:,}",
                f"{group_change:+d}",
                help="Number of different CME product groups"
            )
        
        with col4:
            unique_asset_classes = manager_cme_holdings['cme_asset_class'].nunique()
            st.metric(
                "Asset Classes",
                f"{unique_asset_classes:,}",
                help="Number of different CME asset classes"
            )
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Holdings by Product Group", 
            "🏷️ Holdings by Asset Class", 
            "📈 Detailed Holdings",
            "🔄 Quarter Changes"
        ])
        
        with tab1:
            self._render_manager_product_groups(manager_cme_holdings, cme_changes)
        
        with tab2:
            self._render_manager_asset_classes(manager_cme_holdings)
        
        with tab3:
            self._render_manager_detailed_holdings(manager_cme_holdings)
        
        with tab4:
            self._render_cme_quarter_changes(cme_changes)

    def _render_manager_product_groups(self, manager_cme_holdings, cme_changes):
        """Render manager's CME holdings grouped by product group with changes."""
        
        # Merge current holdings with changes data
        product_group_summary = manager_cme_holdings.groupby('cme_product_group').agg({
            'market_value': ['sum', 'count'],
            'cme_open_interest': 'first',
            'cme_volume': 'first',
            'cme_asset_class': 'first'
        }).reset_index()
        
        # Flatten column names
        product_group_summary.columns = [
            'Product Group', 'Total Value', 'Holdings Count', 
            'CME Open Interest', 'CME Volume', 'Asset Class'
        ]
        
        # Merge with changes data
        product_group_summary = product_group_summary.merge(
            cme_changes[['cme_product_group', 'value_change', 'value_change_pct', 'change_type']], 
            left_on='Product Group', 
            right_on='cme_product_group', 
            how='left'
        ).drop('cme_product_group', axis=1)
        
        # Sort by total value
        product_group_summary = product_group_summary.sort_values('Total Value', ascending=False)
        
        # Display table with changes
        st.dataframe(
            product_group_summary,
            use_container_width=True,
            column_config={
                "Total Value": st.column_config.NumberColumn(
                    "Total Value ($)",
                    format="$%.0f"
                ),
                "Holdings Count": st.column_config.NumberColumn(
                    "Holdings Count",
                    format="%d"
                ),
                "value_change": st.column_config.NumberColumn(
                    "Value Change ($)",
                    format="$%.0f"
                ),
                "value_change_pct": st.column_config.NumberColumn(
                    "Change %",
                    format="%.1f%%"
                ),
                "CME Open Interest": st.column_config.NumberColumn(
                    "CME Open Interest",
                    format="%d"
                ),
                "CME Volume": st.column_config.NumberColumn(
                    "CME Volume",
                    format="%d"
                )
            }
        )
        
        # Create visualization
        if len(product_group_summary) > 0:
            import plotly.express as px
            
            fig = px.bar(
                product_group_summary.head(10),
                x='Total Value',
                y='Product Group',
                orientation='h',
                title="Top 10 CME Product Groups by Value",
                labels={'Total Value': 'Market Value ($)', 'Product Group': 'CME Product Group'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    def _render_manager_asset_classes(self, manager_cme_holdings):
        """Render manager's CME holdings grouped by asset class."""
        
        # Group by asset class
        asset_class_summary = manager_cme_holdings.groupby('cme_asset_class').agg({
            'market_value': ['sum', 'count'],
            'cme_product_group': 'nunique'
        }).reset_index()
        
        # Flatten column names
        asset_class_summary.columns = [
            'Asset Class', 'Total Value', 'Holdings Count', 'Product Groups'
        ]
        
        # Sort by total value
        asset_class_summary = asset_class_summary.sort_values('Total Value', ascending=False)
        
        # Display table
        st.dataframe(
            asset_class_summary,
            use_container_width=True,
            column_config={
                "Total Value": st.column_config.NumberColumn(
                    "Total Value ($)",
                    format="$%.0f"
                ),
                "Holdings Count": st.column_config.NumberColumn(
                    "Holdings Count",
                    format="%d"
                ),
                "Product Groups": st.column_config.NumberColumn(
                    "Product Groups",
                    format="%d"
                )
            }
        )
        
        # Create pie chart
        if len(asset_class_summary) > 0:
            import plotly.express as px
            
            fig = px.pie(
                asset_class_summary,
                values='Total Value',
                names='Asset Class',
                title="CME Holdings Distribution by Asset Class"
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_manager_detailed_holdings(self, manager_cme_holdings):
        """Render detailed CME holdings table for the manager, grouped by cme_product_group."""
        
        # Group by cme_product_group (remove cme_product_name from grouping and display)
        detailed_holdings = manager_cme_holdings.groupby([
            'cme_product_group', 'cme_asset_class'
        ]).agg({
            'market_value': 'sum',
            'issuer_name': lambda x: ', '.join(x.unique()[:5]) + ('...' if len(x.unique()) > 5 else ''),  # Show sample issuers
            'cusip': 'nunique',  # Count unique CUSIPs
            'security_class': lambda x: len(x.unique()),  # Count unique security classes
            'cme_open_interest': 'first',
            'cme_volume': 'first',
            'quarter': 'first'
        }).reset_index()
        
        # Rename columns for display
        detailed_holdings.columns = [
            'CME Product Group', 'Asset Class', 'Total Market Value ($)',
            'Sample Issuers', 'Unique CUSIPs', 'Unique Securities',
            'CME Open Interest', 'CME Volume', 'Quarter'
        ]
        
        # Sort by market value
        detailed_holdings = detailed_holdings.sort_values('Total Market Value ($)', ascending=False)
        
        # Add product group selector for drill-down
        product_groups = ['All'] + sorted(detailed_holdings['CME Product Group'].unique().tolist())
        
        # Initialize session state for selected product group
        if 'selected_cme_product_group' not in st.session_state:
            st.session_state.selected_cme_product_group = 'All'
        
        # Product group selector
        selected_group = st.selectbox(
            "🔍 Select CME Product Group to view detailed holdings:",
            product_groups,
            index=product_groups.index(st.session_state.selected_cme_product_group) if st.session_state.selected_cme_product_group in product_groups else 0,
            key='cme_product_group_selector'
        )
        
        # Update session state
        st.session_state.selected_cme_product_group = selected_group
        
        # Display grouped summary table
        st.dataframe(
            detailed_holdings,
            use_container_width=True,
            column_config={
                "CME Product Group": st.column_config.TextColumn(
                    "CME Product Group",
                    help="Select a product group from the dropdown above to see detailed holdings"
                ),
                "Total Market Value ($)": st.column_config.NumberColumn(
                    "Total Market Value ($)",
                    format="$%.0f"
                ),
                "Unique CUSIPs": st.column_config.NumberColumn(
                    "Unique CUSIPs",
                    format="%d"
                ),
                "Unique Securities": st.column_config.NumberColumn(
                    "Unique Securities",
                    format="%d"
                ),
                "CME Open Interest": st.column_config.NumberColumn(
                    "CME Open Interest",
                    format="%d"
                ),
                "CME Volume": st.column_config.NumberColumn(
                    "CME Volume",
                    format="%d"
                )
            }
        )
        
        # Show detailed holdings if a specific product group is selected
        if selected_group != 'All':
            st.markdown("---")
            st.subheader(f"📋 Detailed Holdings: {selected_group}")
            
            # Filter original holdings by selected product group
            filtered_holdings = manager_cme_holdings[
                manager_cme_holdings['cme_product_group'] == selected_group
            ].copy()
            
            # Prepare detailed table
            detailed_table = filtered_holdings[[
                'issuer_name', 'cusip', 'security_class', 'market_value',
                'cme_asset_class', 'cme_open_interest', 'cme_volume'
            ]].copy()
            
            # Sort by market value
            detailed_table = detailed_table.sort_values('market_value', ascending=False)
            
            # Display detailed table
            st.dataframe(
                detailed_table,
                use_container_width=True,
                column_config={
                    "market_value": st.column_config.NumberColumn(
                        "Market Value ($)",
                        format="$%.0f"
                    ),
                    "cme_open_interest": st.column_config.NumberColumn(
                        "CME Open Interest",
                        format="%d"
                    ),
                    "cme_volume": st.column_config.NumberColumn(
                        "CME Volume",
                        format="%d"
                    )
                }
            )
            
            # Show summary stats for selected product group
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Holdings", len(filtered_holdings))
            with col2:
                st.metric("Total Market Value", f"${filtered_holdings['market_value'].sum():,.0f}")
            with col3:
                st.metric("Unique Issuers", filtered_holdings['issuer_name'].nunique())
        
        # Show summary
        st.info(f"Showing {len(detailed_holdings)} CME product groups. Holdings are aggregated by product group.")

    def _render_cme_quarter_changes(self, cme_changes):
        """Render quarter-over-quarter changes for CME holdings."""
        
        # Filter for significant changes
        significant_changes = cme_changes[abs(cme_changes['value_change']) > 1000000]  # > $1M changes
        
        if significant_changes.empty:
            st.info("No significant quarter-over-quarter changes (>$1M) found in CME holdings.")
            return
        
        # Display changes table
        st.dataframe(
            significant_changes,
            use_container_width=True,
            column_config={
                "current_value": st.column_config.NumberColumn(
                    "Current Value ($)",
                    format="$%.0f"
                ),
                "previous_value": st.column_config.NumberColumn(
                    "Previous Value ($)",
                    format="$%.0f"
                ),
                "value_change": st.column_config.NumberColumn(
                    "Value Change ($)",
                    format="$%.0f"
                ),
                "value_change_pct": st.column_config.NumberColumn(
                    "Change %",
                    format="%.1f%%"
                ),
                "current_count": st.column_config.NumberColumn(
                    "Current Count",
                    format="%d"
                ),
                "previous_count": st.column_config.NumberColumn(
                    "Previous Count",
                    format="%d"
                ),
                "count_change": st.column_config.NumberColumn(
                    "Count Change",
                    format="%d"
                )
            }
        )
        
        # Create visualization for changes
        if len(significant_changes) > 0:
            import plotly.express as px
            
            # Top increases and decreases
            increases = significant_changes[significant_changes['value_change'] > 0].head(10)
            decreases = significant_changes[significant_changes['value_change'] < 0].head(10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if not increases.empty:
                    fig = px.bar(
                        increases,
                        x='value_change',
                        y='cme_product_group',
                        orientation='h',
                        title="Top 10 CME Value Increases",
                        labels={'value_change': 'Value Change ($)', 'cme_product_group': 'Product Group'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if not decreases.empty:
                    fig = px.bar(
                        decreases,
                        x='value_change',
                        y='cme_product_group',
                        orientation='h',
                        title="Top 10 CME Value Decreases",
                        labels={'value_change': 'Value Change ($)', 'cme_product_group': 'Product Group'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(
        page_title="13F Portfolio Streamlit Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 13F Portfolio Streamlit Dashboard")
    st.markdown("**Real-time monitoring of investment manager portfolio changes between quarters**")
    
    # Check if processed_data directory exists
    if not os.path.exists('processed_data'):
        st.error("❌ **Data directory not found**")
        st.markdown("""
        The `processed_data` directory is missing. Please ensure:
        1. Data files are committed to the repository
        2. If using Git LFS, files are properly tracked
        3. Files are present in the repository root
        
        **Expected files:**
        - `processed_data/combined_holdings.parquet`
        - `processed_data/manager_summary.parquet`
        - `processed_data/security_summary.parquet`
        - `processed_data/cme_products_filtered.parquet` (optional)
        """)
        st.stop()
    
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = PortfolioStreamlitDashboard()
    
    dashboard = st.session_state.dashboard
    
    # Check if essential data files exist (but don't load them yet)
    essential_files = [
        'processed_data/manager_summary.parquet',  # Load this first (smallest)
        'processed_data/security_summary.parquet',  # Then this
        'processed_data/combined_holdings.parquet'  # Finally this (largest)
    ]
    
    # Check file existence and sizes
    file_info = {}
    for file in essential_files:
        if os.path.exists(file):
            try:
                size_mb = os.path.getsize(file) / (1024 * 1024)
                file_info[file] = {
                    "exists": True,
                    "size_mb": round(size_mb, 2)
                }
                # Check if file is suspiciously small (might be LFS pointer)
                if size_mb < 0.1 and 'combined_holdings' in file:
                    st.warning(f"⚠️ File {file} is very small ({size_mb:.2f} MB). This might be a Git LFS pointer file, not the actual data.")
            except Exception as e:
                file_info[file] = {"exists": True, "error": str(e)}
        else:
            file_info[file] = {"exists": False}
    
    missing_files = [f for f in essential_files if not file_info[f]["exists"]]
    if missing_files:
        st.error("❌ **Missing essential data files**")
        st.markdown("The following files are required but not found:")
        for file in missing_files:
            st.markdown(f"- `{file}`")
        st.info("💡 **Note:** If you're using Git LFS, ensure files are properly committed and pushed to the repository.")
        st.json(file_info)
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)
    
    # Try to load managers list (this will trigger data loading)
    try:
        managers_list = dashboard.get_managers_list()
    except Exception as e:
        st.error(f"❌ **Error loading managers list:** {e}")
        st.info("**Debugging info:**")
        debug_info = {
            "processed_data_exists": os.path.exists('processed_data'),
            "holdings_file_exists": os.path.exists('processed_data/combined_holdings.parquet'),
            "managers_file_exists": os.path.exists('processed_data/manager_summary.parquet'),
            "security_file_exists": os.path.exists('processed_data/security_summary.parquet'),
        }
        if os.path.exists('processed_data'):
            try:
                debug_info["files_in_processed_data"] = os.listdir('processed_data')
                # Check file sizes
                for file in debug_info["files_in_processed_data"]:
                    file_path = os.path.join('processed_data', file)
                    if os.path.isfile(file_path):
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        debug_info[f"{file}_size_mb"] = round(size_mb, 2)
            except Exception as list_error:
                debug_info["list_error"] = str(list_error)
        st.json(debug_info)
        st.stop()
    
    # Check if we have managers data
    if not managers_list:
        st.warning("⚠️ No managers data available. Please check that data files are properly loaded.")
        st.info("**Debugging info:**")
        debug_info = {
            "processed_data_exists": os.path.exists('processed_data'),
            "holdings_file_exists": os.path.exists('processed_data/combined_holdings.parquet'),
            "managers_file_exists": os.path.exists('processed_data/manager_summary.parquet'),
        }
        if os.path.exists('processed_data'):
            try:
                debug_info["files_in_processed_data"] = os.listdir('processed_data')
            except:
                pass
        st.json(debug_info)
        
        # Check if files exist but are empty or corrupted
        managers_file = 'processed_data/manager_summary.parquet'
        if os.path.exists(managers_file):
            st.info("💡 File exists but appears to be empty or unreadable. This may indicate:")
            st.markdown("- Git LFS file was not properly downloaded")
            st.markdown("- File is corrupted")
            st.markdown("- File permissions issue")
        st.stop()
    
    # Add search functionality
    st.sidebar.markdown("### 🔍 Search Manager")
    
    # Popular manager quick buttons
    st.sidebar.markdown("**Quick Select:**")
    col1, col2 = st.sidebar.columns(2)
    
    popular_managers = [
        "VANGUARD GROUP INC",
        "BlackRock, Inc.",
        "MORGAN STANLEY",
        "STATE STREET CORP",
        "FIDELITY MANAGEMENT & RESEARCH CO",
        "GOLDMAN SACHS GROUP INC"
    ]
    
    # Create quick select buttons
    for i, manager in enumerate(popular_managers):
        if i % 2 == 0:
            if col1.button(manager[:15] + "...", key=f"quick_{i}"):
                st.session_state.selected_quick_manager = manager
        else:
            if col2.button(manager[:15] + "...", key=f"quick_{i}"):
                st.session_state.selected_quick_manager = manager
    
    # Use quick selected manager if available
    if hasattr(st.session_state, 'selected_quick_manager'):
        search_term = st.session_state.selected_quick_manager
        delattr(st.session_state, 'selected_quick_manager')
    else:
        search_term = st.sidebar.text_input(
            "Type to search managers:",
            placeholder="e.g., Vanguard, BlackRock, Morgan Stanley...",
            help="Start typing to filter the manager list"
        )
    
    # Filter managers based on search term
    if search_term:
        # More sophisticated search - search in multiple parts of the name
        search_lower = search_term.lower()
        filtered_managers = []
        
        for manager in managers_list:
            manager_lower = manager.lower()
            # Check if search term appears anywhere in the manager name
            if search_lower in manager_lower:
                filtered_managers.append(manager)
        
        # Sort by relevance (exact matches first, then partial matches)
        filtered_managers.sort(key=lambda x: (
            0 if x.lower().startswith(search_lower) else 1,
            x.lower().find(search_lower),
            x
        ))
    else:
        filtered_managers = managers_list
    
    # Display filtered results
    if filtered_managers:
        selected_manager = st.sidebar.selectbox(
            "Select Investment Manager:",
            filtered_managers,
            index=0,
            help=f"Showing {len(filtered_managers)} of {len(managers_list)} managers"
        )
        
        # Show manager info if selected
        if selected_manager:
            managers_data = dashboard.get_managers_data()
            if not managers_data.empty and 'manager_name' in managers_data.columns:
                manager_info = managers_data[managers_data['manager_name'] == selected_manager]
                if not manager_info.empty:
                    manager_data = manager_info.iloc[0]
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("### 📊 Manager Info")
                    if 'total_market_value' in manager_data:
                        st.sidebar.metric("Portfolio Value", f"${manager_data['total_market_value']/1e9:.1f}B")
                    if 'total_holdings' in manager_data:
                        st.sidebar.metric("Total Holdings", f"{manager_data['total_holdings']:,}")
                    if 'unique_securities' in manager_data:
                        st.sidebar.metric("Unique Securities", f"{manager_data['unique_securities']:,}")
    else:
        st.sidebar.warning("No managers found matching your search")
        selected_manager = None
    
    # Quarter selection
    try:
        available_quarters = dashboard.get_available_quarters()
        if len(available_quarters) >= 2:
            current_quarter = st.sidebar.selectbox(
                "Current Quarter:",
                available_quarters,
                index=0
            )
            previous_quarter = st.sidebar.selectbox(
                "Previous Quarter:",
                available_quarters,
                index=1
            )
        elif len(available_quarters) == 1:
            st.sidebar.warning("Only one quarter of data available. Cannot compare quarters.")
            current_quarter = available_quarters[0]
            previous_quarter = available_quarters[0]
        else:
            st.sidebar.error("No quarter data available. Please check holdings data file.")
            st.stop()
    except Exception as e:
        st.sidebar.error(f"Error loading quarters: {e}")
        st.stop()
    
    # Main dashboard content
    if selected_manager and current_quarter and previous_quarter:
        # Calculate portfolio changes
        comparison_df, error = dashboard.calculate_portfolio_changes(
            selected_manager, current_quarter, previous_quarter
        )
        
        if error:
            st.error(f"❌ {error}")
        elif comparison_df is not None:
            # Create summary
            summary = dashboard.create_portfolio_summary(
                comparison_df, selected_manager, current_quarter, previous_quarter
            )
            
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Portfolio Value",
                    f"${summary['total_current_value']/1e9:.1f}B",
                    f"${summary['total_change']/1e9:+.1f}B ({summary['total_change_pct']:+.1f}%)"
                )
            
            with col2:
                st.metric(
                    "Number of Positions",
                    f"{summary['num_positions_current']:,}",
                    f"{summary['num_positions_current'] - summary['num_positions_previous']:+d}"
                )
            
            with col3:
                st.metric(
                    "New Positions",
                    f"{summary['num_new_positions']:,}",
                    f"${summary['new_positions_value']/1e6:.1f}M"
                )
            
            with col4:
                st.metric(
                    "Sold Positions",
                    f"{summary['num_sold_positions']:,}",
                    f"${summary['sold_positions_value']/1e6:.1f}M"
                )
            
            # Tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Portfolio Changes", "🆕 New Positions", "📉 Sold Positions", "📊 Analytics", "🏛️ CME Products"])
            
            with tab1:
                st.subheader("Portfolio Changes Overview")
                
                # Filter options
                col1, col2, col3 = st.columns(3)
                with col1:
                    show_only_changes = st.checkbox("Show only positions with changes", value=True)
                with col2:
                    min_change_threshold = st.number_input("Minimum change threshold ($M)", value=1.0, min_value=0.0)
                with col3:
                    change_type_filter = st.selectbox("Change Type", ["All", "New Position", "Existing", "Sold Position"])
                
                # Apply filters
                filtered_df = comparison_df.copy()
                
                if show_only_changes:
                    filtered_df = filtered_df[filtered_df['value_change'] != 0]
                
                filtered_df = filtered_df[abs(filtered_df['value_change']) >= min_change_threshold * 1e6]
                
                if change_type_filter != "All":
                    filtered_df = filtered_df[filtered_df['change_type'] == change_type_filter]
                
                # Display table
                if not filtered_df.empty:
                    # Use available columns, fallback to empty string if security_class not available
                    available_cols = filtered_df.columns.tolist()
                    if 'security_class' in available_cols:
                        display_df = filtered_df[['issuer_name', 'cusip', 'security_class', 'security_type', 'current_value', 'previous_value', 
                                               'value_change', 'current_pct', 'pct_change', 'change_type']].copy()
                        display_df.columns = ['Issuer', 'CUSIP', 'Security Description', 'Security Type', 'Current ($M)', 'Previous ($M)', 
                                            'Change ($M)', 'Current %', 'Change %', 'Change Type']
                    else:
                        # Fallback without security_class
                        display_df = filtered_df[['issuer_name', 'cusip', 'security_type', 'current_value', 'previous_value', 
                                               'value_change', 'current_pct', 'pct_change', 'change_type']].copy()
                        display_df.columns = ['Issuer', 'CUSIP', 'Security Type', 'Current ($M)', 'Previous ($M)', 
                                            'Change ($M)', 'Current %', 'Change %', 'Change Type']
                    
                    display_df['Current ($M)'] = display_df['Current ($M)'] / 1e6
                    display_df['Previous ($M)'] = display_df['Previous ($M)'] / 1e6
                    display_df['Change ($M)'] = display_df['Change ($M)'] / 1e6
                    
                    st.dataframe(display_df, width='stretch')
                else:
                    st.info("No positions meet the current filter criteria")
            
            with tab2:
                st.subheader("New Positions")
                new_positions = comparison_df[comparison_df['change_type'] == 'New Position']
                
                if not new_positions.empty:
                    # Chart of new positions
                    fig = px.bar(
                        new_positions.head(20),
                        x='issuer_name',
                        y='current_value',
                        title="Top 20 New Positions by Value",
                        labels={'current_value': 'Value ($)', 'issuer_name': 'Issuer'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, width='stretch')
                    
                    # Table of all new positions
                    st.subheader("All New Positions")
                    if 'security_class' in new_positions.columns:
                        new_display = new_positions[['issuer_name', 'cusip', 'security_class', 'security_type', 'current_value', 'current_pct']].copy()
                        new_display.columns = ['Issuer', 'CUSIP', 'Security Description', 'Type', 'Value ($M)', 'Portfolio %']
                    else:
                        new_display = new_positions[['issuer_name', 'cusip', 'security_type', 'current_value', 'current_pct']].copy()
                        new_display.columns = ['Issuer', 'CUSIP', 'Type', 'Value ($M)', 'Portfolio %']
                    new_display['Value ($M)'] = new_display['Value ($M)'] / 1e6
                    st.dataframe(new_display, width='stretch')
                else:
                    st.info("No new positions in this quarter")
            
            with tab3:
                st.subheader("Sold Positions")
                sold_positions = comparison_df[comparison_df['change_type'] == 'Sold Position']
                
                if not sold_positions.empty:
                    # Chart of sold positions
                    fig = px.bar(
                        sold_positions.head(20),
                        x='issuer_name',
                        y='previous_value',
                        title="Top 20 Sold Positions by Previous Value",
                        labels={'previous_value': 'Previous Value ($)', 'issuer_name': 'Issuer'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, width='stretch')
                    
                    # Table of all sold positions
                    st.subheader("All Sold Positions")
                    if 'security_class' in sold_positions.columns:
                        sold_display = sold_positions[['issuer_name', 'cusip', 'security_class', 'security_type', 'previous_value', 'previous_pct']].copy()
                        sold_display.columns = ['Issuer', 'CUSIP', 'Security Description', 'Type', 'Previous Value ($M)', 'Previous Portfolio %']
                    else:
                        sold_display = sold_positions[['issuer_name', 'cusip', 'security_type', 'previous_value', 'previous_pct']].copy()
                        sold_display.columns = ['Issuer', 'CUSIP', 'Type', 'Previous Value ($M)', 'Previous Portfolio %']
                    sold_display['Previous Value ($M)'] = sold_display['Previous Value ($M)'] / 1e6
                    st.dataframe(sold_display, width='stretch')
                else:
                    st.info("No positions were sold in this quarter")
            
            with tab4:
                st.subheader("Portfolio Analytics")
                
                # Portfolio composition changes
                col1, col2 = st.columns(2)
                
                with col1:
                    # Security type distribution
                    type_changes = comparison_df.groupby('security_type').agg({
                        'current_value': 'sum',
                        'previous_value': 'sum',
                        'value_change': 'sum'
                    }).reset_index()
                    
                    fig = px.pie(
                        type_changes,
                        values='current_value',
                        names='security_type',
                        title="Current Portfolio by Security Type"
                    )
                    st.plotly_chart(fig, width='stretch')
                
                with col2:
                    # Top changes
                    top_changes = comparison_df.nlargest(10, 'value_change')[['issuer_name', 'value_change']]
                    top_changes['value_change'] = top_changes['value_change'] / 1e6
                    
                    fig = px.bar(
                        top_changes,
                        x='value_change',
                        y='issuer_name',
                        orientation='h',
                        title="Top 10 Position Increases",
                        labels={'value_change': 'Change ($M)', 'issuer_name': 'Issuer'}
                    )
                    st.plotly_chart(fig, width='stretch')
            
            with tab5:
                dashboard.render_cme_products_tab(selected_manager, current_quarter, previous_quarter)
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("**Last updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.markdown("*Data source: SEC 13F Filings | Dashboard updates every 30 seconds when auto-refresh is enabled*")

if __name__ == "__main__":
    main()