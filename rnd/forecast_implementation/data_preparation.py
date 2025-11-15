"""
Data preparation module for forecasting pipeline.
Handles data loading, missing data imputation, and aggregation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparator:
    """Handles data preparation, aggregation, and splitting."""
    
    def __init__(self, data_file: Path, missing_month: str = "2020-04"):
        """
        Initialize data preparator.
        
        Args:
            data_file: Path to the CSV file
            missing_month: Missing month to handle (format: YYYY-MM)
        """
        self.data_file = data_file
        self.missing_month = missing_month
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV file."""
        logger.info(f"Loading data from {self.data_file}")
        try:
            df = pd.read_csv(self.data_file)
            df['Date'] = pd.to_datetime(df['Date'])
            logger.info(f"Loaded {len(df):,} rows")
            logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            logger.info(f"Branches: {df['Branch'].unique().tolist()}")
            self.raw_data = df
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def handle_missing_month(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing month by imputation.
        
        Args:
            df: DataFrame with Date column
            
        Returns:
            DataFrame with missing month imputed
        """
        logger.info(f"Handling missing month: {self.missing_month}")
        
        # Create monthly aggregated data to check for missing month
        df_monthly = df.copy()
        df_monthly['YearMonth'] = df_monthly['Date'].dt.to_period('M')
        
        # Check if missing month exists
        missing_period = pd.Period(self.missing_month)
        existing_periods = df_monthly['YearMonth'].unique()
        
        if missing_period not in existing_periods:
            logger.info(f"Missing month {self.missing_month} detected. Imputing...")
            
            # Get surrounding months
            prev_month = missing_period - 1
            next_month = missing_period + 1
            
            # Get data from previous and next months
            prev_data = df_monthly[df_monthly['YearMonth'] == prev_month].copy()
            next_data = df_monthly[df_monthly['YearMonth'] == next_month].copy()
            
            if len(prev_data) > 0 and len(next_data) > 0:
                # Interpolate: average of previous and next month
                # Create synthetic records for missing month
                missing_date = pd.Timestamp(f"{self.missing_month}-15")  # Mid-month
                
                # Group by Branch and aggregate
                prev_agg = prev_data.groupby(['Branch', 'Item Code', 'Star Rating', 
                                             'Segment', 'Region']).agg({
                    'Quantity': 'mean',
                    'Tonnage': 'mean'
                }).reset_index()
                
                next_agg = next_data.groupby(['Branch', 'Item Code', 'Star Rating',
                                             'Segment', 'Region']).agg({
                    'Quantity': 'mean',
                    'Tonnage': 'mean'
                }).reset_index()
                
                # Merge and average
                merged = prev_agg.merge(next_agg, on=['Branch', 'Item Code', 'Star Rating',
                                                      'Segment', 'Region'], 
                                       suffixes=('_prev', '_next'))
                merged['Quantity'] = (merged['Quantity_prev'] + merged['Quantity_next']) / 2
                merged['Tonnage'] = (merged['Tonnage_prev'] + merged['Tonnage_next']) / 2
                merged['Date'] = missing_date
                merged = merged[['Date', 'Item Code', 'Branch', 'Star Rating', 
                                'Segment', 'Quantity', 'Tonnage', 'Region']]
                
                # Add to dataframe
                df = pd.concat([df, merged], ignore_index=True)
                logger.info(f"Imputed {len(merged)} records for missing month")
            else:
                logger.warning(f"Could not impute missing month - insufficient surrounding data")
        else:
            logger.info(f"Month {self.missing_month} exists in data")
        
        return df.sort_values('Date').reset_index(drop=True)
    
    def aggregate_monthly(self, df: pd.DataFrame, group_by: Optional[list] = None) -> pd.DataFrame:
        """
        Aggregate data to monthly level.
        
        Args:
            df: DataFrame with Date and Quantity columns
            group_by: Columns to group by (None for combined)
            
        Returns:
            Monthly aggregated DataFrame
        """
        df = df.copy()
        df['YearMonth'] = df['Date'].dt.to_period('M')
        
        if group_by is None:
            # Combined aggregation
            monthly = df.groupby('YearMonth').agg({
                'Quantity': 'sum',
                'Tonnage': 'sum'
            }).reset_index()
            monthly['Date'] = monthly['YearMonth'].dt.to_timestamp()
            monthly = monthly[['Date', 'Quantity', 'Tonnage']]
        else:
            # Branch-wise or other grouping
            group_cols = group_by + ['YearMonth']
            monthly = df.groupby(group_cols).agg({
                'Quantity': 'sum',
                'Tonnage': 'sum'
            }).reset_index()
            monthly['Date'] = monthly['YearMonth'].dt.to_timestamp()
            monthly = monthly[['Date'] + group_by + ['Quantity', 'Tonnage']]
        
        return monthly.sort_values('Date').reset_index(drop=True)
    
    def aggregate_weekly(self, df: pd.DataFrame, group_by: Optional[list] = None) -> pd.DataFrame:
        """
        Aggregate data to weekly level.
        
        Args:
            df: DataFrame with Date and Quantity columns
            group_by: Columns to group by (None for combined)
            
        Returns:
            Weekly aggregated DataFrame
        """
        df = df.copy()
        df['YearWeek'] = df['Date'].dt.to_period('W-MON')
        
        if group_by is None:
            # Combined aggregation
            weekly = df.groupby('YearWeek').agg({
                'Quantity': 'sum',
                'Tonnage': 'sum'
            }).reset_index()
            weekly['Date'] = weekly['YearWeek'].dt.to_timestamp()
            weekly = weekly[['Date', 'Quantity', 'Tonnage']]
        else:
            # Branch-wise or other grouping
            group_cols = group_by + ['YearWeek']
            weekly = df.groupby(group_cols).agg({
                'Quantity': 'sum',
                'Tonnage': 'sum'
            }).reset_index()
            weekly['Date'] = weekly['YearWeek'].dt.to_timestamp()
            weekly = weekly[['Date'] + group_by + ['Quantity', 'Tonnage']]
        
        return weekly.sort_values('Date').reset_index(drop=True)
    
    def prepare_aggregated_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Prepare all aggregated datasets.
        
        Returns:
            Dictionary with structure:
            {
                'monthly': {
                    'combined': DataFrame,
                    'branch_wise': DataFrame
                },
                'weekly': {
                    'combined': DataFrame,
                    'branch_wise': DataFrame
                }
            }
        """
        if self.raw_data is None:
            self.load_data()
        
        df = self.handle_missing_month(self.raw_data)
        
        logger.info("Creating monthly aggregates...")
        monthly_combined = self.aggregate_monthly(df, group_by=None)
        monthly_branch = self.aggregate_monthly(df, group_by=['Branch'])
        
        logger.info("Creating weekly aggregates...")
        weekly_combined = self.aggregate_weekly(df, group_by=None)
        weekly_branch = self.aggregate_weekly(df, group_by=['Branch'])
        
        return {
            'monthly': {
                'combined': monthly_combined,
                'branch_wise': monthly_branch
            },
            'weekly': {
                'combined': weekly_combined,
                'branch_wise': weekly_branch
            }
        }
    
    def split_data(self, df: pd.DataFrame, 
                   train_end: str,
                   calibration_start: str,
                   calibration_end: str,
                   blind_start: str) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, calibration, and blind sets.
        
        Args:
            df: DataFrame with Date column
            train_end: End date for training (inclusive)
            calibration_start: Start date for calibration
            calibration_end: End date for calibration (inclusive)
            blind_start: Start date for blind evaluation
            
        Returns:
            Dictionary with 'train', 'calibration', 'blind' DataFrames
        """
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        train_end_dt = pd.to_datetime(train_end)
        calibration_start_dt = pd.to_datetime(calibration_start)
        calibration_end_dt = pd.to_datetime(calibration_end)
        blind_start_dt = pd.to_datetime(blind_start)
        
        train = df[df['Date'] <= train_end_dt].copy()
        calibration = df[(df['Date'] >= calibration_start_dt) & 
                        (df['Date'] <= calibration_end_dt)].copy()
        blind = df[df['Date'] >= blind_start_dt].copy()
        
        logger.info(f"Train: {len(train):,} rows ({train['Date'].min()} to {train['Date'].max()})")
        logger.info(f"Calibration: {len(calibration):,} rows ({calibration['Date'].min()} to {calibration['Date'].max()})")
        logger.info(f"Blind: {len(blind):,} rows ({blind['Date'].min()} to {blind['Date'].max()})")
        
        return {
            'train': train,
            'calibration': calibration,
            'blind': blind
        }
    
    def prepare_series(self, df: pd.DataFrame, 
                      value_col: str = 'Quantity',
                      group_by: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Prepare time series for forecasting.
        
        Args:
            df: Aggregated DataFrame
            value_col: Column to use as values
            group_by: Column to group by (e.g., 'Branch')
            
        Returns:
            Dictionary of time series (keyed by group if group_by specified)
        """
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        if group_by is None:
            # Single series
            series = pd.Series(df[value_col].values, index=df['Date'])
            return {'combined': series}
        else:
            # Multiple series (e.g., by branch)
            series_dict = {}
            for group_value in df[group_by].unique():
                group_df = df[df[group_by] == group_value]
                series = pd.Series(group_df[value_col].values, index=group_df['Date'])
                series_dict[str(group_value)] = series
            return series_dict

