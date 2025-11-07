"""
Data preprocessing utilities for time series forecasting.
Handles loading, filtering, and aggregating data by Branch and Quantity.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test data from CSV files.
    
    Args:
        train_path: Path to training CSV file
        test_path: Path to test CSV file
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Convert Date column to datetime
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    
    return train_df, test_df


def filter_by_branch_quantity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe to only include Branch and Quantity columns.
    
    Args:
        df: Input dataframe
        
    Returns:
        Filtered dataframe with Date, Branch, and Quantity columns
    """
    return df[['Date', 'Branch', 'Quantity']].copy()


def create_weekly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weekly aggregates by summing Quantity per Branch.
    Week starts on Monday (W-MON).
    
    Args:
        df: Dataframe with Date, Branch, and Quantity columns
        
    Returns:
        Aggregated dataframe with weekly sums
    """
    df = df.copy()
    df = df.sort_values(['Branch', 'Date'])
    
    # Set Date as index for resampling
    df = df.set_index('Date')
    
    # Group by Branch and resample to weekly (Monday start)
    weekly_df = df.groupby('Branch').resample('W-MON')['Quantity'].sum().reset_index()
    
    # Ensure non-negative quantities
    weekly_df['Quantity'] = weekly_df['Quantity'].abs()
    
    return weekly_df.sort_values(['Branch', 'Date']).reset_index(drop=True)


def create_monthly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create monthly aggregates by summing Quantity per Branch.
    
    Args:
        df: Dataframe with Date, Branch, and Quantity columns
        
    Returns:
        Aggregated dataframe with monthly sums
    """
    df = df.copy()
    df = df.sort_values(['Branch', 'Date'])
    
    # Set Date as index for resampling
    df = df.set_index('Date')
    
    # Group by Branch and resample to monthly (month start)
    monthly_df = df.groupby('Branch').resample('MS')['Quantity'].sum().reset_index()
    
    # Ensure non-negative quantities
    monthly_df['Quantity'] = monthly_df['Quantity'].abs()
    
    return monthly_df.sort_values(['Branch', 'Date']).reset_index(drop=True)


def get_branch_series(df: pd.DataFrame, branch: str) -> pd.Series:
    """
    Extract time series for a specific branch.
    
    Args:
        df: Aggregated dataframe (weekly or monthly)
        branch: Branch name
        
    Returns:
        Time series with Date index and Quantity values
    """
    branch_data = df[df['Branch'] == branch].copy()
    branch_data = branch_data.sort_values('Date')
    branch_data = branch_data.set_index('Date')
    
    return branch_data['Quantity']


def prepare_data_for_training(
    train_path: str,
    test_path: str,
    branches: List[str] = None
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Prepare data for training by creating weekly and monthly aggregates per branch.
    
    Args:
        train_path: Path to training CSV file
        test_path: Path to test CSV file
        branches: List of branch names to process. If None, processes all branches.
        
    Returns:
        Dictionary with structure:
        {
            'train': {
                'weekly': {branch: pd.Series},
                'monthly': {branch: pd.Series}
            },
            'test': {
                'weekly': {branch: pd.Series},
                'monthly': {branch: pd.Series}
            }
        }
    """
    # Load data
    train_df, test_df = load_data(train_path, test_path)
    
    # Filter to Branch and Quantity
    train_df = filter_by_branch_quantity(train_df)
    test_df = filter_by_branch_quantity(test_df)
    
    # Create aggregates
    train_weekly = create_weekly_aggregates(train_df)
    train_monthly = create_monthly_aggregates(train_df)
    test_weekly = create_weekly_aggregates(test_df)
    test_monthly = create_monthly_aggregates(test_df)
    
    # Get unique branches
    if branches is None:
        branches = sorted(train_df['Branch'].unique())
    
    # Extract series per branch
    result = {
        'train': {
            'weekly': {},
            'monthly': {}
        },
        'test': {
            'weekly': {},
            'monthly': {}
        }
    }
    
    for branch in branches:
        # Training data
        try:
            result['train']['weekly'][branch] = get_branch_series(train_weekly, branch)
        except Exception as e:
            print(f"Warning: Could not create weekly series for {branch}: {e}")
            result['train']['weekly'][branch] = pd.Series(dtype=float)
        
        try:
            result['train']['monthly'][branch] = get_branch_series(train_monthly, branch)
        except Exception as e:
            print(f"Warning: Could not create monthly series for {branch}: {e}")
            result['train']['monthly'][branch] = pd.Series(dtype=float)
        
        # Test data
        try:
            result['test']['weekly'][branch] = get_branch_series(test_weekly, branch)
        except Exception as e:
            print(f"Warning: Could not create weekly test series for {branch}: {e}")
            result['test']['weekly'][branch] = pd.Series(dtype=float)
        
        try:
            result['test']['monthly'][branch] = get_branch_series(test_monthly, branch)
        except Exception as e:
            print(f"Warning: Could not create monthly test series for {branch}: {e}")
            result['test']['monthly'][branch] = pd.Series(dtype=float)
    
    return result


def fill_missing_dates(series: pd.Series, freq: str = 'W-MON') -> pd.Series:
    """
    Fill missing dates in a time series with forward fill or zero.
    
    Args:
        series: Time series with Date index
        freq: Frequency string ('W-MON' for weekly, 'MS' for monthly)
        
    Returns:
        Series with missing dates filled
    """
    # Create complete date range
    start_date = series.index.min()
    end_date = series.index.max()
    complete_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Reindex to complete range
    filled_series = series.reindex(complete_range, fill_value=0)
    
    # Forward fill zeros if needed
    filled_series = filled_series.replace(0, np.nan).ffill().fillna(0)
    
    return filled_series

