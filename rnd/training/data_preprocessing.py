"""
Data preprocessing utilities for time series forecasting.
Handles loading, filtering, and aggregating data by Branch and Quantity.
Includes anomaly detection and YoY growth feature calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training.anomaly_detection import AnomalyDetector, detect_anomalies_in_series


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
    Handles missing dates like 2020-04-01.
    
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
    
    # Check for missing dates (like 2020-04-01) and fill them
    for branch in weekly_df['Branch'].unique():
        branch_data = weekly_df[weekly_df['Branch'] == branch].copy()
        if len(branch_data) > 0:
            # Create complete date range
            start_date = branch_data['Date'].min()
            end_date = branch_data['Date'].max()
            complete_range = pd.date_range(start=start_date, end=end_date, freq='W-MON')
            
            # Check for missing dates
            missing_dates = set(complete_range) - set(branch_data['Date'])
            if len(missing_dates) > 0:
                # Fill missing dates with forward fill
                branch_data = branch_data.set_index('Date')
                branch_data = branch_data.reindex(complete_range, method='ffill', fill_value=0)
                branch_data = branch_data.reset_index()
                branch_data['Branch'] = branch
                branch_data['Date'] = branch_data['index']
                branch_data = branch_data.drop('index', axis=1)
                
                # Update weekly_df with filled data
                weekly_df = weekly_df[weekly_df['Branch'] != branch]
                weekly_df = pd.concat([weekly_df, branch_data], ignore_index=True)
                
                if len(missing_dates) > 0:
                    print(f"Warning: Filled {len(missing_dates)} missing weekly dates for {branch}, including 2020-04-01 if applicable")
    
    return weekly_df.sort_values(['Branch', 'Date']).reset_index(drop=True)


def create_monthly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create monthly aggregates by summing Quantity per Branch.
    Handles missing dates like 2020-04-01.
    
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
    
    # Check for missing dates (like 2020-04-01) and fill them
    for branch in monthly_df['Branch'].unique():
        branch_data = monthly_df[monthly_df['Branch'] == branch].copy()
        if len(branch_data) > 0:
            # Create complete date range
            start_date = branch_data['Date'].min()
            end_date = branch_data['Date'].max()
            complete_range = pd.date_range(start=start_date, end=end_date, freq='MS')
            
            # Check for missing dates
            missing_dates = set(complete_range) - set(branch_data['Date'])
            if len(missing_dates) > 0:
                # Fill missing dates with forward fill
                branch_data = branch_data.set_index('Date')
                branch_data = branch_data.reindex(complete_range, method='ffill', fill_value=0)
                branch_data = branch_data.reset_index()
                branch_data['Branch'] = branch
                branch_data['Date'] = branch_data['index']
                branch_data = branch_data.drop('index', axis=1)
                
                # Update monthly_df with filled data
                monthly_df = monthly_df[monthly_df['Branch'] != branch]
                monthly_df = pd.concat([monthly_df, branch_data], ignore_index=True)
                
                # Check specifically for 2020-04-01
                if pd.Timestamp('2020-04-01') in missing_dates:
                    print(f"Warning: Filled missing date 2020-04-01 for {branch} (monthly aggregate)")
                elif len(missing_dates) > 0:
                    print(f"Warning: Filled {len(missing_dates)} missing monthly dates for {branch}")
    
    return monthly_df.sort_values(['Branch', 'Date']).reset_index(drop=True)


def get_branch_series(df: pd.DataFrame, branch: str, freq: str = 'W-MON') -> pd.Series:
    """
    Extract time series for a specific branch.
    Handles missing dates by filling gaps.
    
    Args:
        df: Aggregated dataframe (weekly or monthly)
        branch: Branch name
        freq: Frequency string for filling missing dates
        
    Returns:
        Time series with Date index and Quantity values
    """
    branch_data = df[df['Branch'] == branch].copy()
    branch_data = branch_data.sort_values('Date')
    branch_data = branch_data.set_index('Date')
    
    series = branch_data['Quantity']
    
    # Fill any remaining missing dates
    if len(series) > 0:
        series = fill_missing_dates(series, freq)
    
    return series


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


def calculate_yoy_growth_rate(series: pd.Series, freq: str = 'W-MON') -> pd.Series:
    """
    Calculate year-over-year growth rate for a time series.
    
    Args:
        series: Time series with Date index
        freq: Frequency string ('W-MON' for weekly, 'MS' for monthly)
        
    Returns:
        Series with YoY growth rates (as percentages)
    """
    if len(series) < 2:
        return pd.Series(dtype=float, index=series.index)
    
    # Determine period shift based on frequency
    if freq == 'W-MON':
        periods = 52  # 52 weeks in a year
    elif freq == 'MS':
        periods = 12  # 12 months in a year
    else:
        periods = 12  # Default to monthly
    
    # Shift by one year
    series_shifted = series.shift(periods)
    
    # Calculate YoY growth rate: (current - previous_year) / previous_year * 100
    yoy_growth = ((series - series_shifted) / series_shifted) * 100
    
    # Replace inf and NaN with 0
    yoy_growth = yoy_growth.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return yoy_growth


def add_anomaly_detection_to_series(
    series: pd.Series,
    detector: Optional[AnomalyDetector] = None,
    contamination: float = 0.1
) -> Tuple[pd.Series, pd.Series, pd.Series, Optional[AnomalyDetector]]:
    """
    Add anomaly detection to a time series.
    
    Args:
        series: Time series with Date index
        detector: Pre-fitted detector (if None, will fit on this series)
        contamination: Expected proportion of anomalies (used if detector is None)
        
    Returns:
        Tuple of (series_with_anomalies, labels_series, severity_series, fitted_detector)
    """
    if len(series) == 0:
        labels = pd.Series(dtype=int, index=series.index, name='anomaly_label')
        severity = pd.Series(dtype=float, index=series.index, name='anomaly_severity')
        return series, labels, severity, None
    
    if detector is None:
        # Fit new detector
        labels, severity, stats = detect_anomalies_in_series(
            series, contamination=contamination
        )
        # Create detector from stats (we'll need to refit for transform)
        detector = AnomalyDetector(contamination=contamination)
        detector.fit(series.values)
    else:
        # Use existing detector
        labels, severity_scores, _ = detector.transform(series.values)
        labels = pd.Series(labels, index=series.index, name='anomaly_label')
        severity = pd.Series(severity_scores, index=series.index, name='anomaly_severity')
    
    return series, labels, severity, detector


def prepare_data_for_training(
    train_path: str,
    test_path: str,
    branches: List[str] = None,
    apply_anomaly_detection: bool = True,
    contamination: float = 0.1,
    use_combined_data: bool = True
) -> Dict:
    """
    Prepare data for training by creating weekly and monthly aggregates per branch.
    Now includes anomaly detection and YoY growth features.
    Supports combined data mode for using all available data.
    
    Args:
        train_path: Path to training CSV file (2024 data)
        test_path: Path to test CSV file (2025 data)
        branches: List of branch names to process. If None, processes all branches.
        apply_anomaly_detection: Whether to apply anomaly detection
        contamination: Expected proportion of anomalies
        use_combined_data: If True, combines 2024 and 2025 data, then splits into train/val/test
        
    Returns:
        Dictionary with structure:
        {
            'train': {
                'weekly': {
                    branch: {
                        'series': pd.Series,
                        'anomaly_labels': pd.Series,
                        'anomaly_severity': pd.Series,
                        'yoy_growth': pd.Series
                    }
                },
                'monthly': {same structure}
            },
            'validation': {same structure} - only if use_combined_data=True
            'test': {same structure}
            'detectors': {
                'weekly': {branch: AnomalyDetector},
                'monthly': {branch: AnomalyDetector}
            }
        }
    """
    # Load data
    train_df, test_df = load_data(train_path, test_path)
    
    # Filter to Branch and Quantity
    train_df = filter_by_branch_quantity(train_df)
    test_df = filter_by_branch_quantity(test_df)
    
    if use_combined_data:
        # Combine 2024 and 2025 data
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        combined_df = combined_df.sort_values(['Branch', 'Date'])
        
        # Define split dates for combined data
        # Train: 2020-03-23 to 2024-03-31
        # Validation: 2024-04-01 to 2024-12-31
        # Test: 2025-01-01 to 2025-03-31
        train_end = pd.Timestamp('2024-03-31')
        val_end = pd.Timestamp('2024-12-31')
        test_start = pd.Timestamp('2025-01-01')
        test_end = pd.Timestamp('2025-03-31')
        
        # Split combined data
        train_combined = combined_df[combined_df['Date'] <= train_end].copy()
        val_combined = combined_df[(combined_df['Date'] > train_end) & (combined_df['Date'] <= val_end)].copy()
        test_combined = combined_df[(combined_df['Date'] >= test_start) & (combined_df['Date'] <= test_end)].copy()
        
        # Create aggregates for each split
        train_weekly = create_weekly_aggregates(train_combined)
        train_monthly = create_monthly_aggregates(train_combined)
        val_weekly = create_weekly_aggregates(val_combined)
        val_monthly = create_monthly_aggregates(val_combined)
        test_weekly = create_weekly_aggregates(test_combined)
        test_monthly = create_monthly_aggregates(test_combined)
    else:
        # Original behavior: separate train and test
        train_weekly = create_weekly_aggregates(train_df)
        train_monthly = create_monthly_aggregates(train_df)
        test_weekly = create_weekly_aggregates(test_df)
        test_monthly = create_monthly_aggregates(test_df)
        val_weekly = pd.DataFrame(columns=['Branch', 'Date', 'Quantity'])
        val_monthly = pd.DataFrame(columns=['Branch', 'Date', 'Quantity'])
    
    # Get unique branches
    if branches is None:
        branches = sorted(train_df['Branch'].unique())
    
    # Extract series per branch and apply anomaly detection
    result = {
        'train': {
            'weekly': {},
            'monthly': {}
        },
        'test': {
            'weekly': {},
            'monthly': {}
        },
        'detectors': {
            'weekly': {},
            'monthly': {}
        }
    }
    
    # Add validation split if using combined data
    if use_combined_data:
        result['validation'] = {
            'weekly': {},
            'monthly': {}
        }
    
    for branch in branches:
        for aggregation in ['weekly', 'monthly']:
            freq = 'W-MON' if aggregation == 'weekly' else 'MS'
            
            # Training data
            try:
                train_series = get_branch_series(
                    train_weekly if aggregation == 'weekly' else train_monthly,
                    branch,
                    freq
                )
                
                if len(train_series) > 0:
                    # Calculate YoY growth
                    yoy_growth = calculate_yoy_growth_rate(train_series, freq)
                    
                    # Apply anomaly detection
                    if apply_anomaly_detection:
                        _, labels, severity, detector = add_anomaly_detection_to_series(
                            train_series, None, contamination
                        )
                        result['detectors'][aggregation][branch] = detector
                    else:
                        labels = pd.Series(0, index=train_series.index, name='anomaly_label')
                        severity = pd.Series(0.0, index=train_series.index, name='anomaly_severity')
                        result['detectors'][aggregation][branch] = None
                    
                    result['train'][aggregation][branch] = {
                        'series': train_series,
                        'anomaly_labels': labels,
                        'anomaly_severity': severity,
                        'yoy_growth': yoy_growth
                    }
                else:
                    result['train'][aggregation][branch] = {
                        'series': pd.Series(dtype=float),
                        'anomaly_labels': pd.Series(dtype=int),
                        'anomaly_severity': pd.Series(dtype=float),
                        'yoy_growth': pd.Series(dtype=float)
                    }
            except Exception as e:
                print(f"Warning: Could not create {aggregation} series for {branch}: {e}")
                result['train'][aggregation][branch] = {
                    'series': pd.Series(dtype=float),
                    'anomaly_labels': pd.Series(dtype=int),
                    'anomaly_severity': pd.Series(dtype=float),
                    'yoy_growth': pd.Series(dtype=float)
                }
            
            # Validation data (if using combined data)
            if use_combined_data:
                try:
                    val_series = get_branch_series(
                        val_weekly if aggregation == 'weekly' else val_monthly,
                        branch,
                        freq
                    )
                    
                    if len(val_series) > 0:
                        # Calculate YoY growth
                        yoy_growth = calculate_yoy_growth_rate(val_series, freq)
                        
                        # Apply anomaly detection using detector from training
                        detector = result['detectors'][aggregation].get(branch)
                        if apply_anomaly_detection and detector is not None:
                            _, labels, severity, _ = add_anomaly_detection_to_series(
                                val_series, detector, contamination
                            )
                        else:
                            labels = pd.Series(0, index=val_series.index, name='anomaly_label')
                            severity = pd.Series(0.0, index=val_series.index, name='anomaly_severity')
                        
                        result['validation'][aggregation][branch] = {
                            'series': val_series,
                            'anomaly_labels': labels,
                            'anomaly_severity': severity,
                            'yoy_growth': yoy_growth
                        }
                    else:
                        result['validation'][aggregation][branch] = {
                            'series': pd.Series(dtype=float),
                            'anomaly_labels': pd.Series(dtype=int),
                            'anomaly_severity': pd.Series(dtype=float),
                            'yoy_growth': pd.Series(dtype=float)
                        }
                except Exception as e:
                    print(f"Warning: Could not create {aggregation} validation series for {branch}: {e}")
                    result['validation'][aggregation][branch] = {
                        'series': pd.Series(dtype=float),
                        'anomaly_labels': pd.Series(dtype=int),
                        'anomaly_severity': pd.Series(dtype=float),
                        'yoy_growth': pd.Series(dtype=float)
                    }
            
            # Test data
            try:
                test_series = get_branch_series(
                    test_weekly if aggregation == 'weekly' else test_monthly,
                    branch,
                    freq
                )
                
                if len(test_series) > 0:
                    # Calculate YoY growth
                    yoy_growth = calculate_yoy_growth_rate(test_series, freq)
                    
                    # Apply anomaly detection using detector from training
                    detector = result['detectors'][aggregation].get(branch)
                    if apply_anomaly_detection and detector is not None:
                        _, labels, severity, _ = add_anomaly_detection_to_series(
                            test_series, detector, contamination
                        )
                    else:
                        labels = pd.Series(0, index=test_series.index, name='anomaly_label')
                        severity = pd.Series(0.0, index=test_series.index, name='anomaly_severity')
                    
                    result['test'][aggregation][branch] = {
                        'series': test_series,
                        'anomaly_labels': labels,
                        'anomaly_severity': severity,
                        'yoy_growth': yoy_growth
                    }
                else:
                    result['test'][aggregation][branch] = {
                        'series': pd.Series(dtype=float),
                        'anomaly_labels': pd.Series(dtype=int),
                        'anomaly_severity': pd.Series(dtype=float),
                        'yoy_growth': pd.Series(dtype=float)
                    }
            except Exception as e:
                print(f"Warning: Could not create {aggregation} test series for {branch}: {e}")
                result['test'][aggregation][branch] = {
                    'series': pd.Series(dtype=float),
                    'anomaly_labels': pd.Series(dtype=int),
                    'anomaly_severity': pd.Series(dtype=float),
                    'yoy_growth': pd.Series(dtype=float)
                }
    
    return result

