"""
Utility functions for forecasting pipeline.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple

def detect_frequency(series: pd.Series) -> Tuple[str, int]:
    """
    Detect frequency of a time series.
    
    Args:
        series: Time series with datetime index
        
    Returns:
        Tuple of (freq_string, periods_per_year)
        freq_string: pandas frequency string ('MS' for monthly, 'W-MON' for weekly, etc.)
        periods_per_year: Number of periods per year
    """
    if len(series) < 2:
        return 'MS', 12  # Default to monthly
    
    # Try to infer frequency
    inferred_freq = pd.infer_freq(series.index)
    
    if inferred_freq:
        if 'M' in inferred_freq or 'MS' in inferred_freq:
            return 'MS', 12
        elif 'W' in inferred_freq:
            return 'W-MON', 52
        elif 'D' in inferred_freq:
            return 'D', 365
        else:
            return inferred_freq, 12
    
    # Fallback: estimate from date differences
    if len(series) > 1:
        date_diffs = series.index.to_series().diff().dropna()
        avg_diff_days = date_diffs.dt.days.mean()
        
        if avg_diff_days > 25:
            return 'MS', 12  # Monthly
        elif avg_diff_days > 5:
            return 'W-MON', 52  # Weekly
        else:
            return 'D', 365  # Daily
    
    return 'MS', 12  # Default to monthly

def get_date_offset(freq: str, periods: int = 1):
    """
    Get appropriate date offset for frequency.
    
    Args:
        freq: Frequency string ('MS', 'W-MON', 'D', etc.)
        periods: Number of periods
        
    Returns:
        DateOffset object
    """
    if freq == 'MS' or 'M' in freq:
        return pd.DateOffset(months=periods)
    elif 'W' in freq:
        return pd.DateOffset(weeks=periods)
    elif 'D' in freq:
        return pd.DateOffset(days=periods)
    else:
        # Default to months
        return pd.DateOffset(months=periods)

def create_future_dates(last_date: pd.Timestamp, periods: int, freq: str) -> pd.DatetimeIndex:
    """
    Create future dates for forecasting.
    
    Args:
        last_date: Last date in the series
        periods: Number of periods to forecast
        freq: Frequency string
        
    Returns:
        DatetimeIndex of future dates
    """
    offset = get_date_offset(freq, 1)
    start_date = last_date + offset
    
    # For weekly frequency, ensure we start on a Monday
    if freq == 'W-MON' or 'W' in freq:
        # If start_date is not a Monday, find the next Monday
        if start_date.weekday() != 0:  # 0 = Monday
            days_until_monday = (7 - start_date.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            start_date = start_date + pd.DateOffset(days=days_until_monday)
    
    return pd.date_range(start=start_date, periods=periods, freq=freq)

