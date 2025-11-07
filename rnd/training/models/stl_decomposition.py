"""
STL Decomposition model for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Optional
import pickle
from pathlib import Path
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA


def train_model(train_data: pd.Series, branch: str, freq: str = 'W-MON',
                use_optimization: bool = True, n_trials: int = 20) -> Optional[dict]:
    """
    Train STL Decomposition model.
    
    Args:
        train_data: Training time series
        branch: Branch name
        freq: Frequency string
        
    Returns:
        Dictionary with STL decomposition and trend model, or None if training fails
    """
    if len(train_data) < 20:
        print(f"Warning: Insufficient data for STL Decomposition for {branch}")
        return None
    
    try:
        # Determine seasonal period
        if freq == 'W-MON':
            seasonal_period = 52  # Weekly data, yearly seasonality
        elif freq == 'MS':
            seasonal_period = 12  # Monthly data, yearly seasonality
        else:
            seasonal_period = 12
        
        # Ensure we have enough data for seasonality
        if len(train_data) < seasonal_period * 2:
            seasonal_period = len(train_data) // 2
        
        # Perform STL decomposition
        stl = STL(train_data.values, seasonal=seasonal_period, robust=True)
        decomposition = stl.fit()
        
        # Fit ARIMA model on trend component
        trend = decomposition.trend
        # Remove NaN values
        trend_clean = trend[~np.isnan(trend)]
        
        if len(trend_clean) < 10:
            # Use simple moving average for trend
            trend_model = None
        else:
            try:
                trend_model = ARIMA(trend_clean, order=(1, 1, 1)).fit()
            except:
                trend_model = None
        
        return {
            'decomposition': decomposition,
            'trend_model': trend_model,
            'seasonal_period': seasonal_period
        }
    except Exception as e:
        print(f"Error training STL Decomposition for {branch}: {e}")
        return None


def forecast(model: dict, n_periods: int, freq: str = 'W-MON', branch: str = None) -> pd.Series:
    """
    Generate forecast using trained model.
    
    Args:
        model: Dictionary with decomposition and trend model
        n_periods: Number of periods to forecast
        freq: Frequency string
        branch: Branch name (not used)
        
    Returns:
        Forecast series with Date index
    """
    if model is None:
        return pd.Series(dtype=float)
    
    try:
        decomposition = model['decomposition']
        trend_model = model['trend_model']
        seasonal_period = model['seasonal_period']
        
        # Forecast trend component
        if trend_model is not None:
            trend_forecast = trend_model.forecast(steps=n_periods)
        else:
            # Use last trend value
            last_trend = decomposition.trend[-1]
            trend_forecast = np.full(n_periods, last_trend)
        
        # Get seasonal component (repeat last seasonal pattern)
        seasonal = decomposition.seasonal
        seasonal_pattern = seasonal[-seasonal_period:]
        
        # Repeat seasonal pattern
        n_repeats = (n_periods // len(seasonal_pattern)) + 1
        seasonal_forecast = np.tile(seasonal_pattern, n_repeats)[:n_periods]
        
        # Combine trend and seasonal
        forecast_values = trend_forecast + seasonal_forecast
        
        # Create date index
        forecast_dates = pd.date_range(start=pd.Timestamp.now(), periods=n_periods, freq=freq)
        
        return pd.Series(forecast_values, index=forecast_dates)
    except Exception as e:
        print(f"Error forecasting with STL Decomposition: {e}")
        return pd.Series(dtype=float)


def save_model(model: dict, path: str) -> None:
    """Save model to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path: str) -> dict:
    """Load model from file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

