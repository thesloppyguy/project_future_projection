"""
Kalman Filter model for time series forecasting.
Uses state space model for forecasting.
"""

import pandas as pd
import numpy as np
from typing import Optional
import pickle
from pathlib import Path
from statsmodels.tsa.statespace.structural import UnobservedComponents


def train_model(train_data: pd.Series, branch: str, freq: str = 'W-MON',
                use_optimization: bool = True, n_trials: int = 20) -> Optional[object]:
    """
    Train Kalman Filter model (using UnobservedComponents from statsmodels).
    
    Args:
        train_data: Training time series
        branch: Branch name
        freq: Frequency string
        
    Returns:
        Trained model or None if training fails
    """
    if len(train_data) < 20:
        print(f"Warning: Insufficient data for Kalman Filter for {branch}")
        return None
    
    try:
        # Determine seasonal period
        if freq == 'W-MON':
            seasonal_period = 52  # Weekly data, yearly seasonality
        elif freq == 'MS':
            seasonal_period = 12  # Monthly data, yearly seasonality
        else:
            seasonal_period = 12
        
        # Use UnobservedComponents as Kalman Filter implementation
        # This is a state space model with Kalman filtering
        model = UnobservedComponents(
            train_data.values,
            level='local level',
            trend=True,
            seasonal=seasonal_period if len(train_data) >= seasonal_period * 2 else None,
            cycle=False
        )
        
        fitted_model = model.fit(disp=False, maxiter=200)
        return fitted_model
    except Exception as e:
        print(f"Error training Kalman Filter for {branch}: {e}")
        # Try simpler model
        try:
            model = UnobservedComponents(
                train_data.values,
                level='local level',
                trend=True,
                seasonal=None
            )
            fitted_model = model.fit(disp=False, maxiter=200)
            return fitted_model
        except Exception as e2:
            print(f"Error with simpler Kalman Filter for {branch}: {e2}")
            return None


def forecast(model: object, n_periods: int, freq: str = 'W-MON', branch: str = None) -> pd.Series:
    """
    Generate forecast using trained model.
    
    Args:
        model: Trained model
        n_periods: Number of periods to forecast
        freq: Frequency string
        branch: Branch name (not used)
        
    Returns:
        Forecast series with Date index
    """
    if model is None:
        return pd.Series(dtype=float)
    
    try:
        forecast_result = model.forecast(steps=n_periods)
        forecast_values = forecast_result.values
        
        # Create date index
        forecast_dates = pd.date_range(start=pd.Timestamp.now(), periods=n_periods, freq=freq)
        
        return pd.Series(forecast_values, index=forecast_dates)
    except Exception as e:
        print(f"Error forecasting with Kalman Filter: {e}")
        return pd.Series(dtype=float)


def save_model(model: object, path: str) -> None:
    """Save model to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path: str) -> object:
    """Load model from file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

