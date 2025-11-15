"""
Holt-Winters (Exponential Smoothing) model for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Optional
import pickle
from pathlib import Path
import sys
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from training.hyperparameter_optimization import get_hyperparameters


def train_model(
    train_data: pd.Series,
    branch: str,
    freq: str = 'W-MON',
    use_optimization: bool = True,
    n_trials: int = 10,
    anomaly_labels: Optional[pd.Series] = None,
    anomaly_severity: Optional[pd.Series] = None,
    yoy_growth: Optional[pd.Series] = None
) -> Optional[object]:
    """
    Train Holt-Winters model.
    
    Args:
        train_data: Training time series
        branch: Branch name
        freq: Frequency string
        
    Returns:
        Trained model or None if training fails
    """
    if len(train_data) < 20:
        print(f"Warning: Insufficient data for Holt-Winters for {branch}")
        return None
    
    try:
        # Determine seasonal period
        if freq == 'W-MON':
            seasonal_periods = 52  # Weekly data, yearly seasonality
        elif freq == 'MS':
            seasonal_periods = 12  # Monthly data, yearly seasonality
        else:
            seasonal_periods = 12
        
        # Get optimized hyperparameters
        if use_optimization:
            try:
                hyperparams = get_hyperparameters('holt_winters', train_data,
                                                 use_optimization=True,
                                                 n_trials=n_trials,
                                                 timeout=300)
                trend = hyperparams.get('trend', 'add')
                seasonal = hyperparams.get('seasonal', 'add')
            except Exception as e:
                print(f"Warning: Hyperparameter optimization failed for Holt-Winters: {e}")
                trend = 'add'
                seasonal = 'add'
        else:
            trend = 'add'
            seasonal = 'add'
        
        # Ensure we have enough data for seasonality
        if len(train_data) < seasonal_periods * 2:
            # Use additive seasonality with fewer periods
            model = ExponentialSmoothing(
                train_data.values,
                seasonal_periods=min(seasonal_periods, len(train_data) // 2),
                trend=trend,
                seasonal=seasonal if seasonal is not None else 'add'
            )
        else:
            model = ExponentialSmoothing(
                train_data.values,
                seasonal_periods=seasonal_periods,
                trend=trend,
                seasonal=seasonal if seasonal is not None else 'add'
            )
        
        fitted_model = model.fit(optimized=True)
        return fitted_model
    except Exception as e:
        print(f"Error training Holt-Winters for {branch}: {e}")
        # Try without seasonality
        try:
            model = ExponentialSmoothing(
                train_data.values,
                trend='add',
                seasonal=None
            )
            fitted_model = model.fit(optimized=True)
            return fitted_model
        except Exception as e2:
            print(f"Error with simpler Holt-Winters for {branch}: {e2}")
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
        forecast_values = model.forecast(steps=n_periods)
        
        # Create date index
        forecast_dates = pd.date_range(start=pd.Timestamp.now(), periods=n_periods, freq=freq)
        
        return pd.Series(forecast_values, index=forecast_dates)
    except Exception as e:
        print(f"Error forecasting with Holt-Winters: {e}")
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

