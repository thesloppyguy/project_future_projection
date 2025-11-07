"""
Prophet model for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Optional
import pickle
from pathlib import Path
import sys
import os

# Import prophet package (local prophet.py file should be renamed to avoid conflict)
try:
    from prophet import Prophet
except ImportError:
    # Fallback: try importing from fbprophet (older package name)
    try:
        from fbprophet import Prophet
    except ImportError:
        raise ImportError("prophet package not found. Please install it with: pip install prophet")

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from training.hyperparameter_optimization import get_hyperparameters


def train_model(train_data: pd.Series, branch: str, freq: str = 'W-MON',
                use_optimization: bool = True, n_trials: int = 10) -> Optional[object]:
    """
    Train Prophet model.
    
    Args:
        train_data: Training time series
        branch: Branch name
        freq: Frequency string
        
    Returns:
        Trained model or None if training fails
    """
    if len(train_data) < 10:
        print(f"Warning: Insufficient data for Prophet for {branch}")
        return None
    
    try:
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        df = pd.DataFrame({
            'ds': train_data.index,
            'y': train_data.values
        })
        
        # Get optimized hyperparameters
        if use_optimization:
            try:
                hyperparams = get_hyperparameters('prophet', train_data,
                                                 use_optimization=True,
                                                 n_trials=n_trials,
                                                 timeout=300)
                yearly_seasonality = hyperparams.get('yearly_seasonality', True)
                weekly_seasonality = hyperparams.get('weekly_seasonality', (freq == 'W-MON'))
                seasonality_mode = hyperparams.get('seasonality_mode', 'additive')
                changepoint_prior_scale = hyperparams.get('changepoint_prior_scale', 0.05)
                seasonality_prior_scale = hyperparams.get('seasonality_prior_scale', 10.0)
            except Exception as e:
                print(f"Warning: Hyperparameter optimization failed for Prophet: {e}")
                yearly_seasonality = True
                weekly_seasonality = (freq == 'W-MON')
                seasonality_mode = 'additive'
                changepoint_prior_scale = 0.05
                seasonality_prior_scale = 10.0
        else:
            yearly_seasonality = True
            weekly_seasonality = (freq == 'W-MON')
            seasonality_mode = 'additive'
            changepoint_prior_scale = 0.05
            seasonality_prior_scale = 10.0
        
        # Create Prophet model with optimized hyperparameters
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=False,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale
        )
        
        model.fit(df)
        return model
    except Exception as e:
        print(f"Error training Prophet for {branch}: {e}")
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
        # Create future dataframe
        if freq == 'W-MON':
            period = 'W'
        elif freq == 'MS':
            period = 'M'
        else:
            period = 'W'
        
        future = model.make_future_dataframe(periods=n_periods, freq=period)
        forecast_df = model.predict(future)
        
        # Extract only the forecasted periods
        forecast_values = forecast_df['yhat'].tail(n_periods).values
        forecast_dates = forecast_df['ds'].tail(n_periods).values
        
        return pd.Series(forecast_values, index=pd.to_datetime(forecast_dates))
    except Exception as e:
        print(f"Error forecasting with Prophet: {e}")
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

