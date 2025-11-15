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
    Train Prophet model with YoY growth component.
    
    Args:
        train_data: Training time series
        branch: Branch name
        freq: Frequency string
        anomaly_labels: Anomaly labels (optional, for future use)
        anomaly_severity: Anomaly severity scores (optional, for future use)
        yoy_growth: Year-over-year growth rates
        
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
        
        # Add YoY growth as regressor if provided
        if yoy_growth is not None and len(yoy_growth) > 0:
            yoy_aligned = yoy_growth.reindex(df['ds'], fill_value=0)
            df['yoy_growth'] = yoy_aligned.values
        
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
        
        # Add YoY growth regressor if available
        if 'yoy_growth' in df.columns:
            model.add_regressor('yoy_growth')
        
        model.fit(df)
        
        # Store metadata for forecasting
        model.yoy_growth_available = 'yoy_growth' in df.columns
        if model.yoy_growth_available:
            # Store last YoY growth value for forecasting
            model.last_yoy_growth = df['yoy_growth'].iloc[-1] if len(df) > 0 else 0.0
        
        return model
    except Exception as e:
        print(f"Error training Prophet for {branch}: {e}")
        return None


def forecast(
    model: object,
    n_periods: int,
    freq: str = 'W-MON',
    branch: str = None,
    yoy_growth: Optional[pd.Series] = None
) -> pd.Series:
    """
    Generate forecast using trained model.
    
    Args:
        model: Trained model
        n_periods: Number of periods to forecast
        freq: Frequency string
        branch: Branch name (not used)
        yoy_growth: YoY growth rates for forecast period (optional)
        
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
        
        # Add YoY growth regressor if model was trained with it
        if hasattr(model, 'yoy_growth_available') and model.yoy_growth_available:
            if yoy_growth is not None and len(yoy_growth) > 0:
                # Use provided YoY growth
                yoy_aligned = yoy_growth.reindex(future['ds'], method='ffill', fill_value=0)
                future['yoy_growth'] = yoy_aligned.values
            else:
                # Use last known YoY growth value (simple extrapolation)
                last_yoy = getattr(model, 'last_yoy_growth', 0.0)
                future['yoy_growth'] = last_yoy
        
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

