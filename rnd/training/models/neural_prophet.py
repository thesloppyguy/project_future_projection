"""
Neural Prophet model for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Optional
import pickle
from pathlib import Path
import sys
import os

# Try to import NeuralProphet, but handle version compatibility issues
try:
    from neuralprophet import NeuralProphet
    NEURALPROPHET_AVAILABLE = True
except (ImportError, TypeError) as e:
    NEURALPROPHET_AVAILABLE = False
    print(f"Warning: NeuralProphet not available due to: {e}")
    print("This may be due to holidays package version incompatibility.")
    print("You may need to downgrade holidays: uv pip install 'holidays<0.40'")

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
    Train Neural Prophet model.
    
    Args:
        train_data: Training time series
        branch: Branch name
        freq: Frequency string
        
    Returns:
        Trained model or None if training fails
    """
    if not NEURALPROPHET_AVAILABLE:
        print(f"Warning: NeuralProphet is not available for {branch}")
        return None
    
    if len(train_data) < 10:
        print(f"Warning: Insufficient data for Neural Prophet for {branch}")
        return None
    
    try:
        # Prepare data for Neural Prophet (requires 'ds' and 'y' columns)
        df = pd.DataFrame({
            'ds': train_data.index,
            'y': train_data.values
        })
        
        # Get optimized hyperparameters
        if use_optimization:
            try:
                hyperparams = get_hyperparameters('neural_prophet', train_data,
                                                 use_optimization=True,
                                                 n_trials=n_trials,
                                                 timeout=300)
                yearly_seasonality = hyperparams.get('yearly_seasonality', True)
                weekly_seasonality = hyperparams.get('weekly_seasonality', (freq == 'W-MON'))
                n_lags = hyperparams.get('n_lags', 10)
                n_forecasts = hyperparams.get('n_forecasts', 52)
            except Exception as e:
                print(f"Warning: Hyperparameter optimization failed for Neural Prophet: {e}")
                yearly_seasonality = True
                weekly_seasonality = (freq == 'W-MON')
                n_lags = 10
                n_forecasts = 52
        else:
            yearly_seasonality = True
            weekly_seasonality = (freq == 'W-MON')
            n_lags = 10
            n_forecasts = 52
        
        # Create Neural Prophet model with optimized hyperparameters
        model = NeuralProphet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=False,
            n_lags=n_lags,
            n_forecasts=n_forecasts
        )
        
        model.fit(df)
        return model
    except Exception as e:
        print(f"Error training Neural Prophet for {branch}: {e}")
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
        future = model.make_future_dataframe(periods=n_periods)
        forecast_df = model.predict(future)
        
        # Extract only the forecasted periods
        forecast_values = forecast_df['yhat1'].tail(n_periods).values
        forecast_dates = forecast_df['ds'].tail(n_periods).values
        
        return pd.Series(forecast_values, index=pd.to_datetime(forecast_dates))
    except Exception as e:
        print(f"Error forecasting with Neural Prophet: {e}")
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

