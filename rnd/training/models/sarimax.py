"""
SARIMAX model for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Optional
import pickle
from pathlib import Path
import sys
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from training.hyperparameter_optimization import get_hyperparameters


def train_model(train_data: pd.Series, branch: str, freq: str = 'W-MON', 
                use_optimization: bool = True, n_trials: int = 10) -> Optional[object]:
    """
    Train SARIMAX model.
    
    Args:
        train_data: Training time series
        branch: Branch name
        freq: Frequency string
        
    Returns:
        Trained model or None if training fails
    """
    if len(train_data) < 20:
        print(f"Warning: Insufficient data for SARIMAX for {branch}")
        return None
    
    try:
        # Determine seasonal period
        if freq == 'W-MON':
            seasonal_period = 52  # Weekly data, yearly seasonality
        elif freq == 'MS':
            seasonal_period = 12  # Monthly data, yearly seasonality
        else:
            seasonal_period = 12
        
        # Reduce seasonal period if we don't have enough data
        if len(train_data) < seasonal_period * 2:
            seasonal_period = max(4, len(train_data) // 3)
        
        # Get optimized hyperparameters (with reduced trials for speed)
        if use_optimization:
            try:
                # Use fewer trials for SARIMAX (5 instead of 10) and shorter timeout
                hyperparams = get_hyperparameters('sarimax', train_data, 
                                                 use_optimization=True, 
                                                 n_trials=min(5, n_trials),  # Max 5 trials
                                                 timeout=120)  # 2 min timeout (reduced from 5 min)
                p = hyperparams.get('p', 1)
                d = hyperparams.get('d', 1)
                q = hyperparams.get('q', 1)
                P = hyperparams.get('P', 1)
                D = hyperparams.get('D', 1)
                Q = hyperparams.get('Q', 1)
            except Exception as e:
                print(f"Warning: Hyperparameter optimization failed for SARIMAX: {e}")
                # Use defaults
                p, d, q = 1, 1, 1
                P, D, Q = 1, 1, 1
        else:
            # Use default parameters
            p, d, q = 1, 1, 1
            P, D, Q = 1, 1, 1
        
        # Try simpler models first for speed, then fallback to optimized/complex models
        models_to_try = [
            # Start with simplest model (fastest)
            {
                'order': (1, 1, 1),
                'seasonal_order': (0, 0, 0, 0),
                'maxiter': 30,
                'method': 'nm'  # Nelder-Mead is faster
            },
            # Then try simple seasonal
            {
                'order': (1, 1, 1),
                'seasonal_order': (1, 0, 0, seasonal_period),
                'maxiter': 30,
                'method': 'nm'
            },
            # Then try optimized model
            {
                'order': (p, d, q),
                'seasonal_order': (P, D, Q, seasonal_period),
                'maxiter': 50,  # Reduced from 100
                'method': 'nm'  # Use faster method
            },
            # Fallback to optimized without seasonality
            {
                'order': (p, d, q),
                'seasonal_order': (P, 0, 0, seasonal_period),
                'maxiter': 30,
                'method': 'nm'
            }
        ]
        
        for model_config in models_to_try:
            try:
                model = SARIMAX(
                    train_data.values,
                    order=model_config['order'],
                    seasonal_order=model_config['seasonal_order'],
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                # Use specified method (default to 'nm' for speed)
                method = model_config.get('method', 'nm')
                
                fitted_model = model.fit(
                    disp=False,
                    maxiter=model_config['maxiter'],
                    method=method
                )
                return fitted_model
            except Exception as e:
                # Continue to next model configuration
                continue
        
        # If all models failed, return None
        print(f"Error: All SARIMAX model configurations failed for {branch}")
        return None
        
    except Exception as e:
        print(f"Error training SARIMAX for {branch}: {e}")
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
        
        # Handle both pandas Series and numpy array
        if isinstance(forecast_result, pd.Series):
            forecast_values = forecast_result.values
        elif isinstance(forecast_result, np.ndarray):
            forecast_values = forecast_result
        else:
            # Try to convert to array
            forecast_values = np.array(forecast_result)
        
        # Create date index
        # Approximate: use current date as starting point
        forecast_dates = pd.date_range(start=pd.Timestamp.now(), periods=n_periods, freq=freq)
        
        return pd.Series(forecast_values, index=forecast_dates)
    except Exception as e:
        print(f"Error forecasting with SARIMAX: {e}")
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

