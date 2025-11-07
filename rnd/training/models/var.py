"""
Vector Autoregression (VAR) model for time series forecasting.
Forecasts all branches simultaneously.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import pickle
from pathlib import Path
from statsmodels.tsa.vector_ar.var_model import VAR


def train_model(train_data: Dict[str, pd.Series], branches: list, freq: str = 'W-MON',
                use_optimization: bool = True, n_trials: int = 20) -> Optional[object]:
    """
    Train VAR model on all branches simultaneously.
    
    Args:
        train_data: Dictionary of training time series per branch
        branches: List of branch names
        freq: Frequency string
        
    Returns:
        Trained VAR model or None if training fails
    """
    # Check if we have data for all branches
    available_branches = [b for b in branches if b in train_data and len(train_data[b]) > 0]
    
    if len(available_branches) < 2:
        print(f"Warning: Need at least 2 branches for VAR, got {len(available_branches)}")
        return None
    
    try:
        # Align all series to common dates
        all_dates = set()
        for branch in available_branches:
            all_dates.update(train_data[branch].index)
        
        common_dates = sorted(all_dates)
        
        if len(common_dates) < 20:
            print(f"Warning: Insufficient common dates for VAR")
            return None
        
        # Create DataFrame with all branches
        var_df = pd.DataFrame(index=common_dates)
        for branch in available_branches:
            var_df[branch] = train_data[branch].reindex(common_dates, fill_value=0)
        
        # Fill any remaining NaN values
        var_df = var_df.fillna(0)
        
        # Fit VAR model
        model = VAR(var_df)
        fitted_model = model.fit(maxlags=4, ic='aic')
        
        return {
            'model': fitted_model,
            'branches': available_branches,
            'last_dates': common_dates[-1]
        }
    except Exception as e:
        print(f"Error training VAR: {e}")
        return None


def forecast(model: dict, n_periods: int, freq: str = 'W-MON', branch: str = None) -> pd.Series:
    """
    Generate forecast using trained VAR model.
    
    Args:
        model: Dictionary with trained VAR model
        n_periods: Number of periods to forecast
        freq: Frequency string
        branch: Branch name to extract forecast for
        
    Returns:
        Forecast series with Date index for specified branch
    """
    if model is None or branch is None:
        return pd.Series(dtype=float)
    
    try:
        var_model = model['model']
        branches = model['branches']
        last_date = model['last_dates']
        
        if branch not in branches:
            print(f"Warning: Branch {branch} not in VAR model")
            return pd.Series(dtype=float)
        
        # Forecast all branches
        forecast_result = var_model.forecast(var_model.y, steps=n_periods)
        
        # Create DataFrame from forecast
        forecast_df = pd.DataFrame(forecast_result, columns=branches)
        
        # Extract forecast for requested branch
        forecast_values = forecast_df[branch].values
        
        # Create date index
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_periods, freq=freq)
        
        return pd.Series(forecast_values, index=forecast_dates)
    except Exception as e:
        print(f"Error forecasting with VAR for {branch}: {e}")
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

