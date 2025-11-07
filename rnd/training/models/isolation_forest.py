"""
Isolation Forest model for anomaly detection and time series forecasting.
Uses Isolation Forest to detect anomalies, then forecasts on cleaned data.
"""

import pandas as pd
import numpy as np
from typing import Optional
import pickle
from pathlib import Path
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA


def train_model(train_data: pd.Series, branch: str, freq: str = 'W-MON',
                use_optimization: bool = True, n_trials: int = 20) -> Optional[dict]:
    """
    Train Isolation Forest model for anomaly detection, then fit ARIMA on cleaned data.
    
    Args:
        train_data: Training time series
        branch: Branch name
        freq: Frequency string
        
    Returns:
        Dictionary with Isolation Forest and ARIMA models, or None if training fails
    """
    if len(train_data) < 20:
        print(f"Warning: Insufficient data for Isolation Forest for {branch}")
        return None
    
    try:
        values = train_data.values.astype(float)
        
        # Train Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        # Reshape for sklearn
        values_2d = values.reshape(-1, 1)
        anomalies = iso_forest.fit_predict(values_2d)
        
        # Remove anomalies (keep only inliers: anomalies == 1)
        cleaned_values = values[anomalies == 1]
        cleaned_dates = train_data.index[anomalies == 1]
        
        if len(cleaned_values) < 10:
            print(f"Warning: Too many anomalies detected for {branch}, using original data")
            cleaned_values = values
            cleaned_dates = train_data.index
        
        # Fit ARIMA on cleaned data
        try:
            arima_model = ARIMA(cleaned_values, order=(1, 1, 1)).fit()
        except:
            # Try simpler model
            try:
                arima_model = ARIMA(cleaned_values, order=(1, 0, 0)).fit()
            except:
                print(f"Warning: Could not fit ARIMA for {branch}")
                return None
        
        return {
            'isolation_forest': iso_forest,
            'arima_model': arima_model,
            'original_values': values,
            'cleaned_values': cleaned_values,
            'cleaned_dates': cleaned_dates.tolist()
        }
    except Exception as e:
        print(f"Error training Isolation Forest for {branch}: {e}")
        return None


def forecast(model: dict, n_periods: int, freq: str = 'W-MON', branch: str = None) -> pd.Series:
    """
    Generate forecast using trained model.
    
    Args:
        model: Dictionary with trained models
        n_periods: Number of periods to forecast
        freq: Frequency string
        branch: Branch name (not used)
        
    Returns:
        Forecast series with Date index
    """
    if model is None:
        return pd.Series(dtype=float)
    
    try:
        arima_model = model['arima_model']
        
        # Forecast using ARIMA
        forecast_values = arima_model.forecast(steps=n_periods)
        
        # Create date index
        forecast_dates = pd.date_range(start=pd.Timestamp.now(), periods=n_periods, freq=freq)
        
        return pd.Series(forecast_values, index=forecast_dates)
    except Exception as e:
        print(f"Error forecasting with Isolation Forest: {e}")
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

