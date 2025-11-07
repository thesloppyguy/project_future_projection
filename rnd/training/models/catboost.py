"""
CatBoost model for time series forecasting with feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Optional
import pickle
from pathlib import Path
import sys
import os
from catboost import CatBoostRegressor

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from training.hyperparameter_optimization import get_hyperparameters


def create_features(series: pd.Series, n_lags: int = 12) -> pd.DataFrame:
    """
    Create features from time series.
    
    Args:
        series: Time series data
        n_lags: Number of lag features to create
        
    Returns:
        DataFrame with features
    """
    df = pd.DataFrame({'y': series.values, 'date': series.index})
    df['date'] = pd.to_datetime(df['date'])
    
    # Lag features
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    
    # Rolling statistics
    for window in [4, 8, 12]:
        df[f'rolling_mean_{window}'] = df['y'].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['y'].shift(1).rolling(window=window, min_periods=1).std()
    
    # Date features
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    
    # Fill NaN values
    df = df.fillna(0)
    
    return df


def train_model(train_data: pd.Series, branch: str, freq: str = 'W-MON',
                use_optimization: bool = True, n_trials: int = 20) -> Optional[dict]:
    """
    Train CatBoost model.
    
    Args:
        train_data: Training time series
        branch: Branch name
        freq: Frequency string
        
    Returns:
        Dictionary with model and feature names, or None if training fails
    """
    if len(train_data) < 20:
        print(f"Warning: Insufficient data for CatBoost for {branch}")
        return None
    
    try:
        # Create features
        feature_df = create_features(train_data)
        
        # Prepare training data
        feature_cols = [col for col in feature_df.columns if col not in ['y', 'date']]
        X = feature_df[feature_cols].values
        y = feature_df['y'].values
        
        # Remove rows with NaN in target
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 10:
            print(f"Warning: Insufficient valid data for CatBoost for {branch}")
            return None
        
        # Get optimized hyperparameters
        if use_optimization:
            try:
                hyperparams = get_hyperparameters('catboost', train_data,
                                                 use_optimization=True,
                                                 n_trials=n_trials,
                                                 timeout=600)
                model_params = {
                    'iterations': hyperparams.get('iterations', 100),
                    'depth': hyperparams.get('depth', 6),
                    'learning_rate': hyperparams.get('learning_rate', 0.1),
                    'l2_leaf_reg': hyperparams.get('l2_leaf_reg', 3),
                    'random_seed': 42,
                    'verbose': False
                }
            except Exception as e:
                print(f"Warning: Hyperparameter optimization failed for CatBoost: {e}")
                model_params = {
                    'iterations': 100,
                    'depth': 6,
                    'learning_rate': 0.1,
                    'random_seed': 42,
                    'verbose': False
                }
        else:
            model_params = {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'random_seed': 42,
                'verbose': False
            }
        
        # Train CatBoost model with optimized hyperparameters
        model = CatBoostRegressor(**model_params)
        model.fit(X, y)
        
        return {
            'model': model,
            'feature_names': feature_cols,
            'last_values': train_data.tail(12).values.tolist(),
            'last_dates': train_data.tail(12).index.tolist()
        }
    except Exception as e:
        print(f"Error training CatBoost for {branch}: {e}")
        return None


def forecast(model: dict, n_periods: int, freq: str = 'W-MON', branch: str = None) -> pd.Series:
    """
    Generate forecast using trained model.
    
    Args:
        model: Dictionary with trained model and metadata
        n_periods: Number of periods to forecast
        freq: Frequency string
        branch: Branch name (not used)
        
    Returns:
        Forecast series with Date index
    """
    if model is None:
        return pd.Series(dtype=float)
    
    try:
        cat_model = model['model']
        feature_names = model['feature_names']
        last_values = model['last_values']
        last_dates = model['last_dates']
        
        # Create forecast dates
        if len(last_dates) > 0:
            last_date = pd.to_datetime(last_dates[-1])
        else:
            last_date = pd.Timestamp.now()
        
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_periods, freq=freq)
        
        # Generate forecasts iteratively
        forecast_values = []
        current_values = last_values.copy()
        
        for i in range(n_periods):
            # Create feature vector
            features = {}
            
            # Lag features
            for lag in range(1, 13):
                if lag <= len(current_values):
                    features[f'lag_{lag}'] = current_values[-lag]
                else:
                    features[f'lag_{lag}'] = 0
            
            # Rolling statistics
            values_array = np.array(current_values)
            for window in [4, 8, 12]:
                if len(values_array) >= window:
                    features[f'rolling_mean_{window}'] = np.mean(values_array[-window:])
                    features[f'rolling_std_{window}'] = np.std(values_array[-window:])
                else:
                    features[f'rolling_mean_{window}'] = np.mean(values_array) if len(values_array) > 0 else 0
                    features[f'rolling_std_{window}'] = 0
            
            # Date features
            forecast_date = forecast_dates[i]
            features['week_of_year'] = forecast_date.isocalendar().week
            features['month'] = forecast_date.month
            features['year'] = forecast_date.year
            features['quarter'] = forecast_date.quarter
            
            # Create feature vector in correct order
            feature_vector = np.array([features.get(name, 0) for name in feature_names]).reshape(1, -1)
            
            # Predict
            pred = cat_model.predict(feature_vector)[0]
            forecast_values.append(max(0, pred))  # Ensure non-negative
            
            # Update current values
            current_values.append(pred)
            if len(current_values) > 12:
                current_values.pop(0)
        
        return pd.Series(forecast_values, index=forecast_dates)
    except Exception as e:
        print(f"Error forecasting with CatBoost: {e}")
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

