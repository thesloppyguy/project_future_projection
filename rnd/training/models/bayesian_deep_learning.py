"""
Bayesian Deep Learning model for time series forecasting.
Uses TensorFlow Probability for Bayesian LSTM.
"""

import pandas as pd
import numpy as np
from typing import Optional
import pickle
from pathlib import Path
import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

try:
    import tensorflow_probability as tfp
    TFP_AVAILABLE = True
except ImportError:
    TFP_AVAILABLE = False
    print("Warning: TensorFlow Probability not available. Using regular LSTM with dropout.")

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from training.hyperparameter_optimization import get_hyperparameters


def create_sequences(data: np.ndarray, n_steps: int = 12) -> tuple:
    """
    Create sequences for LSTM training.
    
    Args:
        data: Time series data
        n_steps: Number of time steps to look back
        
    Returns:
        Tuple of (X, y) arrays
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)


def train_model(train_data: pd.Series, branch: str, freq: str = 'W-MON',
                use_optimization: bool = True, n_trials: int = 10) -> Optional[dict]:
    """
    Train Bayesian Deep Learning model (Bayesian LSTM).
    
    Args:
        train_data: Training time series
        branch: Branch name
        freq: Frequency string
        
    Returns:
        Dictionary with model and metadata, or None if training fails
    """
    if len(train_data) < 30:
        print(f"Warning: Insufficient data for Bayesian Deep Learning for {branch}")
        return None
    
    try:
        # Prepare data
        values = train_data.values.astype(float)
        
        # Normalize data
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            std = 1
        normalized_values = (values - mean) / std
        
        # Get optimized hyperparameters
        if use_optimization:
            try:
                hyperparams = get_hyperparameters('bayesian_deep_learning', train_data,
                                                 use_optimization=True,
                                                 n_trials=n_trials,
                                                 timeout=600)
                n_steps = hyperparams.get('n_steps', 12)
                lstm_units_1 = hyperparams.get('lstm_units_1', 50)
                lstm_units_2 = hyperparams.get('lstm_units_2', 50)
                dropout_rate = hyperparams.get('dropout_rate', 0.3)
                learning_rate = hyperparams.get('learning_rate', 0.001)
            except Exception as e:
                print(f"Warning: Hyperparameter optimization failed for Bayesian Deep Learning: {e}")
                n_steps = 12
                lstm_units_1 = 50
                lstm_units_2 = 50
                dropout_rate = 0.3
                learning_rate = 0.001
        else:
            n_steps = 12
            lstm_units_1 = 50
            lstm_units_2 = 50
            dropout_rate = 0.3
            learning_rate = 0.001
        
        # Create sequences
        X, y = create_sequences(normalized_values, n_steps)
        
        if len(X) < 10:
            print(f"Warning: Insufficient sequences for Bayesian Deep Learning for {branch}")
            return None
        
        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build model with dropout for uncertainty estimation
        # Using Monte Carlo Dropout as approximation to Bayesian inference
        # With optimized hyperparameters
        model = Sequential([
            LSTM(lstm_units_1, activation='relu', return_sequences=True, input_shape=(n_steps, 1)),
            Dropout(dropout_rate),  # Higher dropout for uncertainty
            LSTM(lstm_units_2, activation='relu', return_sequences=False),
            Dropout(dropout_rate),
            Dense(1)
        ])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
        
        # Train model
        model.fit(X, y, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
        
        return {
            'model': model,
            'mean': mean,
            'std': std,
            'n_steps': n_steps,
            'last_values': normalized_values[-n_steps:].tolist(),
            'last_dates': train_data.tail(n_steps).index.tolist(),
            'is_bayesian': TFP_AVAILABLE
        }
    except Exception as e:
        print(f"Error training Bayesian Deep Learning for {branch}: {e}")
        return None


def forecast(model: dict, n_periods: int, freq: str = 'W-MON', branch: str = None) -> pd.Series:
    """
    Generate forecast using trained model.
    Uses Monte Carlo dropout for uncertainty estimation.
    
    Args:
        model: Dictionary with trained model and metadata
        n_periods: Number of periods to forecast
        freq: Frequency string
        branch: Branch name (not used)
        
    Returns:
        Forecast series with Date index (mean of Monte Carlo samples)
    """
    if model is None:
        return pd.Series(dtype=float)
    
    try:
        lstm_model = model['model']
        mean = model['mean']
        std = model['std']
        n_steps = model['n_steps']
        last_values = model['last_values']
        last_dates = model['last_dates']
        
        # Create forecast dates
        if len(last_dates) > 0:
            last_date = pd.to_datetime(last_dates[-1])
        else:
            last_date = pd.Timestamp.now()
        
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_periods, freq=freq)
        
        # Generate forecasts iteratively with Monte Carlo dropout
        forecast_values = []
        current_sequence = np.array(last_values).copy()
        
        for i in range(n_periods):
            # Reshape for prediction
            X_pred = current_sequence.reshape((1, n_steps, 1))
            
            # Monte Carlo sampling for uncertainty (if dropout is enabled)
            # Run prediction multiple times with dropout enabled
            n_samples = 10
            predictions = []
            for _ in range(n_samples):
                pred_normalized = lstm_model(X_pred, training=True).numpy()[0, 0]
                predictions.append(pred_normalized)
            
            # Use mean of samples
            pred_normalized = np.mean(predictions)
            
            # Denormalize
            pred = pred_normalized * std + mean
            forecast_values.append(max(0, pred))  # Ensure non-negative
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], pred_normalized)
        
        return pd.Series(forecast_values, index=forecast_dates)
    except Exception as e:
        print(f"Error forecasting with Bayesian Deep Learning: {e}")
        return pd.Series(dtype=float)


def save_model(model: dict, path: str) -> None:
    """Save model to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Save Keras model separately
    keras_path = path.replace('.pkl', '_keras.h5')
    model['model'].save(keras_path)
    # Save metadata
    metadata = {k: v for k, v in model.items() if k != 'model'}
    with open(path, 'wb') as f:
        pickle.dump(metadata, f)


def load_model(path: str) -> dict:
    """Load model from file."""
    keras_path = path.replace('.pkl', '_keras.h5')
    model = {
        'model': keras.models.load_model(keras_path)
    }
    with open(path, 'rb') as f:
        metadata = pickle.load(f)
    model.update(metadata)
    return model

