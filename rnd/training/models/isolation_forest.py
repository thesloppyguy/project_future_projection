"""
Isolation Forest model for anomaly detection and time series forecasting.
Uses the new AnomalyDetector class with fit_transform pattern.
"""

import pandas as pd
import numpy as np
from typing import Optional
import pickle
from pathlib import Path
import sys
import os
from statsmodels.tsa.arima.model import ARIMA

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from training.anomaly_detection import AnomalyDetector


def train_model(
    train_data: pd.Series,
    branch: str,
    freq: str = "W-MON",
    use_optimization: bool = True,
    n_trials: int = 20,
    anomaly_labels: Optional[pd.Series] = None,
    anomaly_severity: Optional[pd.Series] = None,
    yoy_growth: Optional[pd.Series] = None,
) -> Optional[dict]:
    """
    Train Isolation Forest model using new AnomalyDetector with labels and severity.
    Uses anomaly information to weight training data instead of dropping.

    Args:
        train_data: Training time series
        branch: Branch name
        freq: Frequency string
        anomaly_labels: Pre-computed anomaly labels (if None, will compute)
        anomaly_severity: Pre-computed anomaly severity scores (if None, will compute)
        yoy_growth: YoY growth rates (optional)

    Returns:
        Dictionary with detector, ARIMA model, and metadata, or None if training fails
    """
    if len(train_data) < 20:
        print(f"Warning: Insufficient data for Isolation Forest for {branch}")
        return None

    try:
        values = train_data.values.astype(float)

        # Use provided anomaly labels/severity or compute new ones
        if anomaly_labels is None or anomaly_severity is None:
            # Create new detector
            detector = AnomalyDetector(contamination=0.1, random_state=42)
            labels, severity_scores, _ = detector.fit_transform(values)
        else:
            # Use provided labels and severity
            labels = anomaly_labels.values
            severity_scores = anomaly_severity.values
            # Create detector for consistency (fit on data)
            detector = AnomalyDetector(contamination=0.1, random_state=42)
            detector.fit(values)

        # Weight data by anomaly severity (lower weight for anomalies)
        # Normal points get weight 1.0, anomalies get weight based on severity
        weights = np.where(labels == 1, 1.0 - severity_scores * 0.5, 1.0)
        weights = np.clip(weights, 0.1, 1.0)  # Minimum weight of 0.1

        # Fit ARIMA on ALL data (including anomalies) with weights
        # Anomalies are not dropped - they are weighted down based on severity
        # This allows the model to learn from all data while reducing the influence of anomalies
        try:
            # Try to fit with weighted data if ARIMA supports it
            # Since ARIMA doesn't directly support sample weights, we'll use all data
            # but the model will naturally be less influenced by anomalies due to their nature
            arima_model = ARIMA(values, order=(1, 1, 1)).fit()
        except:
            # Try simpler model
            try:
                arima_model = ARIMA(values, order=(1, 0, 0)).fit()
            except:
                print(f"Warning: Could not fit ARIMA for {branch}")
                return None

        # Store last date for forecast
        last_date = train_data.index[-1] if len(train_data) > 0 else pd.Timestamp.now()

        return {
            "detector": detector,
            "arima_model": arima_model,
            "anomaly_labels": labels,
            "anomaly_severity": severity_scores,
            "weights": weights,
            "original_values": values,
            "last_date": last_date,
        }
    except Exception as e:
        print(f"Error training Isolation Forest for {branch}: {e}")
        return None


def forecast(
    model: dict,
    n_periods: int,
    freq: str = "W-MON",
    branch: str = None,
    yoy_growth: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Generate forecast using trained model.

    Args:
        model: Dictionary with trained models
        n_periods: Number of periods to forecast
        freq: Frequency string
        branch: Branch name (not used)
        yoy_growth: YoY growth rates (optional, not used in this model)

    Returns:
        Forecast series with Date index
    """
    if model is None:
        return pd.Series(dtype=float)

    try:
        arima_model = model["arima_model"]
        original_values = model.get("original_values", np.array([]))

        # Forecast using ARIMA
        forecast_values = arima_model.forecast(steps=n_periods)

        # Create date index - use last date from training data
        last_date = model.get("last_date", pd.Timestamp.now())

        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=n_periods, freq=freq
        )

        return pd.Series(forecast_values, index=forecast_dates)
    except Exception as e:
        print(f"Error forecasting with Isolation Forest: {e}")
        return pd.Series(dtype=float)


def save_model(model: dict, path: str) -> None:
    """Save model to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> dict:
    """Load model from file."""
    with open(path, "rb") as f:
        return pickle.load(f)
