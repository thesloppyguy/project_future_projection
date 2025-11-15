"""
Feature engineering module for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features for time series forecasting."""

    def __init__(
        self,
        lag_periods: List[int] = [1, 2, 3, 6, 12],
        rolling_windows: List[int] = [3, 6, 12],
        include_seasonality: bool = True,
        include_trend: bool = True,
    ):
        """
        Initialize feature engineer.

        Args:
            lag_periods: List of lag periods (in months/weeks)
            rolling_windows: List of rolling window sizes
            include_seasonality: Whether to include seasonal features
            include_trend: Whether to include trend features
        """
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows
        self.include_seasonality = include_seasonality
        self.include_trend = include_trend

    def create_features(self, series: pd.Series) -> pd.DataFrame:
        """
        Create features from time series.

        Args:
            series: Time series with datetime index

        Returns:
            DataFrame with features and target
        """
        df = pd.DataFrame({"date": series.index, "value": series.values})
        df = df.sort_values("date").reset_index(drop=True)

        # Lag features
        for lag in self.lag_periods:
            if lag <= len(df):
                df[f"lag_{lag}"] = df["value"].shift(lag)

        # Rolling statistics
        for window in self.rolling_windows:
            if window <= len(df):
                df[f"rolling_mean_{window}"] = (
                    df["value"].rolling(window=window, min_periods=1).mean()
                )
                df[f"rolling_std_{window}"] = (
                    df["value"].rolling(window=window, min_periods=1).std()
                )
                df[f"rolling_min_{window}"] = (
                    df["value"].rolling(window=window, min_periods=1).min()
                )
                df[f"rolling_max_{window}"] = (
                    df["value"].rolling(window=window, min_periods=1).max()
                )

        # Date features
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df["month"] = pd.to_datetime(df["date"]).dt.month
        df["quarter"] = pd.to_datetime(df["date"]).dt.quarter
        df["day_of_year"] = pd.to_datetime(df["date"]).dt.dayofyear

        # Seasonal features
        if self.include_seasonality:
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
            df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
            df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)

        # Trend feature
        if self.include_trend:
            df["trend"] = np.arange(len(df))

        # Year-over-year growth (if enough data)
        if len(df) > 12:
            df["yoy_growth"] = df["value"].pct_change(periods=12)

        # Fill NaN values
        df = df.bfill().fillna(0)

        return df

    def prepare_ml_features(self, series: pd.Series) -> tuple:
        """
        Prepare features for ML models (X, y format).

        Args:
            series: Time series with datetime index

        Returns:
            Tuple of (X, y, feature_names) where X is feature matrix, y is target
        """
        df = self.create_features(series)

        # Separate target
        y = df["value"].values

        # Get feature columns (exclude date and value)
        feature_cols = [col for col in df.columns if col not in ["date", "value"]]
        X = df[feature_cols].values

        return X, y, feature_cols

    def create_features_for_forecast(
        self, historical_values: list, future_date: pd.Timestamp, historical_length: int
    ) -> np.ndarray:
        """
        Create feature vector for a single future date.

        Args:
            historical_values: List of historical values (for lag and rolling features)
            future_date: Date to create features for
            historical_length: Total length of historical series (for trend)

        Returns:
            Feature vector as numpy array
        """
        features = {}

        # Lag features
        for lag in self.lag_periods:
            if lag <= len(historical_values):
                features[f"lag_{lag}"] = historical_values[-lag]
            else:
                features[f"lag_{lag}"] = (
                    historical_values[0] if len(historical_values) > 0 else 0
                )

        # Rolling statistics
        values_array = np.array(historical_values)
        for window in self.rolling_windows:
            if len(values_array) >= window:
                features[f"rolling_mean_{window}"] = np.mean(values_array[-window:])
                features[f"rolling_std_{window}"] = np.std(values_array[-window:])
                features[f"rolling_min_{window}"] = np.min(values_array[-window:])
                features[f"rolling_max_{window}"] = np.max(values_array[-window:])
            else:
                if len(values_array) > 0:
                    features[f"rolling_mean_{window}"] = np.mean(values_array)
                    features[f"rolling_std_{window}"] = np.std(values_array)
                    features[f"rolling_min_{window}"] = np.min(values_array)
                    features[f"rolling_max_{window}"] = np.max(values_array)
                else:
                    features[f"rolling_mean_{window}"] = 0
                    features[f"rolling_std_{window}"] = 0
                    features[f"rolling_min_{window}"] = 0
                    features[f"rolling_max_{window}"] = 0

        # Date features
        features["year"] = future_date.year
        features["month"] = future_date.month
        features["quarter"] = future_date.quarter
        features["day_of_year"] = future_date.dayofyear

        # Seasonal features
        if self.include_seasonality:
            features["month_sin"] = np.sin(2 * np.pi * future_date.month / 12)
            features["month_cos"] = np.cos(2 * np.pi * future_date.month / 12)
            features["quarter_sin"] = np.sin(2 * np.pi * future_date.quarter / 4)
            features["quarter_cos"] = np.cos(2 * np.pi * future_date.quarter / 4)

        # Trend feature
        if self.include_trend:
            features["trend"] = historical_length

        # Year-over-year growth (simplified - use last 12 months if available)
        if len(historical_values) >= 12:
            current_avg = np.mean(historical_values[-12:])
            prev_avg = (
                np.mean(historical_values[-24:-12])
                if len(historical_values) >= 24
                else current_avg
            )
            if prev_avg != 0:
                features["yoy_growth"] = (current_avg - prev_avg) / prev_avg
            else:
                features["yoy_growth"] = 0
        else:
            features["yoy_growth"] = 0

        return features
