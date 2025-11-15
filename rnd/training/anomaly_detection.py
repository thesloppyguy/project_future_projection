"""
Anomaly detection module for time series forecasting.
Uses Isolation Forest with fit_transform pattern and severity scoring.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class AnomalyDetector:
    """
    Anomaly detector with fit_transform pattern.
    Detects anomalies using Isolation Forest and calculates severity scores.
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize anomaly detector.

        Args:
            contamination: Expected proportion of anomalies (0-1)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.isolation_forest = IsolationForest(
            contamination=contamination, random_state=random_state, n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.mean_ = None
        self.std_ = None

    def fit(self, values: np.ndarray) -> "AnomalyDetector":
        """
        Fit the anomaly detector on training data.

        Args:
            values: 1D array of time series values

        Returns:
            Self for method chaining
        """
        if len(values.shape) == 1:
            values_2d = values.reshape(-1, 1)
        else:
            values_2d = values

        # Fit scaler and isolation forest
        self.scaler.fit(values_2d)
        self.isolation_forest.fit(values_2d)

        # Store statistics for z-score calculation
        self.mean_ = np.mean(values)
        self.std_ = np.std(values)
        if self.std_ == 0:
            self.std_ = 1.0  # Avoid division by zero

        self.is_fitted = True
        return self

    def transform(
        self, values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform data to get anomaly labels and severity scores.

        Args:
            values: 1D array of time series values

        Returns:
            Tuple of (labels, severity_scores, isolation_scores)
            - labels: 1 for anomaly, 0 for normal
            - severity_scores: Combined severity score (0-1, where 1 = most severe)
            - isolation_scores: Raw isolation forest scores
        """
        if not self.is_fitted:
            raise ValueError("AnomalyDetector must be fitted before transform")

        if len(values.shape) == 1:
            values_2d = values.reshape(-1, 1)
        else:
            values_2d = values

        # Get isolation forest predictions and scores
        scaled_values = self.scaler.transform(values_2d)
        predictions = self.isolation_forest.predict(scaled_values)
        decision_scores = self.isolation_forest.decision_function(scaled_values)

        # Convert predictions: -1 (anomaly) -> 1, 1 (normal) -> 0
        labels = (predictions == -1).astype(int)

        # Calculate z-score based severity
        z_scores = np.abs((values - self.mean_) / self.std_)
        z_severity = np.clip(z_scores / 3.0, 0, 1)  # Normalize to 0-1 (3 sigma = 1)

        # Normalize isolation forest decision scores to 0-1
        # More negative scores = more anomalous
        min_score = np.min(decision_scores)
        max_score = np.max(decision_scores)
        if max_score - min_score > 0:
            isolation_severity = (decision_scores - min_score) / (max_score - min_score)
            # Invert so that more negative (anomalous) = higher severity
            isolation_severity = 1 - isolation_severity
        else:
            isolation_severity = np.zeros_like(decision_scores)

        # Combine z-score and isolation forest severity (weighted average)
        # Isolation forest gets 60% weight, z-score gets 40%
        combined_severity = 0.6 * isolation_severity + 0.4 * z_severity

        # Only apply severity to detected anomalies
        severity_scores = np.where(labels == 1, combined_severity, 0.0)

        return labels, severity_scores, decision_scores

    def fit_transform(
        self, values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the detector and transform data in one step.

        Args:
            values: 1D array of time series values

        Returns:
            Tuple of (labels, severity_scores, isolation_scores)
        """
        return self.fit(values).transform(values)

    def get_anomaly_statistics(
        self, labels: np.ndarray, severity_scores: np.ndarray
    ) -> Dict:
        """
        Calculate statistics about detected anomalies.

        Args:
            labels: Anomaly labels (0 or 1)
            severity_scores: Severity scores (0-1)

        Returns:
            Dictionary with anomaly statistics
        """
        n_total = len(labels)
        n_anomalies = np.sum(labels)
        anomaly_rate = n_anomalies / n_total if n_total > 0 else 0.0

        anomaly_severities = severity_scores[labels == 1]
        mean_severity = (
            np.mean(anomaly_severities) if len(anomaly_severities) > 0 else 0.0
        )
        max_severity = (
            np.max(anomaly_severities) if len(anomaly_severities) > 0 else 0.0
        )
        min_severity = (
            np.min(anomaly_severities) if len(anomaly_severities) > 0 else 0.0
        )

        return {
            "n_total": n_total,
            "n_anomalies": n_anomalies,
            "anomaly_rate": anomaly_rate,
            "mean_severity": mean_severity,
            "max_severity": max_severity,
            "min_severity": min_severity,
        }

    def save(self, path: str) -> None:
        """Save the fitted detector to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "isolation_forest": self.isolation_forest,
                    "scaler": self.scaler,
                    "mean_": self.mean_,
                    "std_": self.std_,
                    "contamination": self.contamination,
                    "random_state": self.random_state,
                    "is_fitted": self.is_fitted,
                },
                f,
            )

    def load(self, path: str) -> "AnomalyDetector":
        """Load a fitted detector from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.isolation_forest = data["isolation_forest"]
            self.scaler = data["scaler"]
            self.mean_ = data["mean_"]
            self.std_ = data["std_"]
            self.contamination = data["contamination"]
            self.random_state = data["random_state"]
            self.is_fitted = data["is_fitted"]
        return self


def detect_anomalies_in_series(
    series: pd.Series, contamination: float = 0.1, random_state: int = 42
) -> Tuple[pd.Series, pd.Series, Dict]:
    """
    Convenience function to detect anomalies in a pandas Series.

    Args:
        series: Time series with Date index
        contamination: Expected proportion of anomalies
        random_state: Random seed

    Returns:
        Tuple of (labels_series, severity_series, statistics_dict)
    """
    values = series.values
    detector = AnomalyDetector(contamination=contamination, random_state=random_state)
    labels, severity_scores, _ = detector.fit_transform(values)

    labels_series = pd.Series(labels, index=series.index, name="anomaly_label")
    severity_series = pd.Series(
        severity_scores, index=series.index, name="anomaly_severity"
    )

    stats = detector.get_anomaly_statistics(labels, severity_scores)

    return labels_series, severity_series, stats
