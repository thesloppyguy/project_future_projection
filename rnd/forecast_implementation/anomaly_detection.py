"""
Anomaly detection module for time series forecasting.
Uses rolling statistics and z-scores for time-aware anomaly detection.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Time-aware anomaly detector using rolling statistics.
    Detects anomalies based on deviations from rolling mean/std, accounting for trends.
    """

    def __init__(
        self,
        window: int = 12,
        z_threshold: float = 2.5,
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize anomaly detector.

        Args:
            window: Rolling window size for calculating mean/std
            z_threshold: Z-score threshold for anomaly detection (default 2.5 = ~1% outliers)
            contamination: Expected proportion of anomalies (used for adaptive threshold)
            random_state: Random seed for reproducibility
        """
        self.window = window
        self.z_threshold = z_threshold
        self.contamination = contamination
        self.random_state = random_state
        self.is_fitted = False
        self.mean_ = None
        self.std_ = None

    def fit(self, values: np.ndarray) -> "AnomalyDetector":
        """
        Fit the anomaly detector on training data.
        For time series, we just store statistics.

        Args:
            values: 1D array of time series values

        Returns:
            Self for method chaining
        """
        # Store statistics for reference
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
        Transform data to get anomaly labels and severity scores using rolling statistics.

        Args:
            values: 1D array of time series values

        Returns:
            Tuple of (labels, severity_scores, z_scores)
            - labels: 1 for anomaly, 0 for normal
            - severity_scores: Severity score (0-1, where 1 = most severe)
            - z_scores: Z-scores based on rolling statistics
        """
        if not self.is_fitted:
            raise ValueError("AnomalyDetector must be fitted before transform")

        if len(values) < self.window:
            # If not enough data, use simple z-score
            z_scores = np.abs((values - self.mean_) / self.std_)
            labels = (z_scores > self.z_threshold).astype(int)
            severity_scores = np.clip(z_scores / (self.z_threshold * 2), 0, 1)
            severity_scores = np.where(labels == 1, severity_scores, 0.0)
            return labels, severity_scores, z_scores

        # Calculate rolling statistics
        values_series = pd.Series(values)
        rolling_mean = values_series.rolling(
            window=self.window, min_periods=1, center=False
        ).mean()
        rolling_std = values_series.rolling(
            window=self.window, min_periods=1, center=False
        ).std()

        # Handle zero std (use global std as fallback)
        rolling_std = rolling_std.replace(0, self.std_)
        rolling_std = rolling_std.fillna(self.std_)

        # Calculate z-scores based on rolling statistics
        z_scores = np.abs((values - rolling_mean.values) / (rolling_std.values + 1e-6))

        # Detect anomalies: points that deviate significantly from rolling mean
        labels = (z_scores > self.z_threshold).astype(int)

        # Calculate severity: how many standard deviations away
        # Normalize to 0-1 scale where z_threshold = 0.5 severity
        severity_scores = np.clip(z_scores / (self.z_threshold * 2), 0, 1)
        # Only apply severity to detected anomalies
        severity_scores = np.where(labels == 1, severity_scores, 0.0)

        # If contamination is specified and we have too many/few anomalies,
        # adjust threshold adaptively
        if self.contamination > 0:
            current_rate = np.mean(labels)
            if current_rate > self.contamination * 1.5:
                # Too many anomalies, increase threshold
                adaptive_threshold = np.percentile(
                    z_scores, (1 - self.contamination) * 100
                )
                labels = (z_scores > adaptive_threshold).astype(int)
                severity_scores = np.clip(z_scores / (adaptive_threshold * 2), 0, 1)
                severity_scores = np.where(labels == 1, severity_scores, 0.0)

        return labels, severity_scores, z_scores

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

    def calculate_sample_weights(
        self, labels: np.ndarray, severity_scores: np.ndarray, 
        min_weight: float = 0.1, max_weight: float = 1.0
    ) -> np.ndarray:
        """
        Calculate sample weights for training.
        Normal points get max_weight, anomalies get reduced weight based on severity.

        Args:
            labels: Anomaly labels (0 or 1)
            severity_scores: Severity scores (0-1)
            min_weight: Minimum weight for severe anomalies
            max_weight: Maximum weight for normal points

        Returns:
            Array of sample weights
        """
        # Normal points get full weight
        # Anomalies get weight reduced by severity: weight = max_weight - (severity * (max_weight - min_weight))
        weights = np.where(
            labels == 1,
            max_weight - severity_scores * (max_weight - min_weight),
            max_weight
        )
        return np.clip(weights, min_weight, max_weight)


def detect_anomalies_in_series(
    series: pd.Series,
    contamination: float = 0.1,
    random_state: int = 42,
    window: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series, np.ndarray, Dict]:
    """
    Convenience function to detect anomalies in a pandas Series.
    Uses time-aware rolling statistics approach.

    Args:
        series: Time series with Date index
        contamination: Expected proportion of anomalies
        random_state: Random seed
        window: Rolling window size (auto-determined if None)

    Returns:
        Tuple of (labels_series, severity_series, sample_weights, statistics_dict)
    """
    values = series.values

    # Auto-determine window based on data frequency
    if window is None:
        # For monthly data: use 12 months
        # For weekly data: use 12 weeks
        # For daily data: use 30 days
        if len(series) > 0:
            # Try to infer frequency from index
            if hasattr(series.index, "freq") and series.index.freq:
                freq_str = str(series.index.freq)
                if "M" in freq_str or "MS" in freq_str:
                    window = 12  # Monthly
                elif "W" in freq_str:
                    window = 12  # Weekly
                elif "D" in freq_str:
                    window = 30  # Daily
                else:
                    window = min(12, len(series) // 4)  # Default
            else:
                # Estimate from date differences
                if len(series) > 1:
                    avg_diff = (series.index[-1] - series.index[0]).days / len(series)
                    if avg_diff > 25:
                        window = 12  # Monthly
                    elif avg_diff > 5:
                        window = 12  # Weekly
                    else:
                        window = 30  # Daily
                else:
                    window = 12
        else:
            window = 12

    # Adjust window if data is too short
    window = min(window, len(series) // 2) if len(series) > 0 else 12
    window = max(3, window)  # Minimum window of 3

    detector = AnomalyDetector(
        window=window, contamination=contamination, random_state=random_state
    )
    labels, severity_scores, z_scores = detector.fit_transform(values)

    labels_series = pd.Series(labels, index=series.index, name="anomaly_label")
    severity_series = pd.Series(
        severity_scores, index=series.index, name="anomaly_severity"
    )

    # Calculate sample weights
    sample_weights = detector.calculate_sample_weights(labels, severity_scores)

    # Calculate statistics
    n_total = len(labels)
    n_anomalies = np.sum(labels)
    anomaly_rate = n_anomalies / n_total if n_total > 0 else 0.0

    anomaly_severities = severity_scores[labels == 1]
    mean_severity = (
        np.mean(anomaly_severities) if len(anomaly_severities) > 0 else 0.0
    )

    stats = {
        "n_total": n_total,
        "n_anomalies": n_anomalies,
        "anomaly_rate": anomaly_rate,
        "mean_severity": mean_severity,
    }

    logger.info(
        f"Detected {n_anomalies} anomalies ({anomaly_rate*100:.1f}%) "
        f"with mean severity {mean_severity:.3f}"
    )

    return labels_series, severity_series, sample_weights, stats

