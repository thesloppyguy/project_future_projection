"""
Rolling forecast calibration module.
Implements rolling forecast with actuals for model calibration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from .models import ModelTrainer
from .feature_engineering import FeatureEngineer
from .anomaly_detection import detect_anomalies_in_series
from .config import ENABLE_ANOMALY_DETECTION, ANOMALY_CONTAMINATION
from .utils import detect_frequency, create_future_dates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RollingForecastCalibrator:
    """Implements rolling forecast with calibration."""

    def __init__(
        self,
        model_trainer: ModelTrainer,
        feature_engineer: FeatureEngineer,
        rolling_window_months: int = 1,
    ):
        """
        Initialize rolling forecast calibrator.

        Args:
            model_trainer: ModelTrainer instance
            feature_engineer: FeatureEngineer instance
            rolling_window_months: How often to update forecast (in months)
        """
        self.model_trainer = model_trainer
        self.feature_engineer = feature_engineer
        self.rolling_window_months = rolling_window_months

    def rolling_forecast_calibration(
        self,
        train_series: pd.Series,
        calibration_series: pd.Series,
        model_name: str,
        forecast_horizon: int = 12,
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform rolling forecast during calibration period.

        Args:
            train_series: Training time series
            calibration_series: Calibration time series (with actuals)
            model_name: Name of model to use
            forecast_horizon: Forecast horizon in periods

        Returns:
            Dictionary with forecasts and metrics for each rolling window
        """
        logger.info(f"Starting rolling forecast calibration for {model_name}")
        logger.info(
            f"Calibration period: {calibration_series.index.min()} to {calibration_series.index.max()}"
        )

        results = {
            "forecasts": [],
            "metrics": [],
            "all_forecasts": pd.DataFrame(),
            "all_actuals": pd.DataFrame(),
        }

        # Detect frequency from series
        freq, _ = detect_frequency(calibration_series)
        logger.info(f"Detected frequency: {freq}")

        # Get calibration dates based on detected frequency
        calibration_dates = pd.date_range(
            start=calibration_series.index.min(),
            end=calibration_series.index.max(),
            freq=freq,
        )

        # Filter to actual dates in calibration series
        calibration_dates = [
            d for d in calibration_dates if d in calibration_series.index
        ]

        logger.info(f"Rolling forecast at {len(calibration_dates)} time points")

        for i, current_date in enumerate(calibration_dates):
            logger.info(
                f"Rolling forecast {i + 1}/{len(calibration_dates)}: {current_date}"
            )

            # Get training data up to current date
            train_data = train_series[train_series.index < current_date].copy()

            # Get actuals for next periods (for evaluation)
            future_dates = create_future_dates(
                current_date,
                min(forecast_horizon, len(calibration_dates) - i),
                freq
            )
            future_actuals = calibration_series[
                calibration_series.index.isin(future_dates)
            ]

            if len(train_data) < 12:  # Minimum training data
                logger.warning(
                    f"Insufficient training data at {current_date}, skipping"
                )
                continue

            # Train model
            train_result = self._train_model(train_data, model_name)
            if train_result is None:
                logger.warning(f"Model training failed at {current_date}, skipping")
                continue

            model, feature_names = train_result

            # Generate forecast (feature_names is None for Prophet)
            forecast = self._generate_forecast(
                model,
                train_data,
                model_name,
                len(future_actuals),
                current_date,
                feature_names,
                freq=freq,
            )

            if forecast is None or len(forecast) == 0:
                logger.warning(
                    f"Forecast generation failed at {current_date}, skipping"
                )
                continue

            # Align forecast with actuals
            aligned_forecast, aligned_actuals = self._align_forecast_actuals(
                forecast, future_actuals
            )

            if len(aligned_forecast) > 0:
                # Calculate metrics
                metrics = self._calculate_metrics(aligned_actuals, aligned_forecast)
                metrics["forecast_date"] = current_date
                metrics["n_periods"] = len(aligned_forecast)

                results["forecasts"].append(
                    {
                        "date": current_date,
                        "forecast": aligned_forecast,
                        "actuals": aligned_actuals,
                    }
                )
                results["metrics"].append(metrics)

                # Store for overall analysis
                forecast_df = pd.DataFrame(
                    {
                        "forecast_date": current_date,
                        "date": aligned_forecast.index,
                        "forecast": aligned_forecast.values,
                    }
                )
                actuals_df = pd.DataFrame(
                    {
                        "forecast_date": current_date,
                        "date": aligned_actuals.index,
                        "actual": aligned_actuals.values,
                    }
                )

                if len(results["all_forecasts"]) == 0:
                    results["all_forecasts"] = forecast_df
                    results["all_actuals"] = actuals_df
                else:
                    results["all_forecasts"] = pd.concat(
                        [results["all_forecasts"], forecast_df]
                    )
                    results["all_actuals"] = pd.concat(
                        [results["all_actuals"], actuals_df]
                    )

        # Convert to DataFrames
        if len(results["metrics"]) > 0:
            results["metrics_df"] = pd.DataFrame(results["metrics"])
        else:
            results["metrics_df"] = pd.DataFrame()

        logger.info(
            f"Rolling forecast calibration completed. {len(results['forecasts'])} forecasts generated"
        )

        return results

    def _train_model(
        self, train_series: pd.Series, model_name: str
    ) -> Optional[Tuple[Any, Optional[List[str]]]]:
        """Train model on training series. Returns (model, feature_names)."""
        try:
            # Time series models (work directly with series)
            if model_name == "prophet":
                return (self.model_trainer.train_prophet(train_series), None)
            elif model_name == "auto_arima":
                return (self.model_trainer.train_auto_arima(train_series), None)
            elif model_name == "holt_winters":
                return (self.model_trainer.train_holt_winters(train_series), None)
            elif model_name == "stl_decomposition":
                return (self.model_trainer.train_stl_decomposition(train_series), None)
            elif model_name == "sarimax":
                return (self.model_trainer.train_sarimax(train_series), None)
            else:
                # ML models (need features)
                # Prepare features
                X, y, feature_names = self.feature_engineer.prepare_ml_features(
                    train_series
                )

                if len(X) < 12:
                    return None

                # Detect anomalies and calculate sample weights if enabled
                sample_weight = None
                if ENABLE_ANOMALY_DETECTION:
                    try:
                        _, _, sample_weights, stats = detect_anomalies_in_series(
                            train_series,
                            contamination=ANOMALY_CONTAMINATION,
                            random_state=42
                        )
                        # Align sample weights with feature matrix (after removing NaN rows)
                        # Since features are created from series, weights should align
                        # But we need to handle the case where some rows might be dropped
                        if len(sample_weights) == len(X):
                            # Split weights same way as data
                            split_idx = int(len(X) * 0.8)
                            sample_weight = sample_weights[:split_idx]
                            logger.info(
                                f"Using anomaly-aware training: {stats['n_anomalies']} anomalies "
                                f"({stats['anomaly_rate']*100:.1f}%) detected"
                            )
                        else:
                            logger.warning(
                                f"Sample weights length ({len(sample_weights)}) doesn't match "
                                f"feature matrix length ({len(X)}), skipping anomaly weighting"
                            )
                    except Exception as e:
                        logger.warning(f"Anomaly detection failed: {e}, training without weights")

                # Split for validation
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                if model_name == "lightgbm":
                    model = self.model_trainer.train_lightgbm(
                        X_train, y_train, X_val, y_val, sample_weight=sample_weight
                    )
                elif model_name == "xgboost":
                    model = self.model_trainer.train_xgboost(
                        X_train, y_train, X_val, y_val, sample_weight=sample_weight
                    )
                elif model_name == "catboost":
                    model = self.model_trainer.train_catboost(
                        X_train, y_train, X_val, y_val, sample_weight=sample_weight
                    )
                elif model_name == "random_forest":
                    model = self.model_trainer.train_random_forest(
                        X_train, y_train, X_val, y_val, sample_weight=sample_weight
                    )
                elif model_name == "quantile_regression":
                    model = self.model_trainer.train_quantile_regression(
                        X_train, y_train, X_val, y_val, sample_weight=sample_weight
                    )
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    return None

                return (model, feature_names) if model is not None else None
        except Exception as e:
            logger.error(f"Error training model {model_name}: {e}")
            return None

    def _generate_forecast(
        self,
        model: Any,
        train_series: pd.Series,
        model_name: str,
        n_periods: int,
        last_date: pd.Timestamp,
        feature_names: Optional[List[str]] = None,
        freq: str = "MS",
    ) -> Optional[pd.Series]:
        """Generate forecast from model."""
        try:
            # Time series models
            if model_name == "prophet":
                return self.model_trainer.forecast_prophet(model, n_periods, last_date, freq)
            elif model_name == "auto_arima":
                if isinstance(model, dict) and 'last_date' in model:
                    last_date = model['last_date']
                return self.model_trainer.forecast_auto_arima(model, n_periods, last_date, freq)
            elif model_name == "holt_winters":
                if isinstance(model, dict) and 'last_date' in model:
                    last_date = model['last_date']
                return self.model_trainer.forecast_holt_winters(model, n_periods, last_date, freq)
            elif model_name == "stl_decomposition":
                if isinstance(model, dict) and 'last_date' in model:
                    last_date = model['last_date']
                return self.model_trainer.forecast_stl_decomposition(model, n_periods, last_date, freq)
            elif model_name == "sarimax":
                if isinstance(model, dict) and 'last_date' in model:
                    last_date = model['last_date']
                return self.model_trainer.forecast_sarimax(model, n_periods, last_date, freq)
            else:
                # ML models
                # For ML models, use FeatureEngineer to create features consistently
                if feature_names is None:
                    # Fallback: get feature names from training data
                    _, _, feature_names = self.feature_engineer.prepare_ml_features(
                        train_series
                    )

                # Get last values for lag features (need at least 12 for yoy_growth)
                max_lag_needed = (
                    max(self.feature_engineer.lag_periods)
                    if self.feature_engineer.lag_periods
                    else 12
                )
                last_values = train_series.tail(max(12, max_lag_needed)).values

                # Generate future dates using detected frequency
                future_dates = create_future_dates(last_date, n_periods, freq)

                forecasts = []
                current_values = list(last_values)

                for i, future_date in enumerate(future_dates):
                    # Create features using FeatureEngineer method
                    features_dict = self.feature_engineer.create_features_for_forecast(
                        current_values, future_date, len(train_series) + i
                    )

                    # Create feature vector in the same order as training
                    feature_vector = np.array(
                        [features_dict.get(name, 0) for name in feature_names]
                    ).reshape(1, -1)

                    # Predict
                    if model_name == "lightgbm":
                        pred = self.model_trainer.forecast_lightgbm(
                            model, feature_vector
                        )[0]
                    elif model_name == "xgboost":
                        pred = self.model_trainer.forecast_xgboost(
                            model, feature_vector
                        )[0]
                    elif model_name == "catboost":
                        pred = self.model_trainer.forecast_catboost(
                            model, feature_vector
                        )[0]
                    elif model_name == "random_forest":
                        pred = self.model_trainer.forecast_random_forest(
                            model, feature_vector
                        )[0]
                    elif model_name == "quantile_regression":
                        pred = self.model_trainer.forecast_quantile_regression(
                            model, feature_vector
                        )[0]
                    else:
                        pred = 0

                    pred = max(0, pred)  # Ensure non-negative
                    forecasts.append(pred)
                    current_values.append(pred)
                    # Keep enough values for lag features
                    max_lag = (
                        max(self.feature_engineer.lag_periods)
                        if self.feature_engineer.lag_periods
                        else 12
                    )
                    if len(current_values) > max_lag:
                        current_values.pop(0)

                return pd.Series(forecasts, index=future_dates)
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def _align_forecast_actuals(
        self, forecast: pd.Series, actuals: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Align forecast and actuals on common dates."""
        common_dates = forecast.index.intersection(actuals.index)
        if len(common_dates) == 0:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        aligned_forecast = forecast.loc[common_dates]
        aligned_actuals = actuals.loc[common_dates]

        return aligned_forecast, aligned_actuals

    def _calculate_metrics(
        self, actuals: pd.Series, forecast: pd.Series
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        if len(actuals) == 0 or len(forecast) == 0:
            return {
                "rmse": np.nan,
                "mae": np.nan,
                "mape": np.nan,
                "nrmse": np.nan,
                "r2": np.nan,
            }

        # Align
        common_idx = actuals.index.intersection(forecast.index)
        actuals_aligned = actuals.loc[common_idx]
        forecast_aligned = forecast.loc[common_idx]

        if len(actuals_aligned) == 0:
            return {
                "rmse": np.nan,
                "mae": np.nan,
                "mape": np.nan,
                "nrmse": np.nan,
                "r2": np.nan,
            }

        # Calculate metrics
        rmse = np.sqrt(np.mean((actuals_aligned - forecast_aligned) ** 2))
        mae = np.mean(np.abs(actuals_aligned - forecast_aligned))

        # MAPE (handle zero actuals)
        non_zero_mask = actuals_aligned != 0
        if non_zero_mask.sum() > 0:
            mape = (
                np.mean(
                    np.abs(
                        (
                            actuals_aligned[non_zero_mask]
                            - forecast_aligned[non_zero_mask]
                        )
                        / actuals_aligned[non_zero_mask]
                    )
                )
                * 100
            )
        else:
            mape = np.nan

        # NRMSE
        if actuals_aligned.mean() != 0:
            nrmse = (rmse / actuals_aligned.mean()) * 100
        else:
            nrmse = np.nan

        # R²
        ss_res = np.sum((actuals_aligned - forecast_aligned) ** 2)
        ss_tot = np.sum((actuals_aligned - actuals_aligned.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        return {"rmse": rmse, "mae": mae, "mape": mape, "nrmse": nrmse, "r2": r2}
