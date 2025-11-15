"""
Main orchestration script for comprehensive forecasting pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from forecast_implementation.config import *
from forecast_implementation.data_preparation import DataPreparator
from forecast_implementation.feature_engineering import FeatureEngineer
from forecast_implementation.models import ModelTrainer
from forecast_implementation.rolling_forecast import RollingForecastCalibrator
from forecast_implementation.evaluation import Evaluator
from forecast_implementation.anomaly_detection import detect_anomalies_in_series
from forecast_implementation.utils import detect_frequency
from forecast_implementation.ensemble_optimizer import EnsembleOptimizer
from forecast_implementation.diagnostics import ForecastDiagnostics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ForecastingPipeline:
    """Main forecasting pipeline orchestrator."""

    def __init__(self):
        """Initialize pipeline."""
        self.data_preparator = DataPreparator(DATA_FILE, MISSING_MONTH)
        self.feature_engineer = FeatureEngineer(
            lag_periods=LAG_PERIODS,
            rolling_windows=ROLLING_WINDOWS,
            include_seasonality=INCLUDE_SEASONALITY,
            include_trend=INCLUDE_TREND,
        )
        self.model_trainer = ModelTrainer(random_state=RANDOM_STATE)
        self.evaluator = Evaluator()
        self.ensemble_optimizer = EnsembleOptimizer(
            max_mape_threshold=ENSEMBLE_MAX_MAPE_THRESHOLD,
            min_models=ENSEMBLE_MIN_MODELS,
            use_calibration_weights=True,
        )
        self.diagnostics = None  # Will be initialized when saving results

        self.aggregated_data = None
        self.trained_models = {}
        self.calibration_results = {}
        self.final_forecasts = {}
        self.blind_evaluation_results = {}

    def run_full_pipeline(self):
        """Run the complete forecasting pipeline."""
        logger.info("=" * 80)
        logger.info("Starting Comprehensive Forecasting Pipeline")
        logger.info("=" * 80)

        # Step 1: Data Preparation
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Data Preparation")
        logger.info("=" * 80)
        self.prepare_data()

        # Step 2: Model Development & Calibration
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Model Development & Calibration")
        logger.info("=" * 80)
        self.develop_and_calibrate_models()

        # Step 3: Final Model Training
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Final Model Training")
        logger.info("=" * 80)
        self.train_final_models()

        # Step 4: Generate Forecasts
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Generate Final Forecasts")
        logger.info("=" * 80)
        self.generate_final_forecasts()

        # Step 5: Blind Evaluation
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Blind Evaluation")
        logger.info("=" * 80)
        self.evaluate_blind_data()

        # Step 6: Save Results
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Save Results")
        logger.info("=" * 80)
        self.save_all_results()

        logger.info("\n" + "=" * 80)
        logger.info("Pipeline Completed Successfully!")
        logger.info("=" * 80)

    def prepare_data(self):
        """Prepare and aggregate data."""
        logger.info("Loading and preparing data...")

        # Load and prepare aggregated data
        self.aggregated_data = self.data_preparator.prepare_aggregated_data()

        logger.info("Data preparation completed")
        logger.info(
            f"Monthly combined: {len(self.aggregated_data['monthly']['combined'])} rows"
        )
        logger.info(
            f"Monthly branch-wise: {len(self.aggregated_data['monthly']['branch_wise'])} rows"
        )
        logger.info(
            f"Weekly combined: {len(self.aggregated_data['weekly']['combined'])} rows"
        )
        logger.info(
            f"Weekly branch-wise: {len(self.aggregated_data['weekly']['branch_wise'])} rows"
        )

    def develop_and_calibrate_models(self):
        """Develop models and perform rolling forecast calibration."""
        logger.info("Starting model development and calibration...")

        # Process each aggregation level
        for freq in ["monthly", "weekly"]:
            for agg_level in AGGREGATION_LEVELS:
                logger.info(f"\nProcessing {freq} - {agg_level}")

                # Get data
                df = self.aggregated_data[freq][agg_level]

                # Split data
                splits = self.data_preparator.split_data(
                    df,
                    TRAIN_END_DATE,
                    CALIBRATION_START_DATE,
                    CALIBRATION_END_DATE,
                    BLIND_START_DATE,
                )

                # Prepare series
                if agg_level == "combined":
                    train_series = self.data_preparator.prepare_series(
                        splits["train"], value_col="Quantity"
                    )["combined"]
                    calibration_series = self.data_preparator.prepare_series(
                        splits["calibration"], value_col="Quantity"
                    )["combined"]
                else:
                    # Branch-wise: process each branch
                    train_series_dict = self.data_preparator.prepare_series(
                        splits["train"], value_col="Quantity", group_by="Branch"
                    )
                    calibration_series_dict = self.data_preparator.prepare_series(
                        splits["calibration"], value_col="Quantity", group_by="Branch"
                    )

                    # Process each branch
                    for branch in train_series_dict.keys():
                        if branch not in calibration_series_dict:
                            continue

                        train_series = train_series_dict[branch]
                        calibration_series = calibration_series_dict[branch]

                        self._calibrate_models_for_series(
                            train_series,
                            calibration_series,
                            f"{freq}_{agg_level}_{branch}",
                        )
                    continue

                # Calibrate models for combined series
                self._calibrate_models_for_series(
                    train_series, calibration_series, f"{freq}_{agg_level}_combined"
                )

    def _calibrate_models_for_series(
        self, train_series: pd.Series, calibration_series: pd.Series, series_key: str
    ):
        """Calibrate models for a specific series."""
        logger.info(f"Calibrating models for {series_key}")

        if len(train_series) < MIN_TRAIN_MONTHS:
            logger.warning(f"Insufficient training data for {series_key}, skipping")
            return

        calibrator = RollingForecastCalibrator(
            self.model_trainer,
            self.feature_engineer,
            rolling_window_months=ROLLING_WINDOW_MONTHS,
        )

        calibration_results = {}

        for model_name in MODELS_TO_TRAIN:
            logger.info(f"  Calibrating {model_name}...")
            try:
                results = calibrator.rolling_forecast_calibration(
                    train_series,
                    calibration_series,
                    model_name,
                    forecast_horizon=FORECAST_HORIZON_MONTHS,
                )
                calibration_results[model_name] = results

                # Log summary metrics
                if len(results["metrics"]) > 0:
                    avg_rmse = np.mean(
                        [
                            m["rmse"]
                            for m in results["metrics"]
                            if not np.isnan(m["rmse"])
                        ]
                    )
                    avg_mae = np.mean(
                        [m["mae"] for m in results["metrics"] if not np.isnan(m["mae"])]
                    )
                    logger.info(
                        f"    Average RMSE: {avg_rmse:.2f}, Average MAE: {avg_mae:.2f}"
                    )
            except Exception as e:
                logger.error(f"    Error calibrating {model_name}: {e}")
                continue

        self.calibration_results[series_key] = calibration_results

    def train_final_models(self):
        """Train final models on full training data (including calibration period)."""
        logger.info("Training final models on full data...")

        for freq in ["monthly", "weekly"]:
            for agg_level in AGGREGATION_LEVELS:
                logger.info(f"\nTraining final models for {freq} - {agg_level}")

                # Get full data (train + calibration)
                df = self.aggregated_data[freq][agg_level]
                full_train = df[df["Date"] <= CALIBRATION_END_DATE].copy()

                # Prepare series
                if agg_level == "combined":
                    full_series = self.data_preparator.prepare_series(
                        full_train, value_col="Quantity"
                    )["combined"]

                    self._train_final_models_for_series(
                        full_series, f"{freq}_{agg_level}_combined"
                    )
                else:
                    # Branch-wise
                    series_dict = self.data_preparator.prepare_series(
                        full_train, value_col="Quantity", group_by="Branch"
                    )

                    for branch, series in series_dict.items():
                        self._train_final_models_for_series(
                            series, f"{freq}_{agg_level}_{branch}"
                        )

    def _train_final_models_for_series(self, series: pd.Series, series_key: str):
        """Train final models for a specific series."""
        logger.info(f"  Training final models for {series_key}")

        if len(series) < MIN_TRAIN_MONTHS:
            logger.warning(f"Insufficient data for {series_key}, skipping")
            return

        models = {}

        for model_name in MODELS_TO_TRAIN:
            logger.info(f"    Training {model_name}...")
            try:
                feature_names = None

                # Time series models (work directly with series)
                if model_name == "prophet":
                    model = self.model_trainer.train_prophet(series)
                elif model_name == "auto_arima":
                    model = self.model_trainer.train_auto_arima(series)
                elif model_name == "holt_winters":
                    model = self.model_trainer.train_holt_winters(series)
                elif model_name == "stl_decomposition":
                    model = self.model_trainer.train_stl_decomposition(series)
                elif model_name == "sarimax":
                    model = self.model_trainer.train_sarimax(series)
                else:
                    # ML models (need features)
                    X, y, feature_names = self.feature_engineer.prepare_ml_features(
                        series
                    )

                    if len(X) < 12:
                        continue

                    # Detect anomalies and calculate sample weights if enabled
                    sample_weight = None
                    if ENABLE_ANOMALY_DETECTION:
                        try:
                            _, _, sample_weights, stats = detect_anomalies_in_series(
                                series,
                                contamination=ANOMALY_CONTAMINATION,
                                random_state=RANDOM_STATE,
                            )
                            if len(sample_weights) == len(X):
                                split_idx = int(len(X) * 0.8)
                                sample_weight = sample_weights[:split_idx]
                                logger.info(
                                    f"      Anomaly-aware training: {stats['n_anomalies']} anomalies "
                                    f"({stats['anomaly_rate'] * 100:.1f}%) detected"
                                )
                        except Exception as e:
                            logger.warning(
                                f"      Anomaly detection failed: {e}, training without weights"
                            )

                    # Use 80/20 split for validation
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
                        model = None

                    if model is None:
                        continue

                if model is not None:
                    models[model_name] = {
                        "model": model,
                        "series": series,
                        "feature_names": feature_names,
                    }

                    # Save model
                    model_path = MODEL_DIR / f"{series_key}_{model_name}.pkl"
                    self.model_trainer.save_model(model, model_name, model_path)
            except Exception as e:
                logger.error(f"    Error training {model_name}: {e}")
                continue

        self.trained_models[series_key] = models
        logger.info(f"    Trained {len(models)} models for {series_key}")

    def generate_final_forecasts(self):
        """Generate final 12-month forecasts."""
        logger.info("Generating final forecasts...")

        for series_key, models in self.trained_models.items():
            logger.info(f"\nGenerating forecasts for {series_key}")

            series = models[list(models.keys())[0]]["series"]
            last_date = series.index.max()

            # Detect frequency and calculate appropriate forecast horizon
            freq, periods_per_year = detect_frequency(series)
            if freq == "W-MON" or "W" in freq:
                # For weekly, convert months to weeks (approximately 4.33 weeks per month)
                forecast_horizon = int(FORECAST_HORIZON_MONTHS * 4.33)
            else:
                # For monthly, use months directly
                forecast_horizon = FORECAST_HORIZON_MONTHS

            forecasts = {}

            for model_name, model_data in models.items():
                logger.info(f"  Forecasting with {model_name}...")
                try:
                    model = model_data["model"]

                    # Time series models
                    if model_name == "prophet":
                        forecast = self.model_trainer.forecast_prophet(
                            model, forecast_horizon, last_date, freq
                        )
                    elif model_name == "auto_arima":
                        if isinstance(model, dict) and "last_date" in model:
                            last_date = model["last_date"]
                        forecast = self.model_trainer.forecast_auto_arima(
                            model, forecast_horizon, last_date, freq
                        )
                    elif model_name == "holt_winters":
                        if isinstance(model, dict) and "last_date" in model:
                            last_date = model["last_date"]
                        forecast = self.model_trainer.forecast_holt_winters(
                            model, forecast_horizon, last_date, freq
                        )
                    elif model_name == "stl_decomposition":
                        if isinstance(model, dict) and "last_date" in model:
                            last_date = model["last_date"]
                        forecast = self.model_trainer.forecast_stl_decomposition(
                            model, forecast_horizon, last_date, freq
                        )
                    elif model_name == "sarimax":
                        if isinstance(model, dict) and "last_date" in model:
                            last_date = model["last_date"]
                        forecast = self.model_trainer.forecast_sarimax(
                            model, forecast_horizon, last_date, freq
                        )
                    else:
                        # For ML models, use rolling forecast approach
                        feature_names = model_data.get("feature_names")
                        forecast = self._generate_ml_forecast(
                            model,
                            series,
                            model_name,
                            forecast_horizon,
                            last_date,
                            feature_names,
                        )

                    if forecast is not None and len(forecast) > 0:
                        forecasts[model_name] = forecast
                        logger.info(f"    Generated {len(forecast)} periods")
                except Exception as e:
                    logger.error(
                        f"    Error generating forecast with {model_name}: {e}"
                    )
                    continue

            # Create optimized ensemble if enabled
            if USE_ENSEMBLE and len(forecasts) > 1:
                logger.info("  Creating optimized ensemble forecast...")

                # Get calibration results for this series if available
                cal_results = self.calibration_results.get(series_key, None)

                # Use optimized ensemble
                ensemble, ensemble_metadata = (
                    self.ensemble_optimizer.create_optimized_ensemble(
                        forecasts,
                        actuals=None,  # No actuals available during forecasting
                        calibration_results=cal_results,
                        method=ENSEMBLE_METHOD,
                    )
                )

                if len(ensemble) > 0:
                    forecasts["ensemble"] = ensemble
                    logger.info(
                        f"    Optimized ensemble created with {len(ensemble_metadata['filtered_models'])} models: "
                        f"{', '.join(ensemble_metadata['filtered_models'])}"
                    )
                else:
                    logger.warning(
                        "    Failed to create ensemble, using simple average"
                    )
                    # Fallback to simple ensemble
                    weights = self._calculate_ensemble_weights(
                        series_key, list(forecasts.keys())
                    )
                    ensemble = self.evaluator.create_ensemble_forecast(
                        forecasts, method=ENSEMBLE_METHOD, weights=weights
                    )
                    forecasts["ensemble"] = ensemble

            self.final_forecasts[series_key] = forecasts

    def _generate_ml_forecast(
        self,
        model,
        series: pd.Series,
        model_name: str,
        n_periods: int,
        last_date: pd.Timestamp,
        feature_names: Optional[List[str]] = None,
    ) -> pd.Series:
        """Generate forecast for ML model."""
        # Detect frequency from series
        freq, _ = detect_frequency(series)

        # Use the same approach as in rolling_forecast.py
        calibrator = RollingForecastCalibrator(
            self.model_trainer, self.feature_engineer
        )
        return calibrator._generate_forecast(
            model, series, model_name, n_periods, last_date, feature_names, freq=freq
        )

    def _calculate_ensemble_weights(
        self, series_key: str, model_names: List[str]
    ) -> Dict[str, float]:
        """Calculate ensemble weights based on calibration performance."""
        weights = {}

        # Get calibration results
        if series_key in self.calibration_results:
            cal_results = self.calibration_results[series_key]

            # Calculate average RMSE for each model
            model_rmse = {}
            for model_name in model_names:
                if model_name in cal_results:
                    metrics = cal_results[model_name]["metrics"]
                    if len(metrics) > 0:
                        avg_rmse = np.mean(
                            [
                                m["rmse"]
                                for m in metrics
                                if not np.isnan(m.get("rmse", np.nan))
                            ]
                        )
                        if not np.isnan(avg_rmse) and avg_rmse > 0:
                            model_rmse[model_name] = avg_rmse

            # Calculate inverse weights (lower RMSE = higher weight)
            if len(model_rmse) > 0:
                total_inv_rmse = sum(1.0 / rmse for rmse in model_rmse.values())
                for model_name, rmse in model_rmse.items():
                    weights[model_name] = (1.0 / rmse) / total_inv_rmse
            else:
                # Equal weights if no calibration data
                weights = {name: 1.0 / len(model_names) for name in model_names}
        else:
            # Equal weights if no calibration results
            weights = {name: 1.0 / len(model_names) for name in model_names}

        return weights

    def evaluate_blind_data(self):
        """Evaluate forecasts on blind data."""
        logger.info("Evaluating on blind data...")

        for freq in ["monthly", "weekly"]:
            for agg_level in AGGREGATION_LEVELS:
                logger.info(f"\nEvaluating {freq} - {agg_level}")

                # Get blind data
                df = self.aggregated_data[freq][agg_level]
                blind_data = df[df["Date"] >= BLIND_START_DATE].copy()

                if len(blind_data) == 0:
                    logger.warning(f"No blind data available for {freq} - {agg_level}")
                    continue

                # Prepare series
                if agg_level == "combined":
                    blind_series = self.data_preparator.prepare_series(
                        blind_data, value_col="Quantity"
                    )["combined"]

                    series_key = f"{freq}_{agg_level}_combined"
                    if series_key in self.final_forecasts:
                        self._evaluate_forecasts_for_series(
                            self.final_forecasts[series_key], blind_series, series_key
                        )
                else:
                    # Branch-wise
                    blind_series_dict = self.data_preparator.prepare_series(
                        blind_data, value_col="Quantity", group_by="Branch"
                    )

                    for branch, blind_series in blind_series_dict.items():
                        series_key = f"{freq}_{agg_level}_{branch}"
                        if series_key in self.final_forecasts:
                            self._evaluate_forecasts_for_series(
                                self.final_forecasts[series_key],
                                blind_series,
                                series_key,
                            )

    def _evaluate_forecasts_for_series(
        self, forecasts: Dict[str, pd.Series], blind_series: pd.Series, series_key: str
    ):
        """Evaluate forecasts for a specific series."""
        logger.info(f"  Evaluating {series_key}")

        evaluation_results = {}

        for model_name, forecast in forecasts.items():
            logger.info(f"    Evaluating {model_name}...")
            metrics = self.evaluator.evaluate_forecast(
                forecast, blind_series, model_name
            )
            evaluation_results[model_name] = metrics

            logger.info(
                f"      RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, "
                f"MAPE: {metrics['mape']:.2f}%"
            )

        self.blind_evaluation_results[series_key] = evaluation_results

    def save_all_results(self):
        """Save all results to files."""
        logger.info("Saving results...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = RESULTS_DIR / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize diagnostics
        self.diagnostics = ForecastDiagnostics(results_dir)

        # Save calibration results
        calibration_dir = results_dir / "calibration"
        calibration_dir.mkdir(exist_ok=True)

        for series_key, cal_results in self.calibration_results.items():
            for model_name, results in cal_results.items():
                # Save metrics
                if "metrics_df" in results and len(results["metrics_df"]) > 0:
                    results["metrics_df"].to_csv(
                        calibration_dir / f"{series_key}_{model_name}_metrics.csv",
                        index=False,
                    )

                # Save forecasts
                if "all_forecasts" in results and len(results["all_forecasts"]) > 0:
                    results["all_forecasts"].to_csv(
                        calibration_dir / f"{series_key}_{model_name}_forecasts.csv",
                        index=False,
                    )

        # Save final forecasts
        forecasts_dir = results_dir / "forecasts"
        forecasts_dir.mkdir(exist_ok=True)

        for series_key, forecasts in self.final_forecasts.items():
            for model_name, forecast in forecasts.items():
                forecast_df = pd.DataFrame(
                    {"date": forecast.index, "forecast": forecast.values}
                )
                forecast_df.to_csv(
                    forecasts_dir / f"{series_key}_{model_name}_forecast.csv",
                    index=False,
                )

        # Save blind evaluation results
        evaluation_dir = results_dir / "evaluation"
        evaluation_dir.mkdir(exist_ok=True)

        # Create model comparison
        all_evaluation_results = {}
        for series_key, eval_results in self.blind_evaluation_results.items():
            for model_name, metrics in eval_results.items():
                key = f"{series_key}_{model_name}"
                all_evaluation_results[key] = metrics
                all_evaluation_results[key]["series"] = series_key
                all_evaluation_results[key]["model"] = model_name

        if len(all_evaluation_results) > 0:
            # Convert to list of dicts for DataFrame
            comparison_data = []
            for key, metrics in all_evaluation_results.items():
                row = metrics.copy()
                comparison_data.append(row)

            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(evaluation_dir / "model_comparison.csv", index=False)

            # Generate diagnostics and visualizations for weekly forecasts
            diagnostics_dir = results_dir / "diagnostics"
            diagnostics_dir.mkdir(exist_ok=True)
            plots_dir = results_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            for series_key, forecasts in self.final_forecasts.items():
                # Check if this is a weekly series
                if "weekly" in series_key:
                    # Get blind data for this series
                    if series_key in self.blind_evaluation_results:
                        # Extract frequency and aggregation level from series_key
                        parts = series_key.split("_")
                        if len(parts) >= 3:
                            freq_type = parts[0]  # "weekly"
                            agg_type = parts[1]  # "combined" or "branch_wise"

                            # Get the appropriate aggregated data
                            df = self.aggregated_data.get(freq_type, {}).get(agg_type)
                            if df is not None:
                                blind_data = df[df["Date"] >= BLIND_START_DATE].copy()

                                if agg_type == "combined":
                                    # Combined data
                                    blind_series = self.data_preparator.prepare_series(
                                        blind_data, value_col="Quantity"
                                    )["combined"]

                                    # Run diagnostics
                                    try:
                                        diag = (
                                            self.diagnostics.diagnose_weekly_forecasts(
                                                forecasts, blind_series, series_key
                                            )
                                        )
                                        self.diagnostics.save_diagnostics_report(
                                            diag,
                                            diagnostics_dir
                                            / f"{series_key}_diagnostics.csv",
                                        )

                                        # Generate plots
                                        self.diagnostics.plot_forecast_comparison(
                                            forecasts,
                                            blind_series,
                                            series_key,
                                            plots_dir
                                            / f"{series_key}_forecast_comparison.png",
                                        )
                                        self.diagnostics.plot_residuals_analysis(
                                            forecasts,
                                            blind_series,
                                            series_key,
                                            plots_dir / f"{series_key}_residuals.png",
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            f"Error generating diagnostics for {series_key}: {e}"
                                        )

                                elif agg_type == "branch_wise":
                                    # Branch-wise data - check if Branch column exists
                                    if "Branch" in blind_data.columns:
                                        blind_series_dict = (
                                            self.data_preparator.prepare_series(
                                                blind_data,
                                                value_col="Quantity",
                                                group_by="Branch",
                                            )
                                        )
                                        for (
                                            branch,
                                            blind_series,
                                        ) in blind_series_dict.items():
                                            if (
                                                f"{freq_type}_{agg_type}_{branch}"
                                                == series_key
                                            ):
                                                # Run diagnostics
                                                try:
                                                    diag = self.diagnostics.diagnose_weekly_forecasts(
                                                        forecasts,
                                                        blind_series,
                                                        series_key,
                                                    )
                                                    self.diagnostics.save_diagnostics_report(
                                                        diag,
                                                        diagnostics_dir
                                                        / f"{series_key}_diagnostics.csv",
                                                    )

                                                    # Generate plots
                                                    self.diagnostics.plot_forecast_comparison(
                                                        forecasts,
                                                        blind_series,
                                                        series_key,
                                                        plots_dir
                                                        / f"{series_key}_forecast_comparison.png",
                                                    )
                                                    self.diagnostics.plot_residuals_analysis(
                                                        forecasts,
                                                        blind_series,
                                                        series_key,
                                                        plots_dir
                                                        / f"{series_key}_residuals.png",
                                                    )
                                                except Exception as e:
                                                    logger.warning(
                                                        f"Error generating diagnostics for {series_key}: {e}"
                                                    )
                                                break
                                    else:
                                        logger.warning(
                                            f"Branch column not found in blind_data for {series_key}"
                                        )

            # Create summary
            if "model" in comparison_df.columns:
                summary = (
                    comparison_df.groupby("model")
                    .agg({"rmse": "mean", "mae": "mean", "mape": "mean", "r2": "mean"})
                    .sort_values("rmse")
                )
                summary.to_csv(evaluation_dir / "model_summary.csv")

        # Save configuration
        config_dict = {
            "train_end_date": TRAIN_END_DATE,
            "calibration_start_date": CALIBRATION_START_DATE,
            "calibration_end_date": CALIBRATION_END_DATE,
            "blind_start_date": BLIND_START_DATE,
            "forecast_horizon_months": FORECAST_HORIZON_MONTHS,
            "models_trained": MODELS_TO_TRAIN,
            "use_ensemble": USE_ENSEMBLE,
        }

        with open(results_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Results saved to {results_dir}")


def main():
    """Main entry point."""
    pipeline = ForecastingPipeline()
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
