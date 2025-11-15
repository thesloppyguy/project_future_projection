"""
Main orchestrator script for training all 16 time series forecasting models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.data_preprocessing import prepare_data_for_training
from training.evaluation import (
    evaluate_forecast,
    save_evaluation_results,
    generate_summary_report,
    save_anomaly_detection_results,
    generate_anomaly_summary_report,
)

# Import all models
from training.models import auto_arima
from training.models import sarimax
from training.models import holt_winters
from training.models import prophet
from training.models import neural_prophet
from training.models import stl_decomposition
from training.models import xgboost
from training.models import lightgbm
from training.models import catboost
from training.models import random_forest

# Deep learning models commented out for now
# from training.models import lstm
# from training.models import bayesian_deep_learning
from training.models import isolation_forest
from training.models import quantile_regression
from training.models import var
from training.models import kalman_filter


# Model configurations
MODELS = {
    "auto_arima": {"module": auto_arima, "name": "Auto-ARIMA"},
    "sarimax": {"module": sarimax, "name": "SARIMAX"},
    "holt_winters": {"module": holt_winters, "name": "Holt-Winters"},
    "prophet": {"module": prophet, "name": "Prophet"},
    "neural_prophet": {"module": neural_prophet, "name": "Neural Prophet"},
    "stl_decomposition": {"module": stl_decomposition, "name": "STL Decomposition"},
    "xgboost": {"module": xgboost, "name": "XGBoost"},
    "lightgbm": {"module": lightgbm, "name": "LightGBM"},
    "catboost": {"module": catboost, "name": "CatBoost"},
    "random_forest": {"module": random_forest, "name": "Random Forest"},
    # Deep learning models commented out for now
    # 'lstm': {
    #     'module': lstm,
    #     'name': 'LSTM'
    # },
    # 'bayesian_deep_learning': {
    #     'module': bayesian_deep_learning,
    #     'name': 'Bayesian Deep Learning'
    # },
    "isolation_forest": {"module": isolation_forest, "name": "Isolation Forest"},
    "quantile_regression": {
        "module": quantile_regression,
        "name": "Quantile Regression",
    },
    "var": {"module": var, "name": "Vector Autoregression"},
    "kalman_filter": {"module": kalman_filter, "name": "Kalman Filter"},
}

# Branches
BRANCHES = ["BLR", "COK", "MAA", "SBD", "SBD1"]

# Forecast periods for FY 2025-26 (April 2025 to March 2026)
WEEKLY_PERIODS = 52  # 52 weeks for FY 2025-26
MONTHLY_PERIODS = 12  # 12 months for FY 2025-26 (April 2025 to March 2026)


def train_and_forecast_model(
    model_key: str,
    model_config: dict,
    train_data: Dict,
    test_data: Dict,
    branch: str,
    aggregation: str,
    results_dir: Path,
) -> Dict:
    """
    Train a model and generate forecast for a specific branch and aggregation.

    Args:
        model_key: Model key (e.g., 'auto_arima')
        model_config: Model configuration dictionary
        train_data: Training data series
        test_data: Test data series
        branch: Branch name
        aggregation: 'weekly' or 'monthly'
        results_dir: Results directory path

    Returns:
        Evaluation dictionary
    """
    model_module = model_config["module"]
    model_name = model_config["name"]

    print(f"\n{'=' * 60}")
    print(f"Training {model_name} - {branch} - {aggregation}")
    print(f"{'=' * 60}")

    try:
        # Get training and test data (new structure with series, anomaly_labels, etc.)
        train_branch_data = train_data.get(branch, {})
        test_branch_data = test_data.get(branch, {})

        train_series = train_branch_data.get("series", pd.Series(dtype=float))
        test_series = test_branch_data.get("series", pd.Series(dtype=float))
        train_anomaly_labels = train_branch_data.get("anomaly_labels", None)
        train_anomaly_severity = train_branch_data.get("anomaly_severity", None)
        train_yoy_growth = train_branch_data.get("yoy_growth", None)

        if len(train_series) == 0:
            print(f"Warning: No training data for {branch} - {aggregation}")
            return {
                "model": model_name,
                "branch": branch,
                "aggregation": aggregation,
                "rmse": np.nan,
                "mae": np.nan,
                "mape": np.nan,
                "nrmse": np.nan,
                "n_samples": 0,
                "status": "no_data",
            }

        # Determine frequency and forecast periods
        if aggregation == "weekly":
            freq = "W-MON"
            n_periods = WEEKLY_PERIODS
        else:
            freq = "MS"
            n_periods = MONTHLY_PERIODS

        # Create validation split (80/20) maintaining temporal order
        if len(train_series) > 10:
            split_idx = int(len(train_series) * 0.8)
            train_series_split = train_series.iloc[:split_idx]
            val_series_split = train_series.iloc[split_idx:]

            # Split anomaly and YoY data if available
            if train_anomaly_labels is not None:
                train_anomaly_labels_split = train_anomaly_labels.iloc[:split_idx]
                val_anomaly_labels_split = train_anomaly_labels.iloc[split_idx:]
            else:
                train_anomaly_labels_split = None
                val_anomaly_labels_split = None

            if train_anomaly_severity is not None:
                train_anomaly_severity_split = train_anomaly_severity.iloc[:split_idx]
                val_anomaly_severity_split = train_anomaly_severity.iloc[split_idx:]
            else:
                train_anomaly_severity_split = None
                val_anomaly_severity_split = None

            if train_yoy_growth is not None:
                train_yoy_growth_split = train_yoy_growth.iloc[:split_idx]
                val_yoy_growth_split = train_yoy_growth.iloc[split_idx:]
            else:
                train_yoy_growth_split = None
                val_yoy_growth_split = None
        else:
            # Not enough data for validation split
            train_series_split = train_series
            val_series_split = pd.Series(dtype=float)
            train_anomaly_labels_split = train_anomaly_labels
            train_anomaly_severity_split = train_anomaly_severity
            train_yoy_growth_split = train_yoy_growth
            val_anomaly_labels_split = None
            val_anomaly_severity_split = None
            val_yoy_growth_split = None

        # Special handling for VAR (needs all branches)
        if model_key == "var":
            # Train VAR on all branches - extract series from new data structure
            all_train_data = {}
            all_test_data = {}
            for b in BRANCHES:
                branch_train_data = train_data.get(b, {})
                branch_test_data = test_data.get(b, {})
                branch_series = branch_train_data.get("series", pd.Series(dtype=float))
                if len(branch_series) > 0:
                    all_train_data[b] = branch_series
                branch_test_series = branch_test_data.get(
                    "series", pd.Series(dtype=float)
                )
                if len(branch_test_series) > 0:
                    all_test_data[b] = branch_test_series

            if len(all_train_data) < 2:
                print(
                    f"Warning: VAR needs at least 2 branches, got {len(all_train_data)}"
                )
                return {
                    "model": model_name,
                    "branch": branch,
                    "aggregation": aggregation,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "mape": np.nan,
                    "nrmse": np.nan,
                    "n_samples": 0,
                    "status": "insufficient_branches",
                }

            # Train model with hyperparameter optimization
            model = model_module.train_model(
                all_train_data, BRANCHES, freq, use_optimization=True, n_trials=20
            )
        else:
            # Train model with hyperparameter optimization, passing anomaly and YoY data
            model = model_module.train_model(
                train_series_split,
                branch,
                freq,
                use_optimization=True,
                n_trials=20,
                anomaly_labels=train_anomaly_labels_split,
                anomaly_severity=train_anomaly_severity_split,
                yoy_growth=train_yoy_growth_split,
            )

            # Update last_dates and last_values to use FULL training data (not just the split)
            # This ensures forecasts start from the correct date to align with test data
            if model is not None and isinstance(model, dict):
                # Update to use full training series for forecast date alignment
                if "last_dates" in model:
                    # Use the last dates from the FULL training series, not the split
                    model["last_dates"] = train_series.tail(12).index.tolist()
                    model["last_values"] = train_series.tail(12).values.tolist()
                # Also update last_date if it exists (for some models)
                if "last_date" in model:
                    model["last_date"] = (
                        train_series.index[-1]
                        if len(train_series) > 0
                        else pd.Timestamp.now()
                    )

        if model is None:
            print(
                f"Warning: Model training failed for {model_name} - {branch} - {aggregation}"
            )
            return {
                "model": model_name,
                "branch": branch,
                "aggregation": aggregation,
                "rmse": np.nan,
                "mae": np.nan,
                "mape": np.nan,
                "nrmse": np.nan,
                "r2": np.nan,
                "n_samples": 0,
                "status": "training_failed",
            }

        # Generate forecast for FY 2025-26 (April 2025 to March 2026)
        # Forecast starts from 2025-04-01
        forecast_start_date = pd.Timestamp("2025-04-01")

        # Generate forecast dates for FY 2025-26
        if aggregation == "weekly":
            forecast_dates = pd.date_range(
                start=forecast_start_date, periods=n_periods, freq="W-MON"
            )
        else:
            forecast_dates = pd.date_range(
                start=forecast_start_date, periods=n_periods, freq="MS"
            )

        # For YoY growth, we'll need to extrapolate or use historical patterns
        # Since we're forecasting into the future, we can't use actual test data YoY growth
        forecast_yoy_growth = None

        # Check if forecast function accepts yoy_growth parameter
        import inspect

        forecast_params = inspect.signature(model_module.forecast).parameters
        if "yoy_growth" in forecast_params:
            forecast_series = model_module.forecast(
                model, n_periods, freq, branch, yoy_growth=forecast_yoy_growth
            )
        else:
            forecast_series = model_module.forecast(model, n_periods, freq, branch)

        # Align forecast dates to FY 2025-26 dates
        if len(forecast_series) > 0:
            # Ensure forecast dates match FY 2025-26 period
            if not forecast_series.index.equals(forecast_dates):
                # Reindex to match FY 2025-26 dates
                if len(forecast_series) >= len(forecast_dates):
                    # Take first n_periods values and assign to FY 2025-26 dates
                    forecast_values = forecast_series.values[: len(forecast_dates)]
                    forecast_series = pd.Series(forecast_values, index=forecast_dates)
                else:
                    # Extend with last value if needed
                    forecast_values = list(forecast_series.values)
                    last_val = forecast_values[-1] if len(forecast_values) > 0 else 0
                    while len(forecast_values) < len(forecast_dates):
                        forecast_values.append(last_val)
                    forecast_series = pd.Series(
                        forecast_values[: len(forecast_dates)], index=forecast_dates
                    )

        # For evaluation, align forecast with test data dates if available
        if len(forecast_series) > 0 and len(test_series) > 0:
            # Get test data date range - use first n_periods dates from test data
            test_dates = test_series.index[: min(n_periods, len(test_series))]

            # Create evaluation forecast aligned to test dates
            eval_forecast = forecast_series.reindex(
                test_dates, method="nearest", fill_value=0
            )
            if len(eval_forecast) < len(test_dates):
                # Extend if needed
                last_val = eval_forecast.iloc[-1] if len(eval_forecast) > 0 else 0
                extended_values = list(eval_forecast.values) + [last_val] * (
                    len(test_dates) - len(eval_forecast)
                )
                eval_forecast = pd.Series(
                    extended_values[: len(test_dates)], index=test_dates
                )
        else:
            eval_forecast = forecast_series

        if len(forecast_series) == 0:
            print(
                f"Warning: Forecast generation failed for {model_name} - {branch} - {aggregation}"
            )
            return {
                "model": model_name,
                "branch": branch,
                "aggregation": aggregation,
                "rmse": np.nan,
                "mae": np.nan,
                "mape": np.nan,
                "nrmse": np.nan,
                "r2": np.nan,
                "n_samples": 0,
                "status": "forecast_failed",
            }

        # Save forecast
        forecast_dir = results_dir / aggregation / model_key
        forecast_dir.mkdir(parents=True, exist_ok=True)
        forecast_path = forecast_dir / f"{branch}_forecast.csv"
        forecast_df = pd.DataFrame(
            {"Date": forecast_series.index, "Forecast": forecast_series.values}
        )
        forecast_df.to_csv(forecast_path, index=False)

        # Save model (if applicable)
        model_dir = results_dir / "models" / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{branch}_{aggregation}.pkl"
        try:
            model_module.save_model(model, str(model_path))
        except Exception as e:
            print(f"Warning: Could not save model: {e}")

        # Evaluate forecast
        test_anomaly_labels = test_branch_data.get("anomaly_labels", None)
        test_anomaly_severity = test_branch_data.get("anomaly_severity", None)

        if len(test_series) > 0:
            evaluation = evaluate_forecast(
                test_series,
                eval_forecast,
                model_name,
                branch,
                aggregation,
                anomaly_labels=test_anomaly_labels,
                anomaly_severity=test_anomaly_severity,
            )
            evaluation["status"] = "success"
        else:
            print(f"Warning: No test data for {branch} - {aggregation}")
            evaluation = {
                "model": model_name,
                "branch": branch,
                "aggregation": aggregation,
                "rmse": np.nan,
                "mae": np.nan,
                "mape": np.nan,
                "nrmse": np.nan,
                "r2": np.nan,
                "n_samples": 0,
                "status": "no_test_data",
            }

        print(
            f"✓ {model_name} - {branch} - {aggregation}: RMSE={evaluation.get('rmse', np.nan):.2f}, MAE={evaluation.get('mae', np.nan):.2f}, R2={evaluation.get('r2', np.nan):.4f}"
        )

        return evaluation

    except Exception as e:
        print(f"Error in {model_name} - {branch} - {aggregation}: {e}")
        traceback.print_exc()
        return {
            "model": model_name,
            "branch": branch,
            "aggregation": aggregation,
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "nrmse": np.nan,
            "r2": np.nan,
            "n_samples": 0,
            "status": f"error: {str(e)}",
        }


def main():
    """Main function to train all models."""
    print("=" * 60)
    print("Multi-Model Time Series Forecasting Pipeline")
    print("=" * 60)

    # Paths
    train_path = Path("data/merged_filter_ingestion_2024.csv")
    test_path = Path("data/merged_filter_ingestion_2025.csv")
    results_dir = Path("training/results")

    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data with combined mode
    print("\nLoading and preprocessing data...")
    print(
        "Using combined data mode: combining 2024 and 2025 data for training/evaluation"
    )
    print("Forecast target: FY 2025-26 (April 2025 to March 2026)")
    data = prepare_data_for_training(
        str(train_path),
        str(test_path),
        branches=BRANCHES,
        apply_anomaly_detection=True,
        contamination=0.1,
        use_combined_data=True,  # Use combined data mode
    )

    print(f"Data loaded successfully!")
    print(f"  Branches: {BRANCHES}")
    print(
        f"  Training periods (weekly): {[len(data['train']['weekly'][b].get('series', pd.Series())) for b in BRANCHES]}"
    )
    print(
        f"  Training periods (monthly): {[len(data['train']['monthly'][b].get('series', pd.Series())) for b in BRANCHES]}"
    )
    if "validation" in data:
        print(
            f"  Validation periods (weekly): {[len(data['validation']['weekly'][b].get('series', pd.Series())) for b in BRANCHES]}"
        )
        print(
            f"  Validation periods (monthly): {[len(data['validation']['monthly'][b].get('series', pd.Series())) for b in BRANCHES]}"
        )
    print(
        f"  Test periods (weekly): {[len(data['test']['weekly'][b].get('series', pd.Series())) for b in BRANCHES]}"
    )
    print(
        f"  Test periods (monthly): {[len(data['test']['monthly'][b].get('series', pd.Series())) for b in BRANCHES]}"
    )

    # Store all evaluation results
    all_evaluations = []

    # Train all models
    for model_key, model_config in MODELS.items():
        print(f"\n{'#' * 60}")
        print(f"Processing Model: {model_config['name']}")
        print(f"{'#' * 60}")

        # Process weekly aggregation
        for branch in BRANCHES:
            evaluation = train_and_forecast_model(
                model_key,
                model_config,
                data["train"]["weekly"],
                data["test"]["weekly"],
                branch,
                "weekly",
                results_dir,
            )
            all_evaluations.append(evaluation)

        # Process monthly aggregation
        for branch in BRANCHES:
            evaluation = train_and_forecast_model(
                model_key,
                model_config,
                data["train"]["monthly"],
                data["test"]["monthly"],
                branch,
                "monthly",
                results_dir,
            )
            all_evaluations.append(evaluation)

    # Save evaluation results
    print(f"\n{'=' * 60}")
    print("Saving evaluation results...")
    print(f"{'=' * 60}")

    # Save per-model evaluations
    for model_key, model_config in MODELS.items():
        model_name = model_config["name"]

        # Weekly evaluations
        weekly_evals = [
            e
            for e in all_evaluations
            if e["model"] == model_name and e["aggregation"] == "weekly"
        ]
        if weekly_evals:
            weekly_path = (
                results_dir / "evaluations" / f"{model_key}_weekly_metrics.csv"
            )
            save_evaluation_results(weekly_evals, str(weekly_path))

        # Monthly evaluations
        monthly_evals = [
            e
            for e in all_evaluations
            if e["model"] == model_name and e["aggregation"] == "monthly"
        ]
        if monthly_evals:
            monthly_path = (
                results_dir / "evaluations" / f"{model_key}_monthly_metrics.csv"
            )
            save_evaluation_results(monthly_evals, str(monthly_path))

    # Generate summary report
    summary_path = results_dir / "model_comparison_summary.csv"
    generate_summary_report(all_evaluations, str(summary_path))

    # Save anomaly detection results
    print(f"\n{'=' * 60}")
    print("Saving anomaly detection results...")
    print(f"{'=' * 60}")

    all_anomaly_results = []
    for branch in BRANCHES:
        for aggregation in ["weekly", "monthly"]:
            branch_data = data["train"][aggregation].get(branch, {})
            if "series" in branch_data and "anomaly_labels" in branch_data:
                anomaly_path = (
                    results_dir / "anomalies" / aggregation / f"{branch}_anomalies.csv"
                )
                save_anomaly_detection_results(
                    branch_data["anomaly_labels"],
                    branch_data["anomaly_severity"],
                    branch_data["series"],
                    str(anomaly_path),
                    branch,
                    aggregation,
                )

                # Collect statistics for summary
                labels = branch_data["anomaly_labels"]
                severity = branch_data["anomaly_severity"]
                if len(labels) > 0:
                    anomaly_rate = labels.mean()
                    anomaly_severities = severity[labels == 1]
                    all_anomaly_results.append(
                        {
                            "branch": branch,
                            "aggregation": aggregation,
                            "anomaly_rate": anomaly_rate,
                            "mean_anomaly_severity": anomaly_severities.mean()
                            if len(anomaly_severities) > 0
                            else 0.0,
                            "max_anomaly_severity": anomaly_severities.max()
                            if len(anomaly_severities) > 0
                            else 0.0,
                        }
                    )

    # Generate anomaly summary
    if all_anomaly_results:
        anomaly_summary_path = results_dir / "anomaly_summary.csv"
        generate_anomaly_summary_report(all_anomaly_results, str(anomaly_summary_path))

    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"{'=' * 60}")
    print(f"Results saved to: {results_dir}")
    print(f"Summary report: {summary_path}")
    if all_anomaly_results:
        print(f"Anomaly summary: {anomaly_summary_path}")


if __name__ == "__main__":
    main()
