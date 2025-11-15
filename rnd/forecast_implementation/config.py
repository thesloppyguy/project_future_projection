"""
Configuration file for comprehensive forecasting implementation.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "forecast_outputs"
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Data file
DATA_FILE = DATA_DIR / "merged_filter_ingestion_latest.csv"

# Date splits
TRAIN_END_DATE = "2024-03-31"  # End of training period
CALIBRATION_START_DATE = "2024-04-01"  # Start of calibration period
CALIBRATION_END_DATE = "2025-03-31"  # End of calibration period
BLIND_START_DATE = "2025-04-01"  # Start of blind evaluation period
BLIND_END_DATE = "2025-05-31"  # End of available blind data

# Missing month to handle
MISSING_MONTH = "2020-04"

# Forecast horizon
FORECAST_HORIZON_MONTHS = 12

# Aggregation levels
AGGREGATION_LEVELS = ["combined", "branch_wise"]

# Model configuration
MODELS_TO_TRAIN = [
    "lightgbm",
    "xgboost",
    "catboost",
    "prophet",
    "auto_arima",
    "holt_winters",
    "random_forest",
    "quantile_regression",
    "stl_decomposition",
    "sarimax",
    # "lstm",  # Uncomment if you want deep learning models
]

# Ensemble configuration
USE_ENSEMBLE = True
ENSEMBLE_METHOD = "weighted_average"  # "average", "weighted_average", "median"
ENSEMBLE_MAX_MAPE_THRESHOLD = 100.0  # Filter out models with MAPE > this value
ENSEMBLE_MIN_MODELS = 2  # Minimum models required for ensemble

# Rolling forecast configuration
ROLLING_WINDOW_MONTHS = 1  # Update forecast every month during calibration
MIN_TRAIN_MONTHS = 12  # Minimum months required for training

# Anomaly detection
ENABLE_ANOMALY_DETECTION = True
ANOMALY_CONTAMINATION = 0.1  # Expected proportion of anomalies
# Uses rolling statistics (rolling mean/std) with z-score threshold
# Anomalies are detected as points deviating >2.5 std from rolling mean
# This accounts for trends and seasonality in time series data

# Feature engineering
LAG_PERIODS = [1, 2, 3, 6, 12]  # Months
ROLLING_WINDOWS = [3, 6, 12]  # Months
INCLUDE_SEASONALITY = True
INCLUDE_TREND = True

# Evaluation metrics
METRICS = ["rmse", "mae", "mape", "nrmse", "r2"]

# Hyperparameter tuning
ENABLE_HYPERPARAMETER_TUNING = (
    True  # Set to True to enable tuning (slower but better performance)
)
HYPERPARAMETER_TUNING_TRIALS = 50  # Number of trials for Optuna (if available)

# Branch-specific model selection
ENABLE_BRANCH_MODEL_SELECTION = True  # Select best models per branch
BRANCH_TOP_N_MODELS = 3  # Number of top models to select per branch

# Random seed
RANDOM_STATE = 42
