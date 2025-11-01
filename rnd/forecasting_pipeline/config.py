"""
Configuration file for the HVAC forecasting pipeline.
"""

from pathlib import Path
from typing import List, Tuple

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "forecasting_pipeline" / "outputs"
MODEL_DIR = PROJECT_ROOT / "forecasting_pipeline" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
SOURCE_DATA_FILE = DATA_DIR / "merged_filter_ingestion.csv"
AGGREGATED_DATA_FILE = OUTPUT_DIR / "aggregated_weekly_data.parquet"
FEATURED_DATA_FILE = OUTPUT_DIR / "featured_data.parquet"
CLEANED_DATA_FILE = OUTPUT_DIR / "cleaned_data.parquet"
FORECAST_FILE = OUTPUT_DIR / "forecast_52_weeks.csv"

# Aggregation settings
AGGREGATION_FREQ = "W-MON"  # Weekly aggregation starting Monday
EXCLUDE_TONNAGE = False  # Set to True to aggregate only by Branch (excludes Tonnage from combination)

# Set GROUP_BY_COLS based on EXCLUDE_TONNAGE setting
if EXCLUDE_TONNAGE:
    GROUP_BY_COLS = ["Date", "Branch"]
else:
    GROUP_BY_COLS = ["Date", "Branch", "Tonnage"]

# Feature engineering settings
PEAK_SEASON_MONTHS = [5, 6, 7, 8]  # May-August (1-indexed)
LAG_PERIODS = [1, 4, 52]  # 1 week, 4 weeks, 52 weeks
ROLLING_WINDOW = 4  # 4-week rolling window
ROLLING_STATS = ["mean", "std", "max"]

# Outlier detection settings
OUTLIER_METHOD = "isolation_forest"  # Options: "isolation_forest", "lof", "percentile"
WINSORIZE_PERCENTILE = 99  # Cap values at 99th percentile
OUTLIER_CONTAMINATION = 0.05  # Expected proportion of outliers (for Isolation Forest)

# Model settings
MODEL_TYPE = "lightgbm"
RANDOM_STATE = 42

# Model selection
INCLUDE_DEEP_LEARNING = True  # Set to False to exclude LSTM, GRU, and Simple RNN models
# When False, only tree-based models (LightGBM, XGBoost, CatBoost), Prophet, and Ensemble are used

# Hyperparameter optimization settings
USE_OPTUNA = False  # Set to True to enable hyperparameter optimization for all models
OPTUNA_N_TRIALS = 50  # Number of optimization trials per model
# Note: Hyperparameter optimization is available for:
#   - LightGBM: num_leaves, learning_rate, feature_fraction, bagging_fraction, etc.
#   - XGBoost: max_depth, learning_rate, subsample, colsample_bytree, etc.
#   - CatBoost: depth, learning_rate, l2_leaf_reg, bagging_temperature, etc.
#   - Prophet: seasonality_mode, changepoint_prior_scale, seasonality_prior_scale, etc.
#   - Neural networks: Not optimized (use fixed architectures for speed)

# LightGBM default parameters
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": RANDOM_STATE,
    "categorical_feature": "auto",  # Auto-detect categorical features
}

# Validation settings
VALIDATION_METHOD = "rolling_origin"  # Time series cross-validation
VALIDATION_FOLDS = [
    ("2019-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),  # Fold 1
    ("2019-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),  # Fold 2
]

# Forecasting settings
FORECAST_HORIZON_WEEKS = 52
FORECAST_START_DATE = None  # Will use latest date in data if None

# Evaluation metrics
METRICS = ["mae", "rmse", "mape", "r2", "nrmse_mean", "nrmse_range", "nrmse_std"]

# Categorical columns (features) - automatically adjusted based on EXCLUDE_TONNAGE
# Set CATEGORICAL_COLS based on EXCLUDE_TONNAGE setting
if EXCLUDE_TONNAGE:
    CATEGORICAL_COLS = ["Branch"]
else:
    CATEGORICAL_COLS = ["Branch", "Tonnage"]

# Date column
DATE_COL = "Date"
TARGET_COL = "Quantity"

