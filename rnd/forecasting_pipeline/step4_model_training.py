"""
Step 4: Model Training

Trains a LightGBM model with optional hyperparameter optimization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import json
from datetime import datetime

import lightgbm as lgb
from sklearn.model_selection import train_test_split

try:
    from . import config
    from .utils import plot_feature_importance
except ImportError:
    import config
    from utils import plot_feature_importance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional Optuna import
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Hyperparameter optimization will be disabled.")


def prepare_training_data(df: pd.DataFrame, feature_cols: list):
    """
    Prepare training data by removing rows with missing features.
    
    Returns:
        X (features), y (target), and feature names
    """
    logger.info("Preparing training data")
    
    # Remove rows with missing target
    df_clean = df.dropna(subset=[config.TARGET_COL])
    
    # Remove rows with missing critical features (lag features may be NaN for early periods)
    # Keep rows where at least some features are available
    feature_df = df_clean[feature_cols].copy()
    
    # For now, fill NaN in lag/rolling features with 0 or median
    # This is acceptable for early time periods
    for col in feature_cols:
        if feature_df[col].dtype in [np.float64, np.int64]:
            if feature_df[col].isna().any():
                # Fill with 0 for lag features, median for others
                if "lag" in col or "rolling" in col:
                    feature_df[col] = feature_df[col].fillna(0)
                else:
                    feature_df[col] = feature_df[col].fillna(feature_df[col].median())
    
    X = feature_df.values
    y = df_clean[config.TARGET_COL].values
    
    logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Target statistics - Mean: {y.mean():.2f}, Std: {y.std():.2f}, Min: {y.min():.2f}, Max: {y.max():.2f}")
    
    return X, y, feature_cols


def optimize_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list,
    n_trials: int = 50,
    categorical_indices: list = None
) -> dict:
    """Optimize hyperparameters using Optuna."""
    if not OPTUNA_AVAILABLE or not config.USE_OPTUNA:
        return None
    
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
    
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "verbose": -1,
            "random_state": config.RANDOM_STATE,
        }
        
        # Create datasets with categorical features if specified
        if categorical_indices:
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names,
                                    categorical_feature=categorical_indices)
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names,
                                  categorical_feature=categorical_indices, reference=train_data)
        else:
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        predictions = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - predictions) ** 2))
        
        return rmse
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best hyperparameters: {study.best_params}")
    logger.info(f"Best RMSE: {study.best_value:.4f}")
    
    # Update base params with best params (remove categorical_feature if present)
    best_params = config.LIGHTGBM_PARAMS.copy()
    best_params.pop("categorical_feature", None)  # Remove from config params
    best_params.update(study.best_params)
    
    return best_params


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    params: dict = None,
    validation_split: float = 0.2
):
    """
    Train LightGBM model.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        params: Model parameters (uses config defaults if None)
        validation_split: Fraction of data to use for validation
    """
    logger.info("Training LightGBM model")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=config.RANDOM_STATE
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples, Validation set: {X_val.shape[0]} samples")
    
    # Get parameters
    if params is None:
        params = config.LIGHTGBM_PARAMS.copy()
    
    # Remove categorical_feature from params (will specify in Dataset constructor)
    params = params.copy()
    categorical_feature = params.pop("categorical_feature", None)
    
    # Identify categorical feature indices if needed
    categorical_indices = None
    if categorical_feature == "auto" or categorical_feature:
        # Find indices of categorical features (those ending with _encoded)
        categorical_indices = [i for i, name in enumerate(feature_names) 
                              if name.endswith("_encoded") or name in config.CATEGORICAL_COLS]
        if len(categorical_indices) > 0:
            logger.info(f"Identified {len(categorical_indices)} categorical features: {[feature_names[i] for i in categorical_indices]}")
    
    # Optimize hyperparameters if requested
    if config.USE_OPTUNA and OPTUNA_AVAILABLE:
        logger.info("Optimizing hyperparameters...")
        params = optimize_hyperparameters(
            X_train, y_train, X_val, y_val, feature_names, config.OPTUNA_N_TRIALS,
            categorical_indices=categorical_indices
        )
        # Remove categorical_feature from optimized params too
        if "categorical_feature" in params:
            params.pop("categorical_feature")
    
    # Create datasets with categorical feature specification
    if categorical_indices:
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, 
                                categorical_feature=categorical_indices)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, 
                              categorical_feature=categorical_indices, reference=train_data)
    else:
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
    
    # Train model
    logger.info("Training model...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Evaluate on validation set
    val_predictions = model.predict(X_val)
    val_rmse = np.sqrt(np.mean((y_val - val_predictions) ** 2))
    val_mae = np.mean(np.abs(y_val - val_predictions))
    
    logger.info(f"Validation RMSE: {val_rmse:.4f}")
    logger.info(f"Validation MAE: {val_mae:.4f}")
    
    return model, params


def save_model(model: lgb.Booster, params: dict, feature_names: list, model_dir: Path):
    """Save trained model with metadata."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = model_dir / f"lightgbm_model_{timestamp}.pkl"
    
    logger.info(f"Saving model to {model_path}")
    
    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata = {
        "model_type": "lightgbm",
        "timestamp": timestamp,
        "parameters": params,
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "config": {
            "peak_season_months": config.PEAK_SEASON_MONTHS,
            "lag_periods": config.LAG_PERIODS,
            "rolling_window": config.ROLLING_WINDOW,
        }
    }
    
    metadata_path = model_dir / f"model_metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Saved model metadata to {metadata_path}")
    
    # Create symlink to latest model
    latest_model_path = model_dir / "latest_model.pkl"
    if latest_model_path.exists():
        latest_model_path.unlink()
    latest_model_path.symlink_to(model_path.name)
    
    return model_path, metadata_path


def main():
    """Main function to run Step 4."""
    logger.info("=" * 50)
    logger.info("STEP 4: Model Training")
    logger.info("=" * 50)
    
    # Load cleaned data
    logger.info(f"Loading cleaned data from {config.CLEANED_DATA_FILE}")
    df = pd.read_parquet(config.CLEANED_DATA_FILE)
    logger.info(f"Loaded {len(df)} rows")
    
    # Load feature list
    feature_list_path = config.OUTPUT_DIR / "feature_list.txt"
    if feature_list_path.exists():
        with open(feature_list_path, "r") as f:
            feature_cols = [line.strip() for line in f if line.strip()]
    else:
        # Fallback: get features from dataframe
        exclude_cols = [config.DATE_COL, config.TARGET_COL, "is_outlier", f"{config.TARGET_COL}_original"]
        exclude_cols.extend([f"{col}_encoded" not in col for col in config.CATEGORICAL_COLS])
        feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith("_")]
        # Ensure we have encoded categorical columns
        for col in config.CATEGORICAL_COLS:
            encoded_col = f"{col}_encoded"
            if encoded_col in df.columns and encoded_col not in feature_cols:
                feature_cols.append(encoded_col)
    
    logger.info(f"Using {len(feature_cols)} features")
    
    # Prepare training data
    X, y, feature_names = prepare_training_data(df, feature_cols)
    
    # Train model
    model, params = train_model(X, y, feature_names)
    
    # Plot feature importance
    importance_path = config.OUTPUT_DIR / "feature_importance.png"
    plot_feature_importance(model, feature_names, top_n=30, save_path=importance_path)
    
    # Save model
    model_path, metadata_path = save_model(model, params, feature_names, config.MODEL_DIR)
    
    logger.info("Step 4 completed successfully!")
    return model, params, feature_names


if __name__ == "__main__":
    main()

