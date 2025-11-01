"""
Step 5: Validation

Performs time series cross-validation and trains final model on all data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import json
from datetime import datetime

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from . import config
    from .utils import calculate_metrics, plot_validation_results
except ImportError:
    import config
    from utils import calculate_metrics, plot_validation_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_fold_data(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    feature_cols: list,
    date_col: str = "Date"
):
    """Prepare train and validation data for a fold."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Filter data
    train_mask = (df[date_col] >= train_start) & (df[date_col] <= train_end)
    val_mask = (df[date_col] >= val_start) & (df[date_col] <= val_end)
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    
    logger.info(f"Train period: {train_start} to {train_end} ({len(train_df)} rows)")
    logger.info(f"Validation period: {val_start} to {val_end} ({len(val_df)} rows)")
    
    # Prepare features
    def prepare_features(df_subset):
        X = df_subset[feature_cols].copy()
        # Fill NaN
        for col in feature_cols:
            if X[col].dtype in [np.float64, np.int64]:
                if X[col].isna().any():
                    if "lag" in col or "rolling" in col:
                        X[col] = X[col].fillna(0)
                    else:
                        X[col] = X[col].fillna(X[col].median())
        return X.values
    
    X_train = prepare_features(train_df)
    y_train = train_df[config.TARGET_COL].values
    X_val = prepare_features(val_df)
    y_val = val_df[config.TARGET_COL].values
    
    return X_train, y_train, X_val, y_val


def train_fold_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list,
    params: dict = None
) -> tuple:
    """Train model for a validation fold."""
    if params is None:
        params = config.LIGHTGBM_PARAMS.copy()
    
    # Remove categorical_feature from params and handle it separately
    params = params.copy()
    categorical_feature = params.pop("categorical_feature", None)
    
    # Identify categorical feature indices
    categorical_indices = None
    if categorical_feature == "auto" or categorical_feature:
        categorical_indices = [i for i, name in enumerate(feature_names) 
                              if name.endswith("_encoded") or name in config.CATEGORICAL_COLS]
    
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
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(0)
        ]
    )
    
    predictions = model.predict(X_val)
    metrics = calculate_metrics(y_val, predictions)
    
    return model, predictions, metrics


def run_validation(
    df: pd.DataFrame,
    feature_cols: list,
    folds: list
) -> dict:
    """Run time series cross-validation."""
    logger.info("Running time series cross-validation")
    
    results = {}
    
    for fold_idx, (train_start, train_end, val_start, val_end) in enumerate(folds, 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold_idx}")
        logger.info(f"{'='*50}")
        
        # Prepare fold data
        X_train, y_train, X_val, y_val = prepare_fold_data(
            df, train_start, train_end, val_start, val_end, feature_cols
        )
        
        # Train model
        model, predictions, metrics = train_fold_model(
            X_train, y_train, X_val, y_val, feature_cols
        )
        
        # Store results
        results[f"fold_{fold_idx}"] = {
            "train_period": (train_start, train_end),
            "val_period": (val_start, val_end),
            "metrics": metrics,
            "predictions": predictions.tolist(),
            "actuals": y_val.tolist()
        }
        
        logger.info(f"Fold {fold_idx} Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        # Plot validation results
        plot_path = config.OUTPUT_DIR / f"validation_fold_{fold_idx}.png"
        plot_validation_results(
            y_val, predictions,
            title=f"Validation Results - Fold {fold_idx}",
            save_path=plot_path
        )
    
    return results


def train_final_model(
    df: pd.DataFrame,
    feature_cols: list,
    params: dict = None
) -> lgb.Booster:
    """Train final model on all available data."""
    logger.info("\n" + "="*50)
    logger.info("Training final model on all data")
    logger.info("="*50)
    
    if params is None:
        params = config.LIGHTGBM_PARAMS.copy()
    
    # Remove categorical_feature from params and handle it separately
    params = params.copy()
    categorical_feature = params.pop("categorical_feature", None)
    
    # Identify categorical feature indices
    categorical_indices = None
    if categorical_feature == "auto" or categorical_feature:
        categorical_indices = [i for i, name in enumerate(feature_cols) 
                              if name.endswith("_encoded") or name in config.CATEGORICAL_COLS]
    
    # Prepare all data
    X = df[feature_cols].copy()
    for col in feature_cols:
        if X[col].dtype in [np.float64, np.int64]:
            if X[col].isna().any():
                if "lag" in col or "rolling" in col:
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(X[col].median())
    
    X_all = X.values
    y_all = df[config.TARGET_COL].values
    
    logger.info(f"Training on {len(X_all)} samples")
    
    # Split for early stopping (use last 20% as validation)
    split_idx = int(len(X_all) * 0.8)
    X_train = X_all[:split_idx]
    y_train = y_all[:split_idx]
    X_val = X_all[split_idx:]
    y_val = y_all[split_idx:]
    
    # Create datasets with categorical features if specified
    if categorical_indices:
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols,
                                categorical_feature=categorical_indices)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols,
                              categorical_feature=categorical_indices, reference=train_data)
    else:
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_data)
    
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
    
    # Final evaluation
    val_predictions = model.predict(X_val)
    final_metrics = calculate_metrics(y_val, val_predictions)
    
    logger.info("Final model metrics (on validation split):")
    for metric, value in final_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    return model


def save_validation_results(results: dict, output_dir: Path):
    """Save validation results to JSON."""
    results_path = output_dir / "validation_results.json"
    
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for fold_key, fold_results in results.items():
        results_serializable[fold_key] = {
            "train_period": fold_results["train_period"],
            "val_period": fold_results["val_period"],
            "metrics": fold_results["metrics"]
            # Exclude predictions and actuals from JSON (too large)
        }
    
    with open(results_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"Saved validation results to {results_path}")
    
    # Also save summary CSV
    summary_data = []
    for fold_key, fold_results in results.items():
        summary_data.append({
            "fold": fold_key,
            **fold_results["metrics"]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / "validation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved validation summary to {summary_path}")


def main():
    """Main function to run Step 5."""
    logger.info("=" * 50)
    logger.info("STEP 5: Validation")
    logger.info("=" * 50)
    
    # Load cleaned data
    logger.info(f"Loading cleaned data from {config.CLEANED_DATA_FILE}")
    df = pd.read_parquet(config.CLEANED_DATA_FILE)
    logger.info(f"Loaded {len(df)} rows")
    
    # Ensure date is datetime
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    
    # Load feature list
    feature_list_path = config.OUTPUT_DIR / "feature_list.txt"
    if feature_list_path.exists():
        with open(feature_list_path, "r") as f:
            feature_cols = [line.strip() for line in f if line.strip()]
    else:
        # Fallback
        exclude_cols = [config.DATE_COL, config.TARGET_COL, "is_outlier", f"{config.TARGET_COL}_original"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        for col in config.CATEGORICAL_COLS:
            encoded_col = f"{col}_encoded"
            if encoded_col in df.columns:
                feature_cols.append(encoded_col)
    
    logger.info(f"Using {len(feature_cols)} features")
    
    # Run validation
    validation_results = run_validation(df, feature_cols, config.VALIDATION_FOLDS)
    
    # Save validation results
    save_validation_results(validation_results, config.OUTPUT_DIR)
    
    # Train final model on all data
    final_model = train_final_model(df, feature_cols)
    
    # Save final model
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_model_path = config.MODEL_DIR / f"final_model_{timestamp}.pkl"
    
    with open(final_model_path, "wb") as f:
        pickle.dump(final_model, f)
    
    # Create symlink
    latest_model_path = config.MODEL_DIR / "latest_final_model.pkl"
    if latest_model_path.exists():
        latest_model_path.unlink()
    latest_model_path.symlink_to(final_model_path.name)
    
    logger.info(f"Saved final model to {final_model_path}")
    logger.info("Step 5 completed successfully!")
    
    return final_model, validation_results


if __name__ == "__main__":
    main()

