"""
Step 5: Multi-Model Validation & King of the Hill Comparison

Validates all models using time series cross-validation and ranks them.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from . import config
    from .utils import calculate_metrics, plot_predicted_vs_actual
    from .step4_multi_model_training import (
        train_lightgbm, train_xgboost, train_catboost,
        train_lstm, train_gru, train_simple_rnn,
        create_sequences, get_categorical_indices
    )
except ImportError:
    import config
    from utils import calculate_metrics, plot_predicted_vs_actual
    from step4_multi_model_training import (
        train_lightgbm, train_xgboost, train_catboost,
        train_lstm, train_gru, train_simple_rnn,
        create_sequences, get_categorical_indices
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Neural network models will be skipped.")

np.random.seed(config.RANDOM_STATE)
if TENSORFLOW_AVAILABLE:
    tf.random.set_seed(config.RANDOM_STATE)


def prepare_fold_data(df: pd.DataFrame, train_start: str, train_end: str,
                     val_start: str, val_end: str, feature_cols: list):
    """Prepare train and validation data for a fold."""
    df = df.copy()
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    
    train_mask = (df[config.DATE_COL] >= train_start) & (df[config.DATE_COL] <= train_end)
    val_mask = (df[config.DATE_COL] >= val_start) & (df[config.DATE_COL] <= val_end)
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    
    logger.info(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")
    
    def prepare_features(df_subset):
        X = df_subset[feature_cols].copy()
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
    
    return X_train, y_train, X_val, y_val, train_df, val_df


def validate_tree_model(model_func, model_name: str, X_train, y_train, X_val, y_val, feature_names, optimized_params=None):
    """Validate a tree-based model."""
    try:
        # Pass optimized_params if available
        if optimized_params is not None:
            model = model_func(X_train, y_train, X_val, y_val, feature_names, optimized_params)
        else:
            model = model_func(X_train, y_train, X_val, y_val, feature_names)
        predictions = model.predict(X_val)
        predictions = np.maximum(predictions, 0)  # Ensure non-negative
        return predictions, model
    except Exception as e:
        logger.error(f"{model_name} validation failed: {e}")
        return None, None


def validate_prophet(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Validate Prophet model."""
    try:
        # Aggregate by date for Prophet
        train_df[config.DATE_COL] = pd.to_datetime(train_df[config.DATE_COL])
        val_df[config.DATE_COL] = pd.to_datetime(val_df[config.DATE_COL])
        
        prophet_train = train_df.groupby(config.DATE_COL)[config.TARGET_COL].sum().reset_index()
        prophet_train.columns = ['ds', 'y']
        
        prophet_val = val_df.groupby(config.DATE_COL)[config.TARGET_COL].sum().reset_index()
        prophet_val.columns = ['ds', 'y']
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                       daily_seasonality=False, seasonality_mode='multiplicative')
        model.fit(prophet_train)
        
        # Forecast validation period
        future = model.make_future_dataframe(periods=len(prophet_val))
        forecast = model.predict(future)
        
        # Extract validation predictions (last len(val_df) predictions)
        val_forecast = forecast.tail(len(prophet_val))['yhat'].values
        
        # For panel data, we need to distribute predictions
        # Simple approach: use same value for all combinations in each week
        predictions = np.repeat(val_forecast, len(val_df) // len(prophet_val) + 1)[:len(val_df)]
        predictions = np.maximum(predictions, 0)
        
        return predictions, model
    except Exception as e:
        logger.error(f"Prophet validation failed: {e}")
        return None, None


def validate_neural_network(model_func, model_name: str, X_train, y_train, X_val, y_val, lookback=12):
    """Validate a neural network model."""
    try:
        model = model_func(X_train, y_train, X_val, y_val, lookback)
        
        # Prepare validation data
        scaler_X = model.scaler_X
        scaler_y = model.scaler_y
        
        X_val_scaled = scaler_X.transform(X_val)
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        
        # Create sequences (use last lookback points for each prediction)
        # For simplicity, we'll use a sliding window approach
        if len(X_val_scaled) < lookback:
            # Pad if needed
            X_val_padded = np.vstack([X_train[-lookback:], X_val_scaled])
        else:
            X_val_padded = X_val_scaled
        
        predictions_scaled = []
        for i in range(len(X_val_scaled)):
            if i + lookback <= len(X_val_padded):
                seq = X_val_padded[i:i+lookback].reshape(1, lookback, -1)
                pred = model.predict(seq, verbose=0)[0, 0]
                predictions_scaled.append(pred)
            else:
                # Use last known value
                predictions_scaled.append(predictions_scaled[-1] if predictions_scaled else 0)
        
        predictions_scaled = np.array(predictions_scaled)
        predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
        predictions = np.maximum(predictions, 0)
        
        return predictions, model
    except Exception as e:
        logger.error(f"{model_name} validation failed: {e}")
        return None, None


def create_ensemble_predictions(all_predictions: Dict[str, np.ndarray]) -> np.ndarray:
    """Create ensemble prediction as average of all available models."""
    valid_predictions = {k: v for k, v in all_predictions.items() if v is not None}
    
    if not valid_predictions:
        return None
    
    # Average predictions
    ensemble_pred = np.mean(list(valid_predictions.values()), axis=0)
    ensemble_pred = np.maximum(ensemble_pred, 0)
    
    return ensemble_pred


def run_multi_model_validation(df: pd.DataFrame, feature_cols: list, folds: list, optimize_hyperparams: bool = False) -> Dict:
    """Run validation for all models."""
    logger.info("Running multi-model validation")
    
    all_results = {}
    # Store optimized params per fold
    fold_optimized_params = {}
    
    for fold_idx, (train_start, train_end, val_start, val_end) in enumerate(folds, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"FOLD {fold_idx}")
        logger.info(f"{'='*70}")
        
        # Prepare fold data
        X_train, y_train, X_val, y_val, train_df, val_df = prepare_fold_data(
            df, train_start, train_end, val_start, val_end, feature_cols
        )
        
        # Optimize hyperparameters for this fold if requested
        optimized_params = None
        if optimize_hyperparams and config.USE_OPTUNA:
            try:
                from .hyperparameter_optimization import optimize_all_models
            except ImportError:
                from hyperparameter_optimization import optimize_all_models
            
            logger.info(f"\n--- Optimizing hyperparameters for Fold {fold_idx} ---")
            categorical_indices = get_categorical_indices(feature_cols)
            optimized_params = optimize_all_models(
                X_train, y_train, X_val, y_val, feature_cols, categorical_indices,
                n_trials=min(20, config.OPTUNA_N_TRIALS)  # Use fewer trials per fold
            )
            fold_optimized_params[f"fold_{fold_idx}"] = optimized_params
        
        fold_results = {}
        all_predictions = {}
        
        # Tree-based models
        logger.info("\n--- Tree-based Models ---")
        
        pred, model = validate_tree_model(train_lightgbm, "LightGBM", 
                                         X_train, y_train, X_val, y_val, feature_cols, optimized_params)
        if pred is not None:
            metrics = calculate_metrics(y_val, pred)
            fold_results['lightgbm'] = metrics
            all_predictions['lightgbm'] = pred
            logger.info(f"LightGBM - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}, NRMSE: {metrics.get('nrmse_mean', 0):.2f}%")
        
        pred, model = validate_tree_model(train_xgboost, "XGBoost",
                                         X_train, y_train, X_val, y_val, feature_cols, optimized_params)
        if pred is not None:
            metrics = calculate_metrics(y_val, pred)
            fold_results['xgboost'] = metrics
            all_predictions['xgboost'] = pred
            logger.info(f"XGBoost - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}, NRMSE: {metrics.get('nrmse_mean', 0):.2f}%")
        
        pred, model = validate_tree_model(train_catboost, "CatBoost",
                                         X_train, y_train, X_val, y_val, feature_cols, optimized_params)
        if pred is not None:
            metrics = calculate_metrics(y_val, pred)
            fold_results['catboost'] = metrics
            all_predictions['catboost'] = pred
            logger.info(f"CatBoost - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}, NRMSE: {metrics.get('nrmse_mean', 0):.2f}%")
        
        # Prophet
        logger.info("\n--- Prophet ---")
        pred, model = validate_prophet(train_df, val_df)
        if pred is not None and len(pred) == len(y_val):
            metrics = calculate_metrics(y_val, pred)
            fold_results['prophet'] = metrics
            all_predictions['prophet'] = pred
            logger.info(f"Prophet - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}, NRMSE: {metrics.get('nrmse_mean', 0):.2f}%")
        
        # Neural Networks
        include_deep_learning = getattr(config, 'INCLUDE_DEEP_LEARNING', True)
        
        if not include_deep_learning:
            logger.info("\n--- Neural Networks ---")
            logger.info("Skipping deep learning models (LSTM, GRU, SimpleRNN) - disabled in config")
        elif TENSORFLOW_AVAILABLE and len(X_train) > 12:  # Need enough data for sequences
            logger.info("\n--- Neural Networks ---")
            pred, model = validate_neural_network(train_lstm, "LSTM",
                                                  X_train, y_train, X_val, y_val)
            if pred is not None and len(pred) == len(y_val):
                metrics = calculate_metrics(y_val, pred)
                fold_results['lstm'] = metrics
                all_predictions['lstm'] = pred
                logger.info(f"LSTM - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}, NRMSE: {metrics.get('nrmse_mean', 0):.2f}%")
            
            pred, model = validate_neural_network(train_gru, "GRU",
                                                  X_train, y_train, X_val, y_val)
            if pred is not None and len(pred) == len(y_val):
                metrics = calculate_metrics(y_val, pred)
                fold_results['gru'] = metrics
                all_predictions['gru'] = pred
                logger.info(f"GRU - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}, NRMSE: {metrics.get('nrmse_mean', 0):.2f}%")
            
            pred, model = validate_neural_network(train_simple_rnn, "SimpleRNN",
                                                  X_train, y_train, X_val, y_val)
            if pred is not None and len(pred) == len(y_val):
                metrics = calculate_metrics(y_val, pred)
                fold_results['simple_rnn'] = metrics
                all_predictions['simple_rnn'] = pred
                logger.info(f"SimpleRNN - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}, NRMSE: {metrics.get('nrmse_mean', 0):.2f}%")
        elif not TENSORFLOW_AVAILABLE:
            logger.info("\n--- Neural Networks ---")
            logger.warning("TensorFlow not available. Skipping neural network models.")
        
        # Ensemble
        logger.info("\n--- Ensemble ---")
        ensemble_pred = create_ensemble_predictions(all_predictions)
        if ensemble_pred is not None and len(ensemble_pred) == len(y_val):
            metrics = calculate_metrics(y_val, ensemble_pred)
            fold_results['ensemble'] = metrics
            all_predictions['ensemble'] = ensemble_pred
            logger.info(f"Ensemble - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}, NRMSE: {metrics.get('nrmse_mean', 0):.2f}%")
        
        # Store dates for time series plotting
        val_dates = val_df[config.DATE_COL].values.tolist()
        
        all_results[f"fold_{fold_idx}"] = {
            "train_period": (train_start, train_end),
            "val_period": (val_start, val_end),
            "models": fold_results,
            "actuals": y_val.tolist(),  # Store actuals for plotting
            "dates": val_dates,  # Store dates for time series plotting
            "predictions": {model_name: pred.tolist() 
                          for model_name, pred in all_predictions.items() if pred is not None}
        }
    
    return all_results


def calculate_king_of_hill(results: Dict) -> pd.DataFrame:
    """Calculate rankings and determine 'King of the Hill'."""
    logger.info("\n" + "="*70)
    logger.info("KING OF THE HILL - Model Rankings")
    logger.info("="*70)
    
    # Aggregate metrics across folds
    model_metrics = {}
    
    for fold_key, fold_data in results.items():
        for model_name, metrics in fold_data["models"].items():
            if model_name not in model_metrics:
                model_metrics[model_name] = {
                    'mae': [], 'rmse': [], 'mape': [], 'r2': [],
                    'nrmse_mean': [], 'nrmse_range': [], 'nrmse_std': []
                }
            model_metrics[model_name]['mae'].append(metrics['mae'])
            model_metrics[model_name]['rmse'].append(metrics['rmse'])
            model_metrics[model_name]['mape'].append(metrics['mape'])
            model_metrics[model_name]['r2'].append(metrics['r2'])
            # Add NRMSE metrics if available
            if 'nrmse_mean' in metrics:
                model_metrics[model_name]['nrmse_mean'].append(metrics['nrmse_mean'])
            if 'nrmse_range' in metrics:
                model_metrics[model_name]['nrmse_range'].append(metrics['nrmse_range'])
            if 'nrmse_std' in metrics:
                model_metrics[model_name]['nrmse_std'].append(metrics['nrmse_std'])
    
    # Calculate averages
    rankings = []
    for model_name, metrics in model_metrics.items():
        ranking_dict = {
            'Model': model_name,
            'MAE (mean)': np.mean(metrics['mae']),
            'MAE (std)': np.std(metrics['mae']),
            'RMSE (mean)': np.mean(metrics['rmse']),
            'RMSE (std)': np.std(metrics['rmse']),
            'MAPE (mean)': np.mean(metrics['mape']),
            'MAPE (std)': np.std(metrics['mape']),
            'R² (mean)': np.mean(metrics['r2']),
            'R² (std)': np.std(metrics['r2']),
            'Folds': len(metrics['mae'])
        }
        # Add NRMSE if available
        if len(metrics.get('nrmse_mean', [])) > 0:
            ranking_dict['NRMSE_mean % (mean)'] = np.mean(metrics['nrmse_mean'])
            ranking_dict['NRMSE_mean % (std)'] = np.std(metrics['nrmse_mean'])
        if len(metrics.get('nrmse_range', [])) > 0:
            ranking_dict['NRMSE_range % (mean)'] = np.mean(metrics['nrmse_range'])
            ranking_dict['NRMSE_range % (std)'] = np.std(metrics['nrmse_range'])
        if len(metrics.get('nrmse_std', [])) > 0:
            ranking_dict['NRMSE_std % (mean)'] = np.mean(metrics['nrmse_std'])
            ranking_dict['NRMSE_std % (std)'] = np.std(metrics['nrmse_std'])
        
        rankings.append(ranking_dict)
    
    rankings_df = pd.DataFrame(rankings)
    
    # Rank by RMSE (primary), then MAE, then R²
    rankings_df['Rank_RMSE'] = rankings_df['RMSE (mean)'].rank(ascending=True)
    rankings_df['Rank_MAE'] = rankings_df['MAE (mean)'].rank(ascending=True)
    rankings_df['Rank_R2'] = rankings_df['R² (mean)'].rank(ascending=False)
    
    # Overall rank (lower is better)
    rankings_df['Overall_Rank'] = (
        rankings_df['Rank_RMSE'] * 0.5 +
        rankings_df['Rank_MAE'] * 0.3 +
        (len(rankings_df) + 1 - rankings_df['Rank_R2']) * 0.2
    ).rank(ascending=True)
    
    rankings_df = rankings_df.sort_values('Overall_Rank')
    
    # Print rankings
    logger.info("\nModel Rankings (Lower Overall Rank = Better):")
    logger.info("="*70)
    for idx, row in rankings_df.iterrows():
        nrmse_str = ""
        if 'NRMSE_mean % (mean)' in row and pd.notna(row['NRMSE_mean % (mean)']):
            nrmse_str = f" | NRMSE: {row['NRMSE_mean % (mean)']:>6.2f}%"
        logger.info(f"{int(row['Overall_Rank'])}. {row['Model']:<15} "
                   f"RMSE: {row['RMSE (mean)']:>8.2f} (±{row['RMSE (std)']:.2f}) | "
                   f"MAE: {row['MAE (mean)']:>8.2f} (±{row['MAE (std)']:.2f}) | "
                   f"R²: {row['R² (mean)']:>6.4f} (±{row['R² (std)']:.4f}){nrmse_str}")
    
    king = rankings_df.iloc[0]
    logger.info("\n" + "="*70)
    logger.info(f"🏆 KING OF THE HILL: {king['Model'].upper()}")
    logger.info(f"   RMSE: {king['RMSE (mean)']:.2f} (±{king['RMSE (std)']:.2f})")
    logger.info(f"   MAE: {king['MAE (mean)']:.2f} (±{king['MAE (std)']:.2f})")
    logger.info(f"   R²: {king['R² (mean)']:.4f} (±{king['R² (std)']:.4f})")
    if 'NRMSE_mean % (mean)' in king and pd.notna(king['NRMSE_mean % (mean)']):
        logger.info(f"   NRMSE: {king['NRMSE_mean % (mean)']:.2f}% (±{king['NRMSE_mean % (std)']:.2f}%)")
    logger.info("="*70)
    
    return rankings_df


def plot_model_comparison(rankings_df: pd.DataFrame, save_path: Path):
    """Plot model comparison charts."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # RMSE comparison
    ax1 = axes[0, 0]
    rankings_df_sorted = rankings_df.sort_values('RMSE (mean)')
    ax1.barh(rankings_df_sorted['Model'], rankings_df_sorted['RMSE (mean)'], 
            xerr=rankings_df_sorted['RMSE (std)'], capsize=5)
    ax1.set_xlabel('RMSE')
    ax1.set_title('RMSE Comparison (Lower is Better)')
    ax1.invert_yaxis()
    
    # MAE comparison
    ax2 = axes[0, 1]
    rankings_df_sorted = rankings_df.sort_values('MAE (mean)')
    ax2.barh(rankings_df_sorted['Model'], rankings_df_sorted['MAE (mean)'],
            xerr=rankings_df_sorted['MAE (std)'], capsize=5)
    ax2.set_xlabel('MAE')
    ax2.set_title('MAE Comparison (Lower is Better)')
    ax2.invert_yaxis()
    
    # R² comparison
    ax3 = axes[1, 0]
    rankings_df_sorted = rankings_df.sort_values('R² (mean)', ascending=False)
    ax3.barh(rankings_df_sorted['Model'], rankings_df_sorted['R² (mean)'],
            xerr=rankings_df_sorted['R² (std)'], capsize=5)
    ax3.set_xlabel('R²')
    ax3.set_title('R² Comparison (Higher is Better)')
    ax3.invert_yaxis()
    
    # Overall rank
    ax4 = axes[1, 1]
    rankings_df_sorted = rankings_df.sort_values('Overall_Rank')
    colors = ['gold' if i == 0 else 'lightblue' for i in range(len(rankings_df_sorted))]
    ax4.barh(rankings_df_sorted['Model'], rankings_df_sorted['Overall_Rank'], color=colors)
    ax4.set_xlabel('Overall Rank')
    ax4.set_title('Overall Ranking (Lower is Better)')
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison plots to {save_path}")


def main():
    """Main function to run Step 5."""
    logger.info("=" * 70)
    logger.info("STEP 5: Multi-Model Validation & King of the Hill")
    logger.info("=" * 70)
    
    # Load cleaned data
    logger.info(f"Loading cleaned data from {config.CLEANED_DATA_FILE}")
    df = pd.read_parquet(config.CLEANED_DATA_FILE)
    logger.info(f"Loaded {len(df)} rows")
    
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    
    # Load feature list
    feature_list_path = config.OUTPUT_DIR / "feature_list.txt"
    if feature_list_path.exists():
        with open(feature_list_path, "r") as f:
            feature_cols = [line.strip() for line in f if line.strip()]
    else:
        exclude_cols = [config.DATE_COL, config.TARGET_COL, "is_outlier", f"{config.TARGET_COL}_original"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        for col in config.CATEGORICAL_COLS:
            encoded_col = f"{col}_encoded"
            if encoded_col in df.columns:
                feature_cols.append(encoded_col)
    
    logger.info(f"Using {len(feature_cols)} features")
    
    # Check if hyperparameter optimization should be enabled
    optimize = config.USE_OPTUNA if hasattr(config, 'USE_OPTUNA') else False
    
    # Check if deep learning models should be included
    include_deep_learning = getattr(config, 'INCLUDE_DEEP_LEARNING', True)
    if not include_deep_learning:
        logger.info("Deep learning models (LSTM, GRU, SimpleRNN) will be EXCLUDED")
    
    # Run validation
    results = run_multi_model_validation(df, feature_cols, config.VALIDATION_FOLDS, 
                                        optimize_hyperparams=optimize)
    
    # Save results
    results_path = config.OUTPUT_DIR / "multi_model_validation_results.json"
    results_serializable = {}
    for fold_key, fold_data in results.items():
        results_serializable[fold_key] = {
            "train_period": fold_data["train_period"],
            "val_period": fold_data["val_period"],
            "models": {
                k: {
                    m: float(v) if not (isinstance(v, float) and np.isnan(v)) else None
                    for m, v in metrics.items()
                }
                for k, metrics in fold_data["models"].items()
            }
        }
    
    with open(results_path, "w") as f:
        json.dump(results_serializable, f, indent=2, allow_nan=False)
    logger.info(f"Saved validation results to {results_path}")
    
    # Calculate rankings
    rankings_df = calculate_king_of_hill(results)
    
    # Save rankings
    rankings_path = config.OUTPUT_DIR / "model_rankings.csv"
    rankings_df.to_csv(rankings_path, index=False)
    logger.info(f"Saved rankings to {rankings_path}")
    
    # Plot comparison
    plot_path = config.OUTPUT_DIR / "model_comparison.png"
    plot_model_comparison(rankings_df, plot_path)
    
    # Plot predicted vs actual for each model and fold
    logger.info("\nGenerating predicted vs actual plots...")
    
    plots_dir = config.OUTPUT_DIR / "prediction_plots"
    plots_dir.mkdir(exist_ok=True)
    
    for fold_key, fold_data in results.items():
        fold_idx = fold_key.split("_")[1] if "_" in fold_key else "1"
        y_true = np.array(fold_data.get("actuals", []))
        predictions = fold_data.get("predictions", {})
        dates = fold_data.get("dates", [])
        
        if len(y_true) > 0:
            for model_name, y_pred_list in predictions.items():
                if y_pred_list and len(y_pred_list) == len(y_true):
                    y_pred = np.array(y_pred_list)
                    
                    # Generate scatter plot (existing)
                    plot_path = plots_dir / f"predicted_vs_actual_{model_name}_fold_{fold_idx}.png"
                    plot_predicted_vs_actual(
                        y_true, y_pred, model_name, fold_idx=int(fold_idx), save_path=plot_path
                    )
                    
                    # Generate time series line chart (new)
                    if len(dates) == len(y_true):
                        timeseries_path = plots_dir / f"timeseries_{model_name}_fold_{fold_idx}.png"
                        try:
                            from .utils import plot_actual_vs_predicted_timeseries
                        except ImportError:
                            from utils import plot_actual_vs_predicted_timeseries
                        
                        plot_actual_vs_predicted_timeseries(
                            np.array(dates), y_true, y_pred, model_name, 
                            fold_idx=int(fold_idx), save_path=timeseries_path
                        )
    
    logger.info(f"Saved prediction plots to {plots_dir}")
    
    logger.info("\nStep 5 completed successfully!")
    return results, rankings_df


if __name__ == "__main__":
    main()

