"""
Step 4: Multi-Model Training

Trains multiple models: LightGBM, XGBoost, CatBoost, Prophet, LSTM, GRU, RNN, and Ensemble.
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Neural network models will be skipped.")

# Set random seeds for reproducibility
np.random.seed(config.RANDOM_STATE)
if TENSORFLOW_AVAILABLE:
    tf.random.set_seed(config.RANDOM_STATE)


def prepare_training_data(df: pd.DataFrame, feature_cols: list):
    """Prepare training data by removing rows with missing features."""
    logger.info("Preparing training data")
    
    df_clean = df.dropna(subset=[config.TARGET_COL])
    feature_df = df_clean[feature_cols].copy()
    
    for col in feature_cols:
        if feature_df[col].dtype in [np.float64, np.int64]:
            if feature_df[col].isna().any():
                if "lag" in col or "rolling" in col:
                    feature_df[col] = feature_df[col].fillna(0)
                else:
                    feature_df[col] = feature_df[col].fillna(feature_df[col].median())
    
    X = feature_df.values
    y = df_clean[config.TARGET_COL].values
    
    logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
    return X, y, feature_cols


def get_categorical_indices(feature_names: list) -> list:
    """Get indices of categorical features."""
    return [i for i, name in enumerate(feature_names) 
            if name.endswith("_encoded") or name in config.CATEGORICAL_COLS]


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   feature_names: list, optimized_params: dict = None) -> lgb.Booster:
    """Train LightGBM model."""
    logger.info("Training LightGBM...")
    
    params = config.LIGHTGBM_PARAMS.copy()
    params.pop("categorical_feature", None)
    
    # Use optimized params if available
    if optimized_params and 'lightgbm' in optimized_params and optimized_params['lightgbm']:
        params = optimized_params['lightgbm']
        logger.info("Using optimized hyperparameters")
    
    categorical_indices = get_categorical_indices(feature_names)
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names,
                            categorical_feature=categorical_indices if categorical_indices else None)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names,
                          categorical_feature=categorical_indices if categorical_indices else None,
                          reference=train_data)
    
    model = lgb.train(params, train_data, valid_sets=[val_data],
                     num_boost_round=1000,
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  feature_names: list, optimized_params: dict = None) -> xgb.XGBRegressor:
    """Train XGBoost model."""
    logger.info("Training XGBoost...")
    
    default_params = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "random_state": config.RANDOM_STATE,
        "early_stopping_rounds": 50,
        "eval_metric": 'rmse'
    }
    
    # Use optimized params if available
    if optimized_params and 'xgboost' in optimized_params and optimized_params['xgboost']:
        default_params.update(optimized_params['xgboost'])
        logger.info("Using optimized hyperparameters")
    
    model = xgb.XGBRegressor(**default_params)
    
    model.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             verbose=False)
    return model


def train_catboost(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   feature_names: list, optimized_params: dict = None) -> cb.CatBoostRegressor:
    """Train CatBoost model."""
    logger.info("Training CatBoost...")
    
    categorical_indices = get_categorical_indices(feature_names)
    
    default_params = {
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.05,
        "loss_function": 'RMSE',
        "random_seed": config.RANDOM_STATE,
        "early_stopping_rounds": 50,
        "verbose": False,
        "cat_features": categorical_indices if categorical_indices else None
    }
    
    # Use optimized params if available
    if optimized_params and 'catboost' in optimized_params and optimized_params['catboost']:
        default_params.update(optimized_params['catboost'])
        logger.info("Using optimized hyperparameters")
    
    model = cb.CatBoostRegressor(**default_params)
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return model


def prepare_prophet_data(df: pd.DataFrame, train_mask: pd.Series,
                         val_mask: pd.Series, feature_cols: list) -> Tuple:
    """Prepare data for Prophet (needs date column)."""
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    
    # Prophet needs 'ds' (date) and 'y' (target) columns
    train_prophet = train_df[[config.DATE_COL, config.TARGET_COL]].copy()
    train_prophet.columns = ['ds', 'y']
    train_prophet['ds'] = pd.to_datetime(train_prophet['ds'])
    
    val_prophet = val_df[[config.DATE_COL, config.TARGET_COL]].copy()
    val_prophet.columns = ['ds', 'y']
    val_prophet['ds'] = pd.to_datetime(val_prophet['ds'])
    
    return train_prophet, val_prophet


def train_prophet(df: pd.DataFrame, train_mask: pd.Series,
                  val_mask: pd.Series) -> Prophet:
    """Train Prophet model (aggregates across all groups)."""
    logger.info("Training Prophet...")
    
    # Prophet works on univariate time series, so we aggregate across all groups
    train_df = df[train_mask].copy()
    train_df[config.DATE_COL] = pd.to_datetime(train_df[config.DATE_COL])
    
    # Aggregate by date
    prophet_train = train_df.groupby(config.DATE_COL)[config.TARGET_COL].sum().reset_index()
    prophet_train.columns = ['ds', 'y']
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    model.fit(prophet_train)
    return model


def create_sequences(X: np.ndarray, y: np.ndarray, lookback: int = 12) -> Tuple:
    """Create sequences for RNN models."""
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               lookback: int = 12) -> keras.Model:
    """Train LSTM model."""
    logger.info("Training LSTM...")
    
    # Scale data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    
    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, lookback)
    
    # Build model
    model = keras.Sequential([
        layers.LSTM(50, return_sequences=True, input_shape=(lookback, X_train.shape[1])),
        layers.LSTM(50, return_sequences=False),
        layers.Dense(25),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train
    model.fit(X_train_seq, y_train_seq, 
             validation_data=(X_val_seq, y_val_scaled),
             epochs=50, batch_size=32, verbose=0,
             callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    
    # Store scalers with model
    model.scaler_X = scaler_X
    model.scaler_y = scaler_y
    model.lookback = lookback
    
    return model


def train_gru(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              lookback: int = 12) -> keras.Model:
    """Train GRU model."""
    logger.info("Training GRU...")
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, lookback)
    
    model = keras.Sequential([
        layers.GRU(50, return_sequences=True, input_shape=(lookback, X_train.shape[1])),
        layers.GRU(50, return_sequences=False),
        layers.Dense(25),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    model.fit(X_train_seq, y_train_seq,
             validation_data=(X_val_seq, y_val_scaled),
             epochs=50, batch_size=32, verbose=0,
             callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    
    model.scaler_X = scaler_X
    model.scaler_y = scaler_y
    model.lookback = lookback
    
    return model


def train_simple_rnn(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     lookback: int = 12) -> keras.Model:
    """Train Simple RNN model."""
    logger.info("Training Simple RNN...")
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, lookback)
    
    model = keras.Sequential([
        layers.SimpleRNN(50, return_sequences=True, input_shape=(lookback, X_train.shape[1])),
        layers.SimpleRNN(50, return_sequences=False),
        layers.Dense(25),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    model.fit(X_train_seq, y_train_seq,
             validation_data=(X_val_seq, y_val_scaled),
             epochs=50, batch_size=32, verbose=0,
             callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    
    model.scaler_X = scaler_X
    model.scaler_y = scaler_y
    model.lookback = lookback
    
    return model


def train_all_models(X: np.ndarray, y: np.ndarray, feature_names: list,
                    validation_split: float = 0.2, optimize_hyperparams: bool = False) -> Dict:
    """Train all models and return them in a dictionary."""
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=config.RANDOM_STATE
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples, Validation set: {X_val.shape[0]} samples")
    
    # Optimize hyperparameters if requested
    optimized_params = {}
    if optimize_hyperparams and config.USE_OPTUNA:
        try:
            from .hyperparameter_optimization import optimize_all_models
        except ImportError:
            from hyperparameter_optimization import optimize_all_models
        
        logger.info("\n" + "="*50)
        logger.info("HYPERPARAMETER OPTIMIZATION")
        logger.info("="*50)
        categorical_indices = get_categorical_indices(feature_names)
        optimized_params = optimize_all_models(
            X_train, y_train, X_val, y_val, feature_names, categorical_indices,
            n_trials=config.OPTUNA_N_TRIALS
        )
        logger.info("="*50 + "\n")
    
    models = {}
    
    # Tree-based models
    try:
        models['lightgbm'] = train_lightgbm(X_train, y_train, X_val, y_val, feature_names, optimized_params)
        logger.info("✓ LightGBM trained")
    except Exception as e:
        logger.error(f"✗ LightGBM failed: {e}")
    
    try:
        models['xgboost'] = train_xgboost(X_train, y_train, X_val, y_val, feature_names, optimized_params)
        logger.info("✓ XGBoost trained")
    except Exception as e:
        logger.error(f"✗ XGBoost failed: {e}")
    
    try:
        models['catboost'] = train_catboost(X_train, y_train, X_val, y_val, feature_names, optimized_params)
        logger.info("✓ CatBoost trained")
    except Exception as e:
        logger.error(f"✗ CatBoost failed: {e}")
    
    # Neural network models (skip for now, will train in validation step due to sequence requirements)
    # LSTM, GRU, RNN need to be trained per-fold due to sequence window requirements
    
    logger.info(f"Successfully trained {len(models)} models")
    return models, X_train, X_val, y_train, y_val


def save_models(models: Dict, feature_names: list, model_dir: Path):
    """Save all trained models."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    saved_models = {}
    for model_name, model in models.items():
        if model is not None:
            model_path = model_dir / f"{model_name}_model_{timestamp}.pkl"
            
            # Special handling for Keras models
            if isinstance(model, keras.Model):
                model_path = model_dir / f"{model_name}_model_{timestamp}.h5"
                model.save(model_path)
            else:
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
            
            saved_models[model_name] = str(model_path)
            logger.info(f"Saved {model_name} to {model_path}")
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "models": saved_models,
        "feature_names": feature_names,
        "num_features": len(feature_names)
    }
    
    metadata_path = model_dir / f"models_metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return saved_models


def main():
    """Main function to run Step 4."""
    logger.info("=" * 50)
    logger.info("STEP 4: Multi-Model Training")
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
        exclude_cols = [config.DATE_COL, config.TARGET_COL, "is_outlier", f"{config.TARGET_COL}_original"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        for col in config.CATEGORICAL_COLS:
            encoded_col = f"{col}_encoded"
            if encoded_col in df.columns:
                feature_cols.append(encoded_col)
    
    logger.info(f"Using {len(feature_cols)} features")
    
    # Prepare training data
    X, y, feature_names = prepare_training_data(df, feature_cols)
    
    # Train all models (except Prophet and neural networks - they need special handling)
    optimize = config.USE_OPTUNA if hasattr(config, 'USE_OPTUNA') else False
    models, X_train, X_val, y_train, y_val = train_all_models(X, y, feature_names, optimize_hyperparams=optimize)
    
    # Save models
    saved_models = save_models(models, feature_names, config.MODEL_DIR)
    
    logger.info("Step 4 completed successfully!")
    return models, feature_names


if __name__ == "__main__":
    main()

