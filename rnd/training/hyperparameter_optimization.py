"""
Hyperparameter optimization utilities for time series forecasting models.
Uses Optuna for efficient hyperparameter search.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import optuna
from optuna import Trial
import warnings
warnings.filterwarnings('ignore')


def optimize_sarimax_hyperparameters(
    train_data: pd.Series,
    n_trials: int = 10,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Optimize SARIMAX hyperparameters using Optuna.
    
    Args:
        train_data: Training time series
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with best hyperparameters
    """
    def objective(trial: Trial) -> float:
        # Suggest ARIMA order parameters (reduced search space for speed)
        p = trial.suggest_int('p', 0, 2)
        d = trial.suggest_int('d', 0, 1)
        q = trial.suggest_int('q', 0, 2)
        
        # Suggest seasonal order parameters (reduced search space)
        P = trial.suggest_int('P', 0, 1)
        D = trial.suggest_int('D', 0, 1)
        Q = trial.suggest_int('Q', 0, 1)
        
        # Determine seasonal period
        seasonal_period = 52 if len(train_data) > 100 else 12
        
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            model = SARIMAX(
                train_data.values,
                order=(p, d, q),
                seasonal_order=(P, D, Q, seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            # Use faster Nelder-Mead method with fewer iterations
            fitted_model = model.fit(
                disp=False,
                maxiter=30,  # Reduced from 50
                method='nm'  # Nelder-Mead is faster
            )
            
            # Use AIC as the objective
            return fitted_model.aic
        except:
            return float('inf')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    return study.best_params


def optimize_holt_winters_hyperparameters(
    train_data: pd.Series,
    n_trials: int = 10,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Optimize Holt-Winters hyperparameters.
    
    Args:
        train_data: Training time series
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with best hyperparameters
    """
    def objective(trial: Trial) -> float:
        trend = trial.suggest_categorical('trend', ['add', 'mul', None])
        seasonal = trial.suggest_categorical('seasonal', ['add', 'mul', None])
        
        seasonal_periods = None
        if seasonal is not None:
            seasonal_periods = 52 if len(train_data) > 100 else 12
        
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            model = ExponentialSmoothing(
                train_data.values,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods
            )
            
            fitted_model = model.fit(optimized=True)
            return fitted_model.aic
        except:
            return float('inf')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    return study.best_params


def optimize_xgboost_hyperparameters(
    train_data: pd.Series,
    n_trials: int = 20,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Optimize XGBoost hyperparameters.
    
    Args:
        train_data: Training time series
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with best hyperparameters
    """
    def objective(trial: Trial) -> float:
        import xgboost as xgb
        from sklearn.metrics import mean_squared_error
        
        # Create features
        from training.models.xgboost import create_features
        feature_df = create_features(train_data)
        feature_cols = [col for col in feature_df.columns if col not in ['y', 'date']]
        X = feature_df[feature_cols].values
        y = feature_df['y'].values
        
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 20:
            return float('inf')
        
        # Split for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'random_state': 42,
            'n_jobs': 1
        }
        
        try:
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return mean_squared_error(y_val, y_pred)
        except:
            return float('inf')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    return study.best_params


def optimize_lightgbm_hyperparameters(
    train_data: pd.Series,
    n_trials: int = 20,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Optimize LightGBM hyperparameters.
    
    Args:
        train_data: Training time series
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with best hyperparameters
    """
    def objective(trial: Trial) -> float:
        import lightgbm as lgb
        from sklearn.metrics import mean_squared_error
        
        # Create features
        from training.models.lightgbm import create_features
        feature_df = create_features(train_data)
        feature_cols = [col for col in feature_df.columns if col not in ['y', 'date']]
        X = feature_df[feature_cols].values
        y = feature_df['y'].values
        
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 20:
            return float('inf')
        
        # Split for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'random_state': 42,
            'n_jobs': 1,
            'verbose': -1
        }
        
        try:
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return mean_squared_error(y_val, y_pred)
        except:
            return float('inf')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    return study.best_params


def optimize_catboost_hyperparameters(
    train_data: pd.Series,
    n_trials: int = 20,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Optimize CatBoost hyperparameters.
    
    Args:
        train_data: Training time series
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with best hyperparameters
    """
    def objective(trial: Trial) -> float:
        from catboost import CatBoostRegressor
        from sklearn.metrics import mean_squared_error
        
        # Create features
        from training.models.catboost import create_features
        feature_df = create_features(train_data)
        feature_cols = [col for col in feature_df.columns if col not in ['y', 'date']]
        X = feature_df[feature_cols].values
        y = feature_df['y'].values
        
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 20:
            return float('inf')
        
        # Split for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Suggest hyperparameters
        params = {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_seed': 42,
            'verbose': False
        }
        
        try:
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return mean_squared_error(y_val, y_pred)
        except:
            return float('inf')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    return study.best_params


def optimize_random_forest_hyperparameters(
    train_data: pd.Series,
    n_trials: int = 20,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Optimize Random Forest hyperparameters.
    
    Args:
        train_data: Training time series
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with best hyperparameters
    """
    def objective(trial: Trial) -> float:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        
        # Create features
        from training.models.random_forest import create_features
        feature_df = create_features(train_data)
        feature_cols = [col for col in feature_df.columns if col not in ['y', 'date']]
        X = feature_df[feature_cols].values
        y = feature_df['y'].values
        
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 20:
            return float('inf')
        
        # Split for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': 1
        }
        
        try:
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return mean_squared_error(y_val, y_pred)
        except:
            return float('inf')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    return study.best_params


def optimize_lstm_hyperparameters(
    train_data: pd.Series,
    n_trials: int = 10,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Optimize LSTM hyperparameters.
    
    Args:
        train_data: Training time series
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with best hyperparameters
    """
    def objective(trial: Trial) -> float:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from sklearn.metrics import mean_squared_error
        
        # Prepare data
        from training.models.lstm import create_sequences
        
        values = train_data.values.astype(float)
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            std = 1
        normalized_values = (values - mean) / std
        
        n_steps = trial.suggest_int('n_steps', 6, 24)
        X, y = create_sequences(normalized_values, n_steps)
        
        if len(X) < 20:
            return float('inf')
        
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Suggest hyperparameters
        lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 128)
        lstm_units_2 = trial.suggest_int('lstm_units_2', 32, 128)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
        
        try:
            model = Sequential([
                LSTM(lstm_units_1, activation='relu', return_sequences=True, input_shape=(n_steps, 1)),
                Dropout(dropout_rate),
                LSTM(lstm_units_2, activation='relu', return_sequences=False),
                Dropout(dropout_rate),
                Dense(1)
            ])
            
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
            
            # Early stopping
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val),
                     callbacks=[early_stop], verbose=0)
            
            y_pred = model.predict(X_val, verbose=0).flatten()
            return mean_squared_error(y_val, y_pred)
        except:
            return float('inf')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    return study.best_params


def optimize_prophet_hyperparameters(
    train_data: pd.Series,
    n_trials: int = 10,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Optimize Prophet hyperparameters.
    
    Args:
        train_data: Training time series
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with best hyperparameters
    """
    def objective(trial: Trial) -> float:
        from prophet import Prophet
        from sklearn.metrics import mean_squared_error
        
        # Prepare data
        df = pd.DataFrame({
            'ds': train_data.index,
            'y': train_data.values
        })
        
        # Split for validation
        split_idx = int(len(df) * 0.8)
        train_df = df[:split_idx]
        val_df = df[split_idx:]
        
        # Suggest hyperparameters
        yearly_seasonality = trial.suggest_categorical('yearly_seasonality', [True, False])
        weekly_seasonality = trial.suggest_categorical('weekly_seasonality', [True, False])
        seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
        changepoint_prior_scale = trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True)
        seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True)
        
        try:
            model = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=False,
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale
            )
            
            model.fit(train_df)
            forecast = model.predict(val_df[['ds']])
            
            return mean_squared_error(val_df['y'].values, forecast['yhat'].values)
        except:
            return float('inf')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    return study.best_params


def optimize_neural_prophet_hyperparameters(
    train_data: pd.Series,
    n_trials: int = 10,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Optimize Neural Prophet hyperparameters.
    
    Args:
        train_data: Training time series
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with best hyperparameters
    """
    def objective(trial: Trial) -> float:
        try:
            from neuralprophet import NeuralProphet
        except (ImportError, TypeError):
            return float('inf')
        
        from sklearn.metrics import mean_squared_error
        
        # Prepare data
        df = pd.DataFrame({
            'ds': train_data.index,
            'y': train_data.values
        })
        
        # Split for validation
        split_idx = int(len(df) * 0.8)
        train_df = df[:split_idx]
        val_df = df[split_idx:]
        
        # Suggest hyperparameters
        yearly_seasonality = trial.suggest_categorical('yearly_seasonality', [True, False])
        weekly_seasonality = trial.suggest_categorical('weekly_seasonality', [True, False])
        n_lags = trial.suggest_int('n_lags', 5, 20)
        n_forecasts = trial.suggest_int('n_forecasts', 12, 52)
        
        try:
            model = NeuralProphet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=False,
                n_lags=n_lags,
                n_forecasts=n_forecasts
            )
            
            model.fit(train_df, verbose=False)
            forecast = model.predict(val_df)
            
            # Align forecast with validation data
            forecast_aligned = forecast.set_index('ds').reindex(val_df['ds'])
            valid_forecast = forecast_aligned['yhat1'].dropna()
            valid_actual = val_df.set_index('ds').loc[valid_forecast.index, 'y']
            
            if len(valid_forecast) == 0:
                return float('inf')
            
            return mean_squared_error(valid_actual.values, valid_forecast.values)
        except:
            return float('inf')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    return study.best_params


def optimize_bayesian_deep_learning_hyperparameters(
    train_data: pd.Series,
    n_trials: int = 10,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Optimize Bayesian Deep Learning hyperparameters.
    
    Args:
        train_data: Training time series
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with best hyperparameters
    """
    def objective(trial: Trial) -> float:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from sklearn.metrics import mean_squared_error
        
        # Prepare data
        from training.models.bayesian_deep_learning import create_sequences
        
        values = train_data.values.astype(float)
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            std = 1
        normalized_values = (values - mean) / std
        
        n_steps = trial.suggest_int('n_steps', 6, 24)
        X, y = create_sequences(normalized_values, n_steps)
        
        if len(X) < 20:
            return float('inf')
        
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Suggest hyperparameters
        lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 128)
        lstm_units_2 = trial.suggest_int('lstm_units_2', 32, 128)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)  # Higher dropout for Bayesian
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
        
        try:
            model = Sequential([
                LSTM(lstm_units_1, activation='relu', return_sequences=True, input_shape=(n_steps, 1)),
                Dropout(dropout_rate),
                LSTM(lstm_units_2, activation='relu', return_sequences=False),
                Dropout(dropout_rate),
                Dense(1)
            ])
            
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
            
            # Early stopping
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val),
                     callbacks=[early_stop], verbose=0)
            
            y_pred = model.predict(X_val, verbose=0).flatten()
            return mean_squared_error(y_val, y_pred)
        except:
            return float('inf')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    return study.best_params


# Default hyperparameters for models that don't need optimization
DEFAULT_HYPERPARAMETERS = {
    'auto_arima': {},  # Auto-ARIMA optimizes itself
    'stl_decomposition': {},  # STL doesn't have hyperparameters to optimize
    'isolation_forest': {
        'contamination': 0.1,
        'n_estimators': 100,
        'random_state': 42
    },
    'quantile_regression': {
        'quantile': 0.5,
        'alpha': 0.0,
        'solver': 'highs'
    },
    'var': {
        'maxlags': 4,
        'ic': 'aic'
    },
    'kalman_filter': {
        'level': 'local level',
        'trend': True,
        'seasonal': None
    },
    'neural_prophet': {
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'n_lags': 10,
        'n_forecasts': 52
    },
    'bayesian_deep_learning': {
        'lstm_units_1': 50,
        'lstm_units_2': 50,
        'dropout_rate': 0.3,
        'learning_rate': 0.001
    }
}


def get_hyperparameters(model_name: str, train_data: pd.Series, 
                       use_optimization: bool = True,
                       n_trials: int = 10,
                       timeout: Optional[float] = None) -> Dict[str, Any]:
    """
    Get hyperparameters for a model, either through optimization or defaults.
    
    Args:
        model_name: Name of the model
        train_data: Training time series
        use_optimization: Whether to use hyperparameter optimization
        n_trials: Number of optimization trials
        timeout: Timeout in seconds for optimization
        
    Returns:
        Dictionary with hyperparameters
    """
    if not use_optimization:
        return DEFAULT_HYPERPARAMETERS.get(model_name, {})
    
    optimization_functions = {
        'sarimax': optimize_sarimax_hyperparameters,
        'holt_winters': optimize_holt_winters_hyperparameters,
        'xgboost': optimize_xgboost_hyperparameters,
        'lightgbm': optimize_lightgbm_hyperparameters,
        'catboost': optimize_catboost_hyperparameters,
        'random_forest': optimize_random_forest_hyperparameters,
        'lstm': optimize_lstm_hyperparameters,
        'prophet': optimize_prophet_hyperparameters,
        'neural_prophet': optimize_neural_prophet_hyperparameters,
        'bayesian_deep_learning': optimize_bayesian_deep_learning_hyperparameters
    }
    
    if model_name in optimization_functions:
        try:
            print(f"Optimizing hyperparameters for {model_name}...")
            return optimization_functions[model_name](train_data, n_trials=n_trials, timeout=timeout)
        except Exception as e:
            print(f"Warning: Hyperparameter optimization failed for {model_name}: {e}")
            print(f"Using default hyperparameters...")
            return DEFAULT_HYPERPARAMETERS.get(model_name, {})
    else:
        return DEFAULT_HYPERPARAMETERS.get(model_name, {})
