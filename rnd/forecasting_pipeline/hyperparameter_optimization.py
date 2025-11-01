"""
Hyperparameter Optimization for All Models using Optuna
"""

import numpy as np
import logging
from typing import Dict, Callable, Tuple, Optional

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from prophet import Prophet

try:
    from . import config
except ImportError:
    import config

logger = logging.getLogger(__name__)


def optimize_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      feature_names: list, categorical_indices: Optional[list],
                      n_trials: int = 50) -> dict:
    """Optimize LightGBM hyperparameters."""
    if not OPTUNA_AVAILABLE:
        return None
    
    logger.info(f"Optimizing LightGBM hyperparameters ({n_trials} trials)...")
    
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
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "verbose": -1,
            "random_state": config.RANDOM_STATE,
        }
        
        if categorical_indices:
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names,
                                    categorical_feature=categorical_indices)
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names,
                                  categorical_feature=categorical_indices, reference=train_data)
        else:
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
        
        model = lgb.train(params, train_data, valid_sets=[val_data],
                         num_boost_round=100,
                         callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
        
        predictions = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - predictions) ** 2))
        return rmse
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = config.LIGHTGBM_PARAMS.copy()
    best_params.pop("categorical_feature", None)
    best_params.update(study.best_params)
    
    logger.info(f"Best LightGBM RMSE: {study.best_value:.4f}")
    return best_params


def optimize_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     n_trials: int = 50) -> dict:
    """Optimize XGBoost hyperparameters."""
    if not OPTUNA_AVAILABLE:
        return None
    
    logger.info(f"Optimizing XGBoost hyperparameters ({n_trials} trials)...")
    
    def objective(trial):
        params = {
            "n_estimators": 1000,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "random_state": config.RANDOM_STATE,
            "early_stopping_rounds": 50,
            "eval_metric": "rmse"
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        predictions = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - predictions) ** 2))
        return rmse
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = {
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
        "eval_metric": "rmse",
        "random_state": config.RANDOM_STATE,
        **study.best_params
    }
    
    logger.info(f"Best XGBoost RMSE: {study.best_value:.4f}")
    return best_params


def optimize_catboost(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      categorical_indices: Optional[list],
                      n_trials: int = 50) -> dict:
    """Optimize CatBoost hyperparameters."""
    if not OPTUNA_AVAILABLE:
        return None
    
    logger.info(f"Optimizing CatBoost hyperparameters ({n_trials} trials)...")
    
    def objective(trial):
        params = {
            "iterations": 1000,
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "random_seed": config.RANDOM_STATE,
            "loss_function": "RMSE",
            "early_stopping_rounds": 50,
            "verbose": False
        }
        
        model = cb.CatBoostRegressor(**params)
        if categorical_indices:
            model.fit(X_train, y_train, eval_set=(X_val, y_val),
                     cat_features=categorical_indices)
        else:
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        predictions = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - predictions) ** 2))
        return rmse
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = {
        "iterations": 1000,
        "loss_function": "RMSE",
        "early_stopping_rounds": 50,
        "random_seed": config.RANDOM_STATE,
        "verbose": False,
        **study.best_params
    }
    
    logger.info(f"Best CatBoost RMSE: {study.best_value:.4f}")
    return best_params


def optimize_prophet(train_df, n_trials: int = 20) -> dict:
    """Optimize Prophet hyperparameters."""
    if not OPTUNA_AVAILABLE:
        return None
    
    logger.info(f"Optimizing Prophet hyperparameters ({n_trials} trials)...")
    
    # Prepare data
    prophet_train = train_df.groupby('ds')['y'].sum().reset_index()
    
    def objective(trial):
        params = {
            "yearly_seasonality": trial.suggest_categorical("yearly_seasonality", [True, False]),
            "weekly_seasonality": trial.suggest_categorical("weekly_seasonality", [True, False]),
            "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
            "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
            "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10, log=True),
        }
        
        # Use last 20% for validation
        split_idx = int(len(prophet_train) * 0.8)
        train = prophet_train.iloc[:split_idx]
        val = prophet_train.iloc[split_idx:]
        
        model = Prophet(**params)
        model.fit(train)
        
        future = model.make_future_dataframe(periods=len(val))
        forecast = model.predict(future)
        
        val_forecast = forecast.tail(len(val))['yhat'].values
        val_actual = val['y'].values
        
        rmse = np.sqrt(np.mean((val_actual - val_forecast) ** 2))
        return rmse
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    logger.info(f"Best Prophet RMSE: {study.best_value:.4f}")
    return study.best_params


def optimize_all_models(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        feature_names: list, categorical_indices: Optional[list],
                        train_df_for_prophet=None,
                        n_trials: int = None) -> Dict[str, dict]:
    """
    Optimize hyperparameters for all models.
    
    Returns:
        Dictionary with optimized hyperparameters for each model
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available. Skipping hyperparameter optimization.")
        return {}
    
    if n_trials is None:
        n_trials = config.OPTUNA_N_TRIALS
    
    optimized_params = {}
    
    # Tree-based models
    try:
        optimized_params['lightgbm'] = optimize_lightgbm(
            X_train, y_train, X_val, y_val, feature_names, categorical_indices, n_trials
        )
    except Exception as e:
        logger.error(f"LightGBM optimization failed: {e}")
    
    try:
        optimized_params['xgboost'] = optimize_xgboost(
            X_train, y_train, X_val, y_val, n_trials
        )
    except Exception as e:
        logger.error(f"XGBoost optimization failed: {e}")
    
    try:
        optimized_params['catboost'] = optimize_catboost(
            X_train, y_train, X_val, y_val, categorical_indices, n_trials
        )
    except Exception as e:
        logger.error(f"CatBoost optimization failed: {e}")
    
    # Prophet (if data provided)
    if train_df_for_prophet is not None:
        try:
            optimized_params['prophet'] = optimize_prophet(
                train_df_for_prophet, n_trials=min(20, n_trials)
            )
        except Exception as e:
            logger.error(f"Prophet optimization failed: {e}")
    
    # Note: Neural networks are harder to optimize and can take very long
    # For now, we skip them or use fixed architectures
    
    logger.info(f"Optimized {len([k for k, v in optimized_params.items() if v is not None])} models")
    return optimized_params

