"""
Hyperparameter tuning for top models.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing optuna for advanced tuning
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available, using grid search")


class HyperparameterTuner:
    """Hyperparameter tuning for forecasting models."""
    
    def __init__(self, random_state: int = 42, n_trials: int = 50):
        """
        Initialize hyperparameter tuner.
        
        Args:
            random_state: Random seed
            n_trials: Number of trials for optimization
        """
        self.random_state = random_state
        self.n_trials = n_trials
    
    def tune_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Tune LightGBM hyperparameters.
        
        Returns:
            Dictionary with best parameters
        """
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("LightGBM not available")
            return {}
        
        if OPTUNA_AVAILABLE:
            return self._tune_lightgbm_optuna(X_train, y_train, X_val, y_val, sample_weight)
        else:
            return self._tune_lightgbm_grid(X_train, y_train, X_val, y_val, sample_weight)
    
    def _tune_lightgbm_optuna(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Tune LightGBM using Optuna."""
        import lightgbm as lgb
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'verbose': -1,
                'random_state': self.random_state
            }
            
            train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
            
            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=1000,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=0)
                    ]
                )
                return model.best_score['valid_0']['rmse']
            else:
                model = lgb.train(params, train_data, num_boost_round=500)
                pred = model.predict(X_train)
                return np.sqrt(np.mean((y_train - pred) ** 2))
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return study.best_params
    
    def _tune_lightgbm_grid(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Tune LightGBM using grid search."""
        import lightgbm as lgb
        
        best_params = None
        best_score = np.inf
        
        # Limited grid search
        param_grid = {
            'num_leaves': [31, 50, 70],
            'learning_rate': [0.03, 0.05, 0.1],
            'feature_fraction': [0.8, 0.9, 1.0],
            'bagging_fraction': [0.7, 0.8, 0.9]
        }
        
        for num_leaves in param_grid['num_leaves']:
            for lr in param_grid['learning_rate']:
                for ff in param_grid['feature_fraction']:
                    for bf in param_grid['bagging_fraction']:
                        params = {
                            'objective': 'regression',
                            'metric': 'rmse',
                            'boosting_type': 'gbdt',
                            'num_leaves': num_leaves,
                            'learning_rate': lr,
                            'feature_fraction': ff,
                            'bagging_fraction': bf,
                            'bagging_freq': 5,
                            'verbose': -1,
                            'random_state': self.random_state
                        }
                        
                        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
                        
                        if X_val is not None and y_val is not None:
                            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                            model = lgb.train(
                                params,
                                train_data,
                                valid_sets=[val_data],
                                num_boost_round=1000,
                                callbacks=[
                                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                                    lgb.log_evaluation(period=0)
                                ]
                            )
                            score = model.best_score['valid_0']['rmse']
                        else:
                            model = lgb.train(params, train_data, num_boost_round=500)
                            pred = model.predict(X_train)
                            score = np.sqrt(np.mean((y_train - pred) ** 2))
                        
                        if score < best_score:
                            best_score = score
                            best_params = params.copy()
        
        return best_params if best_params else {}
    
    def tune_catboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Tune CatBoost hyperparameters."""
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            logger.warning("CatBoost not available")
            return {}
        
        if OPTUNA_AVAILABLE:
            return self._tune_catboost_optuna(X_train, y_train, X_val, y_val, sample_weight)
        else:
            return self._tune_catboost_grid(X_train, y_train, X_val, y_val, sample_weight)
    
    def _tune_catboost_optuna(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Tune CatBoost using Optuna."""
        from catboost import CatBoostRegressor
        
        def objective(trial):
            params = {
                'iterations': 1000,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'loss_function': 'RMSE',
                'random_seed': self.random_state,
                'verbose': False
            }
            
            model = CatBoostRegressor(**params)
            
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    sample_weight=sample_weight,
                    early_stopping_rounds=50,
                    verbose=False
                )
                return model.get_best_score()['validation']['RMSE']
            else:
                model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
                pred = model.predict(X_train)
                return np.sqrt(np.mean((y_train - pred) ** 2))
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return study.best_params
    
    def _tune_catboost_grid(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Tune CatBoost using grid search."""
        from catboost import CatBoostRegressor
        
        best_params = None
        best_score = np.inf
        
        param_grid = {
            'learning_rate': [0.03, 0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5]
        }
        
        for lr in param_grid['learning_rate']:
            for depth in param_grid['depth']:
                for l2 in param_grid['l2_leaf_reg']:
                    model = CatBoostRegressor(
                        iterations=1000,
                        learning_rate=lr,
                        depth=depth,
                        l2_leaf_reg=l2,
                        loss_function='RMSE',
                        random_seed=self.random_state,
                        verbose=False
                    )
                    
                    if X_val is not None and y_val is not None:
                        model.fit(
                            X_train, y_train,
                            eval_set=(X_val, y_val),
                            sample_weight=sample_weight,
                            early_stopping_rounds=50,
                            verbose=False
                        )
                        score = model.get_best_score()['validation']['RMSE']
                    else:
                        model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
                        pred = model.predict(X_train)
                        score = np.sqrt(np.mean((y_train - pred) ** 2))
                    
                    if score < best_score:
                        best_score = score
                        best_params = {
                            'learning_rate': lr,
                            'depth': depth,
                            'l2_leaf_reg': l2
                        }
        
        return best_params if best_params else {}
    
    def tune_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Tune Random Forest hyperparameters."""
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            logger.warning("sklearn not available")
            return {}
        
        if OPTUNA_AVAILABLE:
            return self._tune_random_forest_optuna(X_train, y_train, X_val, y_val, sample_weight)
        else:
            return self._tune_random_forest_grid(X_train, y_train, X_val, y_val, sample_weight)
    
    def _tune_random_forest_optuna(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Tune Random Forest using Optuna."""
        from sklearn.ensemble import RandomForestRegressor
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state,
                'n_jobs': 1
            }
            
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train, sample_weight=sample_weight)
            
            if X_val is not None and y_val is not None:
                pred = model.predict(X_val)
                return np.sqrt(np.mean((y_val - pred) ** 2))
            else:
                pred = model.predict(X_train)
                return np.sqrt(np.mean((y_train - pred) ** 2))
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return study.best_params
    
    def _tune_random_forest_grid(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Tune Random Forest using grid search."""
        from sklearn.ensemble import RandomForestRegressor
        
        best_params = None
        best_score = np.inf
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2']
        }
        
        for n_est in param_grid['n_estimators']:
            for depth in param_grid['max_depth']:
                for mss in param_grid['min_samples_split']:
                    for mf in param_grid['max_features']:
                        model = RandomForestRegressor(
                            n_estimators=n_est,
                            max_depth=depth,
                            min_samples_split=mss,
                            max_features=mf,
                            random_state=self.random_state,
                            n_jobs=1
                        )
                        model.fit(X_train, y_train, sample_weight=sample_weight)
                        
                        if X_val is not None and y_val is not None:
                            pred = model.predict(X_val)
                            score = np.sqrt(np.mean((y_val - pred) ** 2))
                        else:
                            pred = model.predict(X_train)
                            score = np.sqrt(np.mean((y_train - pred) ** 2))
                        
                        if score < best_score:
                            best_score = score
                            best_params = {
                                'n_estimators': n_est,
                                'max_depth': depth,
                                'min_samples_split': mss,
                                'max_features': mf
                            }
        
        return best_params if best_params else {}

