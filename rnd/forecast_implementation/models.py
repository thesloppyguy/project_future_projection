"""
Model training and forecasting module.
Supports multiple models: LightGBM, XGBoost, CatBoost, Prophet.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from .utils import create_future_dates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available")

# Try importing additional models
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    logger.warning("pmdarima (Auto-ARIMA) not available")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HOLTWINTERS_AVAILABLE = True
except ImportError:
    HOLTWINTERS_AVAILABLE = False
    logger.warning("statsmodels (Holt-Winters) not available")

try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.arima.model import ARIMA
    STL_AVAILABLE = True
except ImportError:
    STL_AVAILABLE = False
    logger.warning("statsmodels (STL) not available")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMAX_AVAILABLE = True
except ImportError:
    SARIMAX_AVAILABLE = False
    logger.warning("statsmodels (SARIMAX) not available")

try:
    from sklearn.ensemble import RandomForestRegressor
    RANDOM_FOREST_AVAILABLE = True
except ImportError:
    RANDOM_FOREST_AVAILABLE = False
    logger.warning("sklearn (Random Forest) not available")

try:
    from sklearn.linear_model import QuantileRegressor
    QUANTILE_REGRESSION_AVAILABLE = True
except ImportError:
    QUANTILE_REGRESSION_AVAILABLE = False
    logger.warning("sklearn (Quantile Regression) not available")


class ModelTrainer:
    """Train and manage forecasting models."""
    
    def __init__(self, random_state: int = 42):
        """Initialize model trainer."""
        self.random_state = random_state
        self.models = {}
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: Optional[np.ndarray] = None,
                      y_val: Optional[np.ndarray] = None,
                      sample_weight: Optional[np.ndarray] = None) -> Optional[Any]:
        """Train LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available")
            return None
        
        try:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
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
            else:
                model = lgb.train(params, train_data, num_boost_round=500)
            
            logger.info("LightGBM trained successfully")
            return model
        except Exception as e:
            logger.error(f"Error training LightGBM: {e}")
            return None
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: Optional[np.ndarray] = None,
                      y_val: Optional[np.ndarray] = None,
                      sample_weight: Optional[np.ndarray] = None) -> Optional[Any]:
        """Train XGBoost model."""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available")
            return None
        
        try:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'verbosity': 0
            }
            
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
            
            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val)
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=1000,
                    evals=[(dval, 'validation')],
                    early_stopping_rounds=50,
                    verbose_eval=False
                )
            else:
                model = xgb.train(params, dtrain, num_boost_round=500)
            
            logger.info("XGBoost trained successfully")
            return model
        except Exception as e:
            logger.error(f"Error training XGBoost: {e}")
            return None
    
    def train_catboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: Optional[np.ndarray] = None,
                      y_val: Optional[np.ndarray] = None,
                      sample_weight: Optional[np.ndarray] = None) -> Optional[Any]:
        """Train CatBoost model."""
        if not CATBOOST_AVAILABLE:
            logger.warning("CatBoost not available")
            return None
        
        try:
            model = CatBoostRegressor(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
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
            else:
                model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
            
            logger.info("CatBoost trained successfully")
            return model
        except Exception as e:
            logger.error(f"Error training CatBoost: {e}")
            return None
    
    def train_prophet(self, series: pd.Series) -> Optional[Any]:
        """Train Prophet model."""
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available")
            return None
        
        try:
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': series.index,
                'y': series.values
            })
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,  # Monthly data, no weekly
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            model.fit(df)
            
            logger.info("Prophet trained successfully")
            return model
        except Exception as e:
            logger.error(f"Error training Prophet: {e}")
            return None
    
    def forecast_lightgbm(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Generate forecast with LightGBM."""
        return model.predict(X)
    
    def forecast_xgboost(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Generate forecast with XGBoost."""
        dmatrix = xgb.DMatrix(X)
        return model.predict(dmatrix)
    
    def forecast_catboost(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Generate forecast with CatBoost."""
        return model.predict(X)
    
    def train_auto_arima(self, series: pd.Series) -> Optional[Any]:
        """Train Auto-ARIMA model."""
        if not PMDARIMA_AVAILABLE:
            logger.warning("Auto-ARIMA not available")
            return None
        
        try:
            # Determine seasonal period based on frequency
            freq_str = pd.infer_freq(series.index)
            if freq_str and 'M' in freq_str:
                m = 12  # Monthly
            elif freq_str and 'W' in freq_str:
                m = 52  # Weekly
            else:
                m = 12  # Default
            
            model = pm.auto_arima(
                series.values,
                seasonal=True,
                m=m,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_p=5,
                max_d=2,
                max_q=5,
                max_P=2,
                max_D=1,
                max_Q=2,
                random_state=self.random_state
            )
            
            logger.info("Auto-ARIMA trained successfully")
            return {'model': model, 'last_date': series.index[-1]}
        except Exception as e:
            logger.error(f"Error training Auto-ARIMA: {e}")
            return None
    
    def train_holt_winters(self, series: pd.Series) -> Optional[Any]:
        """Train Holt-Winters (Exponential Smoothing) model."""
        if not HOLTWINTERS_AVAILABLE:
            logger.warning("Holt-Winters not available")
            return None
        
        try:
            # Determine seasonal period
            freq_str = pd.infer_freq(series.index)
            if freq_str and 'M' in freq_str:
                seasonal_periods = 12
            elif freq_str and 'W' in freq_str:
                seasonal_periods = 52
            else:
                seasonal_periods = 12
            
            # Ensure enough data
            if len(series) < seasonal_periods * 2:
                seasonal_periods = max(4, len(series) // 2)
            
            model = ExponentialSmoothing(
                series.values,
                seasonal_periods=seasonal_periods,
                trend='add',
                seasonal='add'
            )
            fitted_model = model.fit(optimized=True)
            
            logger.info("Holt-Winters trained successfully")
            return {'model': fitted_model, 'last_date': series.index[-1]}
        except Exception as e:
            logger.error(f"Error training Holt-Winters: {e}")
            # Try without seasonality
            try:
                model = ExponentialSmoothing(series.values, trend='add', seasonal=None)
                fitted_model = model.fit(optimized=True)
                logger.info("Holt-Winters (no seasonality) trained successfully")
                return {'model': fitted_model, 'last_date': series.index[-1]}
            except Exception as e2:
                logger.error(f"Error with simpler Holt-Winters: {e2}")
                return None
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None,
                           y_val: Optional[np.ndarray] = None,
                           sample_weight: Optional[np.ndarray] = None) -> Optional[Any]:
        """Train Random Forest model."""
        if not RANDOM_FOREST_AVAILABLE:
            logger.warning("Random Forest not available")
            return None
        
        try:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=1
            )
            model.fit(X_train, y_train, sample_weight=sample_weight)
            
            logger.info("Random Forest trained successfully")
            return model
        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")
            return None
    
    def train_quantile_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: Optional[np.ndarray] = None,
                                  y_val: Optional[np.ndarray] = None,
                                  sample_weight: Optional[np.ndarray] = None) -> Optional[Any]:
        """Train Quantile Regression model (median)."""
        if not QUANTILE_REGRESSION_AVAILABLE:
            logger.warning("Quantile Regression not available")
            return None
        
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            model = QuantileRegressor(quantile=0.5, alpha=0.0, solver='highs')
            model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
            
            logger.info("Quantile Regression trained successfully")
            return {'model': model, 'scaler': scaler}
        except Exception as e:
            logger.error(f"Error training Quantile Regression: {e}")
            return None
    
    def train_stl_decomposition(self, series: pd.Series) -> Optional[Any]:
        """Train STL Decomposition model."""
        if not STL_AVAILABLE:
            logger.warning("STL Decomposition not available")
            return None
        
        try:
            # Determine seasonal period
            freq_str = pd.infer_freq(series.index)
            if freq_str and 'M' in freq_str:
                seasonal_period = 12
            elif freq_str and 'W' in freq_str:
                seasonal_period = 52
            else:
                seasonal_period = 12
            
            if len(series) < seasonal_period * 2:
                seasonal_period = max(4, len(series) // 2)
            
            # Perform STL decomposition
            stl = STL(series.values, seasonal=seasonal_period, robust=True)
            decomposition = stl.fit()
            
            # Fit ARIMA on trend
            trend = decomposition.trend
            trend_clean = trend[~np.isnan(trend)]
            
            trend_model = None
            if len(trend_clean) >= 10:
                try:
                    trend_model = ARIMA(trend_clean, order=(1, 1, 1)).fit()
                except:
                    pass
            
            logger.info("STL Decomposition trained successfully")
            return {
                'decomposition': decomposition,
                'trend_model': trend_model,
                'seasonal_period': seasonal_period,
                'last_date': series.index[-1]
            }
        except Exception as e:
            logger.error(f"Error training STL Decomposition: {e}")
            return None
    
    def train_sarimax(self, series: pd.Series) -> Optional[Any]:
        """Train SARIMAX model."""
        if not SARIMAX_AVAILABLE:
            logger.warning("SARIMAX not available")
            return None
        
        try:
            # Determine seasonal period
            freq_str = pd.infer_freq(series.index)
            if freq_str and 'M' in freq_str:
                seasonal_period = 12
            elif freq_str and 'W' in freq_str:
                seasonal_period = 52
            else:
                seasonal_period = 12
            
            if len(series) < seasonal_period * 2:
                seasonal_period = max(4, len(series) // 3)
            
            # Try simpler models first
            models_to_try = [
                {'order': (1, 1, 1), 'seasonal_order': (0, 0, 0, 0), 'maxiter': 30},
                {'order': (1, 1, 1), 'seasonal_order': (1, 0, 0, seasonal_period), 'maxiter': 30},
                {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, seasonal_period), 'maxiter': 50},
            ]
            
            for config in models_to_try:
                try:
                    model = SARIMAX(
                        series.values,
                        order=config['order'],
                        seasonal_order=config['seasonal_order'],
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fitted_model = model.fit(disp=False, maxiter=config['maxiter'], method='nm')
                    logger.info("SARIMAX trained successfully")
                    return {'model': fitted_model, 'last_date': series.index[-1]}
                except:
                    continue
            
            logger.warning("All SARIMAX configurations failed")
            return None
        except Exception as e:
            logger.error(f"Error training SARIMAX: {e}")
            return None
    
    def forecast_prophet(self, model: Any, periods: int, 
                        last_date: pd.Timestamp, freq: str = 'MS') -> pd.Series:
        """Generate forecast with Prophet."""
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        
        # Get only future periods
        future_dates = create_future_dates(last_date, periods, freq)
        
        # Align forecast with future dates
        forecast_df = forecast[forecast['ds'] >= future_dates[0]].head(periods)
        return pd.Series(forecast_df['yhat'].values, index=future_dates[:len(forecast_df)])
    
    def forecast_auto_arima(self, model: Any, periods: int, 
                           last_date: pd.Timestamp, freq: str = 'MS') -> pd.Series:
        """Generate forecast with Auto-ARIMA."""
        if model is None or 'model' not in model:
            return pd.Series(dtype=float)
        
        try:
            pm_model = model['model']
            forecast_values, _ = pm_model.predict(n_periods=periods, return_conf_int=True)
            
            future_dates = create_future_dates(last_date, periods, freq)
            
            return pd.Series(forecast_values, index=future_dates[:len(forecast_values)])
        except Exception as e:
            logger.error(f"Error forecasting with Auto-ARIMA: {e}")
            return pd.Series(dtype=float)
    
    def forecast_holt_winters(self, model: Any, periods: int,
                              last_date: pd.Timestamp, freq: str = 'MS') -> pd.Series:
        """Generate forecast with Holt-Winters."""
        if model is None or 'model' not in model:
            return pd.Series(dtype=float)
        
        try:
            fitted_model = model['model']
            forecast_values = fitted_model.forecast(steps=periods)
            
            future_dates = create_future_dates(last_date, periods, freq)
            
            return pd.Series(forecast_values, index=future_dates[:len(forecast_values)])
        except Exception as e:
            logger.error(f"Error forecasting with Holt-Winters: {e}")
            return pd.Series(dtype=float)
    
    def forecast_random_forest(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Generate forecast with Random Forest."""
        return model.predict(X)
    
    def forecast_quantile_regression(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Generate forecast with Quantile Regression."""
        if isinstance(model, dict):
            qr_model = model['model']
            scaler = model['scaler']
            X_scaled = scaler.transform(X)
            return qr_model.predict(X_scaled)
        else:
            return model.predict(X)
    
    def forecast_stl_decomposition(self, model: Any, periods: int,
                                  last_date: pd.Timestamp, freq: str = 'MS') -> pd.Series:
        """Generate forecast with STL Decomposition."""
        if model is None:
            return pd.Series(dtype=float)
        
        try:
            decomposition = model['decomposition']
            trend_model = model['trend_model']
            seasonal_period = model['seasonal_period']
            
            # Forecast trend
            if trend_model is not None:
                trend_forecast = trend_model.forecast(steps=periods)
            else:
                last_trend = decomposition.trend[-1]
                trend_forecast = np.full(periods, last_trend)
            
            # Get seasonal pattern
            seasonal = decomposition.seasonal
            seasonal_pattern = seasonal[-seasonal_period:]
            
            # Repeat seasonal pattern
            n_repeats = (periods // len(seasonal_pattern)) + 1
            seasonal_forecast = np.tile(seasonal_pattern, n_repeats)[:periods]
            
            # Combine
            forecast_values = trend_forecast + seasonal_forecast
            
            future_dates = create_future_dates(last_date, periods, freq)
            
            return pd.Series(forecast_values, index=future_dates)
        except Exception as e:
            logger.error(f"Error forecasting with STL: {e}")
            return pd.Series(dtype=float)
    
    def forecast_sarimax(self, model: Any, periods: int,
                        last_date: pd.Timestamp, freq: str = 'MS') -> pd.Series:
        """Generate forecast with SARIMAX."""
        if model is None or 'model' not in model:
            return pd.Series(dtype=float)
        
        try:
            fitted_model = model['model']
            forecast_result = fitted_model.forecast(steps=periods)
            
            if isinstance(forecast_result, pd.Series):
                forecast_values = forecast_result.values
            else:
                forecast_values = np.array(forecast_result)
            
            future_dates = create_future_dates(last_date, periods, freq)
            
            return pd.Series(forecast_values, index=future_dates[:len(forecast_values)])
        except Exception as e:
            logger.error(f"Error forecasting with SARIMAX: {e}")
            return pd.Series(dtype=float)
    
    def save_model(self, model: Any, model_name: str, filepath: Path):
        """Save model to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if model_name == 'prophet':
            # Prophet models need special handling
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: Path) -> Optional[Any]:
        """Load model from disk."""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

