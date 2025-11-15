"""
Auto-ARIMA model for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Optional
import pickle
from pathlib import Path

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
    STATSFORECAST_AVAILABLE = True
except ImportError:
    try:
        import pmdarima as pm
        PMDARIMA_AVAILABLE = True
        STATSFORECAST_AVAILABLE = False
    except ImportError:
        STATSFORECAST_AVAILABLE = False
        PMDARIMA_AVAILABLE = False
        print("Warning: Neither statsforecast nor pmdarima available. Auto-ARIMA will not work.")


def train_model(
    train_data: pd.Series,
    branch: str,
    freq: str = 'W-MON',
    use_optimization: bool = True,
    n_trials: int = 20,
    anomaly_labels: Optional[pd.Series] = None,
    anomaly_severity: Optional[pd.Series] = None,
    yoy_growth: Optional[pd.Series] = None
) -> Optional[dict]:
    """
    Train Auto-ARIMA model.
    
    Args:
        train_data: Training time series
        branch: Branch name
        freq: Frequency string
        
    Returns:
        Dictionary with model and metadata, or None if training fails
    """
    if len(train_data) < 10:
        print(f"Warning: Insufficient data for Auto-ARIMA for {branch}")
        return None
    
    try:
        if STATSFORECAST_AVAILABLE:
            # Use statsforecast
            model = AutoARIMA()
            sf = StatsForecast(
                models=[model],
                freq=freq,
                n_jobs=1
            )
            # Prepare data for statsforecast
            df = pd.DataFrame({
                'ds': train_data.index,
                'y': train_data.values,
                'unique_id': branch
            })
            sf.fit(df)
            return {
                'model': sf,
                'branch': branch,
                'freq': freq,
                'last_date': train_data.index[-1] if len(train_data) > 0 else None
            }
        elif PMDARIMA_AVAILABLE:
            # Use pmdarima
            model = pm.auto_arima(
                train_data.values,
                seasonal=True,
                m=52 if freq == 'W-MON' else 12,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_p=5,
                max_d=2,
                max_q=5,
                max_P=2,
                max_D=1,
                max_Q=2
            )
            return {
                'model': model,
                'branch': branch,
                'freq': freq,
                'last_date': train_data.index[-1] if len(train_data) > 0 else None
            }
        else:
            print("Error: No Auto-ARIMA library available")
            return None
    except Exception as e:
        print(f"Error training Auto-ARIMA for {branch}: {e}")
        return None


def forecast(model: dict, n_periods: int, freq: str = 'W-MON', branch: str = None) -> pd.Series:
    """
    Generate forecast using trained model.
    
    Args:
        model: Dictionary with trained model and metadata
        n_periods: Number of periods to forecast
        freq: Frequency string
        branch: Branch name (for statsforecast)
        
    Returns:
        Forecast series with Date index
    """
    if model is None:
        return pd.Series(dtype=float)
    
    try:
        if STATSFORECAST_AVAILABLE:
            # Use statsforecast
            sf = model['model']
            branch_name = model.get('branch', branch)
            last_date = model.get('last_date')
            
            # Generate forecast
            forecast_df = sf.predict(h=n_periods)
            
            # Filter for the specific branch
            if branch_name and 'unique_id' in forecast_df.columns:
                forecast_df = forecast_df[forecast_df['unique_id'] == branch_name]
            
            # Extract forecast values - StatsForecast returns columns like 'AutoARIMA'
            # Find the model column (should be 'AutoARIMA')
            model_col = None
            for col in forecast_df.columns:
                if col not in ['unique_id', 'ds']:
                    model_col = col
                    break
            
            if model_col is None:
                # Try common column names
                if 'AutoARIMA' in forecast_df.columns:
                    model_col = 'AutoARIMA'
                else:
                    # Use first numeric column
                    numeric_cols = forecast_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        model_col = numeric_cols[0]
                    else:
                        raise ValueError("Could not find forecast column in StatsForecast output")
            
            forecast_values = forecast_df[model_col].values
            
            # Get forecast dates from the DataFrame
            if 'ds' in forecast_df.columns:
                forecast_dates = pd.to_datetime(forecast_df['ds'].values)
            else:
                # Create dates from last_date
                if last_date:
                    forecast_dates = pd.date_range(
                        start=pd.to_datetime(last_date) + pd.Timedelta(days=1),
                        periods=n_periods,
                        freq=freq
                    )
                else:
                    forecast_dates = pd.date_range(start=pd.Timestamp.now(), periods=n_periods, freq=freq)
            
            return pd.Series(forecast_values, index=forecast_dates)
        elif PMDARIMA_AVAILABLE:
            # Use pmdarima
            pm_model = model['model']
            last_date = model.get('last_date')
            
            forecast_values, conf_int = pm_model.predict(n_periods=n_periods, return_conf_int=True)
            
            # Create date index
            if last_date:
                forecast_dates = pd.date_range(
                    start=pd.to_datetime(last_date) + pd.Timedelta(days=1),
                    periods=n_periods,
                    freq=freq
                )
            else:
                forecast_dates = pd.date_range(start=pd.Timestamp.now(), periods=n_periods, freq=freq)
            
            return pd.Series(forecast_values, index=forecast_dates)
        else:
            return pd.Series(dtype=float)
    except Exception as e:
        print(f"Error forecasting with Auto-ARIMA: {e}")
        import traceback
        traceback.print_exc()
        return pd.Series(dtype=float)


def save_model(model: object, path: str) -> None:
    """Save model to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path: str) -> object:
    """Load model from file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

