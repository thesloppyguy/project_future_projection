import pandas as pd
from datetime import datetime
from collections import defaultdict
import math
import numpy as np
import xgboost as xgb
import optuna
from typing import Dict, List


def generate_forecasts(csv_path: str, forecast_start_date: str = None):
    # Load data
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)

    # Filter from Sept 2021 onwards
    df = df[df['Date'] >= pd.Timestamp('2021-09-01')]

    # Group by date + branch and sum quantities
    grouped = (
        df.groupby(['Date', 'Branch'])['Quantity']
        .sum()
        .reset_index()
        .sort_values('Date')
    )

    # Determine FY start
    now = datetime.now()
    current_year = now.year
    current_month = now.month - 1
    fy_start = datetime(current_year, 4, 1)
    if current_month < 3:
        fy_start = datetime(current_year - 1, 4, 1)

    branches = grouped['Branch'].unique()

    from_last_data_point = defaultdict(list)
    from_fy_start = defaultdict(list)

    for branch in branches:
        branch_data = grouped[grouped['Branch'] == branch].copy()
        if branch_data.empty:
            continue

        # Aggregate by month
        branch_data['year'] = branch_data['Date'].dt.year
        branch_data['month'] = branch_data['Date'].dt.month
        monthly = (
            branch_data.groupby(['year', 'month'])['Quantity']
            .sum()
            .reset_index()
        )
        monthly['Date'] = pd.to_datetime(
            monthly[['year', 'month']].assign(day=1)
        )
        monthly = monthly.sort_values('Date')

        if monthly.empty:
            continue

        last_date = monthly['Date'].max()
        monthly_before_fy = monthly[monthly['Date'] < fy_start]

        # --- Compute Seasonal Indices ---
        monthly_by_month = (
            monthly.groupby('month')['Quantity']
            .apply(list)
            .to_dict()
        )

        seasonal_indices = {
            month: sum(vals) / len(vals)
            for month, vals in monthly_by_month.items()
        }
        overall_avg = sum(seasonal_indices.values()) / len(seasonal_indices)
        seasonal_indices = {
            m: (v / overall_avg if overall_avg > 0 else 1)
            for m, v in seasonal_indices.items()
        }

        # --- Trend averages ---
        recent_monthly = monthly.tail(12)
        avg_recent_qty = recent_monthly['Quantity'].mean(
        ) if not recent_monthly.empty else 0

        fy_recent = monthly_before_fy.tail(12)
        avg_fy_recent_qty = fy_recent['Quantity'].mean(
        ) if not fy_recent.empty else avg_recent_qty

        # --- Forecast 1: From last data point (or specified start date) ---
        # Determine forecast start date
        if forecast_start_date:
            forecast_start = pd.to_datetime(forecast_start_date)
            if forecast_start <= last_date:
                forecast_start = last_date + pd.DateOffset(months=1)
        else:
            forecast_start = last_date + pd.DateOffset(months=1)
        
        for i in range(12):
            forecast_date = forecast_start + pd.DateOffset(months=i)
            month = forecast_date.month
            # Calculate months from last_date for growth factor
            months_from_last = (forecast_date - last_date).days // 30 if forecast_date > last_date else 0
            growth_factor = math.pow(1.03, months_from_last / 12)  # 3% annual
            seasonal = seasonal_indices.get(month, 1)
            forecast_qty = avg_recent_qty * seasonal * growth_factor

            from_last_data_point[branch].append({
                "date": forecast_date.strftime("%Y-%m-%d"),
                "quantity": round(forecast_qty, 1),
                "type": "forecast"
            })

        # --- Forecast 2: From FY start ---
        for i in range(12):
            forecast_date = fy_start + pd.DateOffset(months=i)
            month = forecast_date.month
            growth_factor = math.pow(1.03, i / 12)
            seasonal = seasonal_indices.get(month, 1)
            forecast_qty = avg_fy_recent_qty * seasonal * growth_factor

            from_fy_start[branch].append({
                "date": forecast_date.strftime("%Y-%m-%d"),
                "quantity": round(forecast_qty, 1),
                "type": "forecast"
            })

    return {
        "from_last_data_point": dict(from_last_data_point),
        "from_fy_start": dict(from_fy_start)
    }


def create_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time series features for XGBoost forecasting."""
    df = df.copy()
    df = df.sort_values('Date')
    
    # Time-based features
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    
    # Cyclical encoding for seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    # Lag features
    for lag in [1, 3, 6, 12]:
        df[f'lag_{lag}'] = df['Quantity'].shift(lag)
    
    # Rolling statistics
    for window in [3, 6, 12]:
        df[f'rolling_mean_{window}'] = df['Quantity'].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['Quantity'].rolling(window=window, min_periods=1).std()
        df[f'rolling_min_{window}'] = df['Quantity'].rolling(window=window, min_periods=1).min()
        df[f'rolling_max_{window}'] = df['Quantity'].rolling(window=window, min_periods=1).max()
    
    # Exponential moving averages
    for span in [3, 6, 12]:
        df[f'ema_{span}'] = df['Quantity'].ewm(span=span, adjust=False).mean()
    
    # Trend features
    df['trend'] = range(len(df))
    df['diff'] = df['Quantity'].diff()
    df['pct_change'] = df['Quantity'].pct_change()
    
    # NEW: Seasonal average features (similar to decomposition method)
    # Calculate seasonal indices (average per month across all years)
    monthly_averages = {}
    for month in range(1, 13):
        month_data = df[df['month'] == month]['Quantity']
        if len(month_data) > 0:
            monthly_averages[month] = month_data.mean()
        else:
            monthly_averages[month] = df['Quantity'].mean()
    
    # Overall average for normalization
    overall_avg = sum(monthly_averages.values()) / 12 if monthly_averages else df['Quantity'].mean()
    
    # Seasonal index feature (decomposition-style)
    df['seasonal_index'] = df['month'].map(monthly_averages) / overall_avg if overall_avg > 0 else 1.0
    
    # Recent average (last 12 months) - key feature from decomposition
    df['recent_12m_avg'] = df['Quantity'].rolling(window=12, min_periods=1).mean().shift(1)
    
    # Baseline forecast feature (decomposition-style: recent_avg * seasonal_index)
    df['baseline_forecast'] = df['recent_12m_avg'] * df['seasonal_index']
    
    # Historical average for same month (more stable than recent_12m_avg)
    df['monthly_historical_avg'] = df['month'].map(monthly_averages)
    
    # Ratio of recent average to historical monthly average
    df['recent_to_monthly_ratio'] = np.where(
        df['monthly_historical_avg'] > 0,
        df['recent_12m_avg'] / df['monthly_historical_avg'],
        1.0
    )
    
    # Growth trend (year-over-year same month)
    df['yoy_growth'] = np.nan
    for idx in range(len(df)):
        current_date = df.iloc[idx]['Date']
        current_month = current_date.month
        # Find same month from previous year
        prev_year_date = current_date - pd.DateOffset(years=1)
        prev_year_data = df[(df['Date'] <= prev_year_date) & (df['month'] == current_month)]
        if len(prev_year_data) > 0:
            prev_qty = prev_year_data.iloc[-1]['Quantity']
            current_qty = df.iloc[idx]['Quantity']
            if prev_qty > 0:
                df.iloc[idx, df.columns.get_loc('yoy_growth')] = current_qty / prev_qty
            else:
                df.iloc[idx, df.columns.get_loc('yoy_growth')] = 1.0
    
    # Fill NaN values
    df = df.fillna(0)
    
    # Replace inf values
    df = df.replace([np.inf, -np.inf], 0)
    
    return df


def optimize_xgboost_hyperparameters(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_val: np.ndarray, 
    y_val: np.ndarray,
    n_trials: int = 50
) -> dict:
    """Optimize XGBoost hyperparameters using Optuna."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.5, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "gamma": trial.suggest_float("gamma", 0, 3),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": 42,
            "verbosity": 0
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )
        
        predictions = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - predictions) ** 2))
        return rmse
    
    study = optuna.create_study(direction="minimize", study_name="xgboost_optimization")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = {
        "n_estimators": study.best_params.get("n_estimators", 500),
        "random_state": 42,
        "verbosity": 0,
        **{k: v for k, v in study.best_params.items() if k != "n_estimators"}
    }
    
    return best_params


def generate_xgboost_forecasts(
    csv_path: str,
    n_trials: int = 50,
    train_test_split: float = 0.8,
    forecast_horizon: int = 12,
    forecast_start_date: str = None
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Generate forecasts using hyperparameter-tuned XGBoost models.
    
    Args:
        csv_path: Path to CSV file with Date, Branch, and Quantity columns
        n_trials: Number of Optuna trials for hyperparameter tuning
        train_test_split: Fraction of data to use for training
        forecast_horizon: Number of months to forecast ahead
        forecast_start_date: Optional date string (e.g., "2025-04-01") to start forecasts from this date
    
    Returns:
        Dictionary with forecasts from last data point and from FY start
    """
    # Load data
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
    
    # Filter from Sept 2021 onwards
    df = df[df['Date'] >= pd.Timestamp('2021-09-01')]
    
    # Group by date + branch and sum quantities
    grouped = (
        df.groupby(['Date', 'Branch'])['Quantity']
        .sum()
        .reset_index()
        .sort_values('Date')
    )
    
    # Determine FY start
    now = datetime.now()
    current_year = now.year
    current_month = now.month - 1
    fy_start = datetime(current_year, 4, 1)
    if current_month < 3:
        fy_start = datetime(current_year - 1, 4, 1)
    
    branches = grouped['Branch'].unique()
    
    from_last_data_point = defaultdict(list)
    from_fy_start = defaultdict(list)
    
    # Feature columns (excluding Date, Branch, Quantity)
    feature_columns = [
        'year', 'month', 'quarter', 'day_of_year', 'week_of_year',
        'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
        'lag_1', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_std_3', 'rolling_min_3', 'rolling_max_3',
        'rolling_mean_6', 'rolling_std_6', 'rolling_min_6', 'rolling_max_6',
        'rolling_mean_12', 'rolling_std_12', 'rolling_min_12', 'rolling_max_12',
        'ema_3', 'ema_6', 'ema_12',
        'trend', 'diff', 'pct_change',
        # New decomposition-style features
        'seasonal_index', 'recent_12m_avg', 'baseline_forecast',
        'monthly_historical_avg', 'recent_to_monthly_ratio', 'yoy_growth'
    ]
    
    for branch in branches:
        branch_data = grouped[grouped['Branch'] == branch].copy()
        if branch_data.empty:
            continue
        
        # Aggregate by month
        branch_data['year'] = branch_data['Date'].dt.year
        branch_data['month'] = branch_data['Date'].dt.month
        monthly = (
            branch_data.groupby(['year', 'month'])['Quantity']
            .sum()
            .reset_index()
        )
        monthly['Date'] = pd.to_datetime(
            monthly[['year', 'month']].assign(day=1)
        )
        monthly = monthly.sort_values('Date').reset_index(drop=True)
        
        if monthly.empty or len(monthly) < 12:
            continue
        
        last_date = monthly['Date'].max()
        monthly_before_fy = monthly[monthly['Date'] < fy_start].copy()
        
        # Calculate seasonal indices (same way as decomposition method)
        monthly_by_month = (
            monthly.groupby(monthly['Date'].dt.month)['Quantity']
            .apply(list)
            .to_dict()
        )
        seasonal_avg_by_month = {
            month: sum(vals) / len(vals)
            for month, vals in monthly_by_month.items()
        }
        overall_avg = sum(seasonal_avg_by_month.values()) / len(seasonal_avg_by_month) if seasonal_avg_by_month else monthly['Quantity'].mean()
        seasonal_indices = {
            m: (v / overall_avg if overall_avg > 0 else 1)
            for m, v in seasonal_avg_by_month.items()
        }
        
        # Store recent average (for decomposition-style calibration)
        recent_12m_avg = monthly.tail(12)['Quantity'].mean()
        
        # Create features
        monthly = create_time_series_features(monthly)
        
        # Prepare training data
        train_size = int(len(monthly) * train_test_split)
        train_data = monthly.iloc[:train_size].copy()
        val_data = monthly.iloc[train_size:].copy()
        
        if len(train_data) < 12 or len(val_data) == 0:
            continue
        
        # Prepare features and target
        X_train = train_data[feature_columns].values
        y_train = train_data['Quantity'].values
        X_val = val_data[feature_columns].values
        y_val = val_data['Quantity'].values
        
        # Optimize hyperparameters
        print(f"\nOptimizing XGBoost for branch {branch}...")
        best_params = optimize_xgboost_hyperparameters(
            X_train, y_train, X_val, y_val, n_trials=n_trials
        )
        
        # Train final model on all available data
        X_full = monthly[feature_columns].values
        y_full = monthly['Quantity'].values
        
        final_model = xgb.XGBRegressor(**best_params)
        final_model.fit(
            X_full, y_full,
            eval_set=[(X_val, y_val)],
        )
        
        # Forecast 1: From last data point (or specified start date)
        # Maintain forecast history for proper lag/rolling feature updates
        forecast_history = monthly[['Date', 'Quantity']].copy().reset_index(drop=True)
        
        # Determine forecast start date
        if forecast_start_date:
            forecast_start = pd.to_datetime(forecast_start_date)
            # If start date is before or equal to last_date, start from last_date + 1 month
            # Otherwise, start from the specified date
            if forecast_start <= last_date:
                forecast_start = last_date + pd.DateOffset(months=1)
        else:
            forecast_start = last_date + pd.DateOffset(months=1)
        
        forecast_dates = [
            forecast_start + pd.DateOffset(months=i) 
            for i in range(forecast_horizon)
        ]
        
        for i, forecast_date in enumerate(forecast_dates):
            # Create temporary dataframe with current history + new forecast date
            temp_df = forecast_history.copy()
            temp_df = pd.concat([
                temp_df,
                pd.DataFrame({'Date': [forecast_date], 'Quantity': [0]})
            ], ignore_index=True)
            
            # Create features for this temporary dataframe
            temp_df_with_features = create_time_series_features(temp_df)
            
            # Get the last row (forecast row) features
            forecast_features = temp_df_with_features.iloc[-1][feature_columns].copy()
            
            # Update time features (in case create_time_series_features missed some)
            forecast_features['year'] = forecast_date.year
            forecast_features['month'] = forecast_date.month
            forecast_features['quarter'] = forecast_date.quarter
            forecast_features['day_of_year'] = forecast_date.timetuple().tm_yday
            forecast_features['week_of_year'] = forecast_date.isocalendar()[1]
            
            # Update cyclical features
            forecast_features['month_sin'] = np.sin(2 * np.pi * forecast_date.month / 12)
            forecast_features['month_cos'] = np.cos(2 * np.pi * forecast_date.month / 12)
            forecast_features['quarter_sin'] = np.sin(2 * np.pi * forecast_date.quarter / 4)
            forecast_features['quarter_cos'] = np.cos(2 * np.pi * forecast_date.quarter / 4)
            
            # Update trend
            forecast_features['trend'] = len(forecast_history) + i + 1
            
            X_forecast = forecast_features[feature_columns].values.reshape(1, -1)
            forecast_qty = final_model.predict(X_forecast)[0]
            forecast_qty = max(0, forecast_qty)  # Ensure non-negative
            
            # Calculate decomposition-style forecast directly for calibration
            # This matches the decomposition method's logic exactly
            month = forecast_date.month
            seasonal_index = seasonal_indices.get(month, 1.0)
            
            # Decomposition-style forecast: recent_avg * seasonal_index * growth_factor
            growth_factor = math.pow(1.03, i / 12)  # 3% annual growth (same as decomp)
            decomp_style_forecast = recent_12m_avg * seasonal_index * growth_factor
            
            # Aggressive calibration: Use decomposition-style forecast as strong reference
            if decomp_style_forecast > 0:
                if forecast_qty < decomp_style_forecast * 0.5:
                    # If XGBoost is less than 50% of decomp, use 30% XGBoost + 70% decomp
                    forecast_qty = 0.3 * forecast_qty + 0.7 * decomp_style_forecast
                elif forecast_qty < decomp_style_forecast * 0.7:
                    # If 50-70% of decomp, use 50% XGBoost + 50% decomp
                    forecast_qty = 0.5 * forecast_qty + 0.5 * decomp_style_forecast
                elif forecast_qty < decomp_style_forecast * 0.85:
                    # If 70-85% of decomp, use 70% XGBoost + 30% decomp
                    forecast_qty = 0.7 * forecast_qty + 0.3 * decomp_style_forecast
                else:
                    # If close enough, just ensure we're at least 85% of decomp minimum
                    forecast_qty = max(forecast_qty, decomp_style_forecast * 0.85)
            
            from_last_data_point[branch].append({
                "date": forecast_date.strftime("%Y-%m-%d"),
                "quantity": round(forecast_qty, 1),
                "type": "forecast"
            })
            
            # Add forecast to history for next iteration
            forecast_history = pd.concat([
                forecast_history,
                pd.DataFrame({'Date': [forecast_date], 'Quantity': [forecast_qty]})
            ], ignore_index=True)
        
        # Forecast 2: From FY start
        if not monthly_before_fy.empty:
            fy_train = monthly_before_fy.copy()
            # Calculate FY-based recent average
            fy_recent_12m_avg = fy_train.tail(12)['Quantity'].mean() if len(fy_train) >= 12 else fy_train['Quantity'].mean()
            
            fy_train = create_time_series_features(fy_train)
            
            X_fy = fy_train[feature_columns].values
            y_fy = fy_train['Quantity'].values
            
            fy_model = xgb.XGBRegressor(**best_params)
            fy_model.fit(
                X_fy, y_fy,
                eval_set=[(X_val, y_val)],
            )
            
            fy_forecast_dates = [
                fy_start + pd.DateOffset(months=i)
                for i in range(forecast_horizon)
            ]
            
            # Maintain FY forecast history for proper lag/rolling feature updates
            fy_forecast_history = fy_train[['Date', 'Quantity']].copy().reset_index(drop=True)
            
            for i, forecast_date in enumerate(fy_forecast_dates):
                # Create temporary dataframe with current history + new forecast date
                temp_df = fy_forecast_history.copy()
                temp_df = pd.concat([
                    temp_df,
                    pd.DataFrame({'Date': [forecast_date], 'Quantity': [0]})
                ], ignore_index=True)
                
                # Create features for this temporary dataframe
                temp_df_with_features = create_time_series_features(temp_df)
                
                # Get the last row (forecast row) features
                forecast_features = temp_df_with_features.iloc[-1][feature_columns].copy()
                
                # Update time features
                forecast_features['year'] = forecast_date.year
                forecast_features['month'] = forecast_date.month
                forecast_features['quarter'] = forecast_date.quarter
                forecast_features['day_of_year'] = forecast_date.timetuple().tm_yday
                forecast_features['week_of_year'] = forecast_date.isocalendar()[1]
                
                # Update cyclical features
                forecast_features['month_sin'] = np.sin(2 * np.pi * forecast_date.month / 12)
                forecast_features['month_cos'] = np.cos(2 * np.pi * forecast_date.month / 12)
                forecast_features['quarter_sin'] = np.sin(2 * np.pi * forecast_date.quarter / 4)
                forecast_features['quarter_cos'] = np.cos(2 * np.pi * forecast_date.quarter / 4)
                
                # Update trend
                forecast_features['trend'] = len(fy_forecast_history) + i + 1
                
                X_forecast = forecast_features[feature_columns].values.reshape(1, -1)
                forecast_qty = fy_model.predict(X_forecast)[0]
                forecast_qty = max(0, forecast_qty)
                
                # Calculate decomposition-style forecast directly for calibration
                # Use FY-based recent average (already calculated above)
                month = forecast_date.month
                seasonal_index = seasonal_indices.get(month, 1.0)
                
                growth_factor = math.pow(1.03, i / 12)
                decomp_style_forecast = fy_recent_12m_avg * seasonal_index * growth_factor
                
                # Aggressive calibration: Use decomposition-style forecast as strong reference
                if decomp_style_forecast > 0:
                    if forecast_qty < decomp_style_forecast * 0.5:
                        forecast_qty = 0.3 * forecast_qty + 0.7 * decomp_style_forecast
                    elif forecast_qty < decomp_style_forecast * 0.7:
                        forecast_qty = 0.5 * forecast_qty + 0.5 * decomp_style_forecast
                    elif forecast_qty < decomp_style_forecast * 0.85:
                        forecast_qty = 0.7 * forecast_qty + 0.3 * decomp_style_forecast
                    else:
                        forecast_qty = max(forecast_qty, decomp_style_forecast * 0.85)
                
                from_fy_start[branch].append({
                    "date": forecast_date.strftime("%Y-%m-%d"),
                    "quantity": round(forecast_qty, 1),
                    "type": "forecast"
                })
                
                # Add forecast to history for next iteration
                fy_forecast_history = pd.concat([
                    fy_forecast_history,
                    pd.DataFrame({'Date': [forecast_date], 'Quantity': [forecast_qty]})
                ], ignore_index=True)
        else:
            # Use same model if no FY data
            for item in from_last_data_point[branch][:forecast_horizon]:
                from_fy_start[branch].append(item)
    
    return {
        "from_last_data_point": dict(from_last_data_point),
        "from_fy_start": dict(from_fy_start)
    }


def compare_forecast_methods(
    csv_path: str,
    xgb_n_trials: int = 50,
    forecast_type: str = "from_last_data_point",
    filter_date: str = None
) -> pd.DataFrame:
    """
    Compare the decomposition method and XGBoost method forecasts.
    
    Args:
        csv_path: Path to CSV file with Date, Branch, and Quantity columns
        xgb_n_trials: Number of Optuna trials for XGBoost hyperparameter tuning
        forecast_type: Either "from_last_data_point" or "from_fy_start"
        filter_date: Optional date string (e.g., "2025-04-01") to filter forecasts from this date onwards
    
    Returns:
        DataFrame with comparison of forecasts by branch and date
    """
    print("Generating forecasts using decomposition method...")
    result_decomp = generate_forecasts(
        csv_path,
        forecast_start_date=filter_date  # Pass filter_date as forecast_start_date
    )
    
    print("\nGenerating forecasts using XGBoost method (this may take a while)...")
    result_xgb = generate_xgboost_forecasts(
        csv_path, 
        n_trials=xgb_n_trials,
        forecast_start_date=filter_date  # Pass filter_date as forecast_start_date
    )
    
    # Combine results into a comparison DataFrame
    comparison_data = []
    
    # Get all branches that appear in either result
    branches_decomp = set(result_decomp.get(forecast_type, {}).keys())
    branches_xgb = set(result_xgb.get(forecast_type, {}).keys())
    all_branches = branches_decomp.union(branches_xgb)
    
    for branch in all_branches:
        decomp_forecasts = result_decomp.get(forecast_type, {}).get(branch, [])
        xgb_forecasts = result_xgb.get(forecast_type, {}).get(branch, [])
        
        # Convert to DataFrames for easier merging
        if decomp_forecasts:
            df_decomp = pd.DataFrame(decomp_forecasts)
            df_decomp['date'] = pd.to_datetime(df_decomp['date'])
            df_decomp = df_decomp.rename(columns={'quantity': 'decomp_forecast'})
        else:
            df_decomp = pd.DataFrame(columns=['date', 'decomp_forecast'])
        
        if xgb_forecasts:
            df_xgb = pd.DataFrame(xgb_forecasts)
            df_xgb['date'] = pd.to_datetime(df_xgb['date'])
            df_xgb = df_xgb.rename(columns={'quantity': 'xgb_forecast'})
        else:
            df_xgb = pd.DataFrame(columns=['date', 'xgb_forecast'])
        
        # Merge on date
        if not df_decomp.empty and not df_xgb.empty:
            df_merged = pd.merge(
                df_decomp[['date', 'decomp_forecast']],
                df_xgb[['date', 'xgb_forecast']],
                on='date',
                how='outer',
                suffixes=('_decomp', '_xgb')
            )
        elif not df_decomp.empty:
            df_merged = df_decomp[['date', 'decomp_forecast']].copy()
            df_merged['xgb_forecast'] = np.nan
        elif not df_xgb.empty:
            df_merged = df_xgb[['date', 'xgb_forecast']].copy()
            df_merged['decomp_forecast'] = np.nan
        else:
            continue
        
        df_merged['branch'] = branch
        df_merged = df_merged.sort_values('date')
        
        # Calculate differences
        df_merged['absolute_difference'] = (df_merged['xgb_forecast'] - df_merged['decomp_forecast']).abs()
        df_merged['percentage_difference'] = np.where(
            df_merged['decomp_forecast'] != 0,
            ((df_merged['xgb_forecast'] - df_merged['decomp_forecast']) / df_merged['decomp_forecast']) * 100,
            np.nan
        )
        df_merged['difference'] = df_merged['xgb_forecast'] - df_merged['decomp_forecast']
        
        comparison_data.append(df_merged)
    
    if not comparison_data:
        return pd.DataFrame()
    
    comparison_df = pd.concat(comparison_data, ignore_index=True)
    
    # Filter by date if specified
    if filter_date:
        filter_dt = pd.to_datetime(filter_date)
        comparison_df = comparison_df[comparison_df['date'] >= filter_dt].copy()
    
    return comparison_df


def print_comparison_summary(comparison_df: pd.DataFrame):
    """Print a formatted summary comparing the two forecasting methods."""
    if comparison_df.empty:
        print("No comparison data available.")
        return
    
    # Get date range
    min_date = comparison_df['date'].min()
    max_date = comparison_df['date'].max()
    
    print("\n" + "="*80)
    print("FORECAST METHOD COMPARISON SUMMARY")
    print("="*80)
    if min_date and max_date:
        print(f"Date Range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    # Overall statistics
    print("\n📊 OVERALL STATISTICS")
    print("-" * 80)
    
    valid_comparisons = comparison_df.dropna(subset=['decomp_forecast', 'xgb_forecast'])
    
    if len(valid_comparisons) > 0:
        print(f"Total forecast pairs compared: {len(valid_comparisons)}")
        print("\nDecomposition Method:")
        print(f"  Mean Forecast: {valid_comparisons['decomp_forecast'].mean():.2f}")
        print(f"  Std Forecast: {valid_comparisons['decomp_forecast'].std():.2f}")
        print(f"  Min Forecast: {valid_comparisons['decomp_forecast'].min():.2f}")
        print(f"  Max Forecast: {valid_comparisons['decomp_forecast'].max():.2f}")
        print("\nXGBoost Method:")
        print(f"  Mean Forecast: {valid_comparisons['xgb_forecast'].mean():.2f}")
        print(f"  Std Forecast: {valid_comparisons['xgb_forecast'].std():.2f}")
        print(f"  Min Forecast: {valid_comparisons['xgb_forecast'].min():.2f}")
        print(f"  Max Forecast: {valid_comparisons['xgb_forecast'].max():.2f}")
        
        print("\nDifference Statistics:")
        print(f"  Mean Absolute Difference: {valid_comparisons['absolute_difference'].mean():.2f}")
        print(f"  Mean Percentage Difference: {valid_comparisons['percentage_difference'].abs().mean():.2f}%")
        print(f"  Median Absolute Difference: {valid_comparisons['absolute_difference'].median():.2f}")
        print(f"  Max Absolute Difference: {valid_comparisons['absolute_difference'].max():.2f}")
        print(f"  Mean Difference (XGB - Decomp): {valid_comparisons['difference'].mean():.2f}")
    
    # Branch-level statistics
    print("\n\n📍 BRANCH-LEVEL STATISTICS")
    print("-" * 80)
    
    branch_stats = []
    for branch in comparison_df['branch'].unique():
        branch_data = comparison_df[comparison_df['branch'] == branch].dropna(
            subset=['decomp_forecast', 'xgb_forecast']
        )
        
        if len(branch_data) > 0:
            stats = {
                'branch': branch,
                'decomp_mean': branch_data['decomp_forecast'].mean(),
                'xgb_mean': branch_data['xgb_forecast'].mean(),
                'mean_abs_diff': branch_data['absolute_difference'].mean(),
                'mean_pct_diff': branch_data['percentage_difference'].abs().mean(),
                'count': len(branch_data)
            }
            branch_stats.append(stats)
    
    if branch_stats:
        branch_df = pd.DataFrame(branch_stats)
        branch_df = branch_df.sort_values('mean_abs_diff', ascending=False)
        
        print("\nBranches sorted by mean absolute difference:")
        print(branch_df.to_string(index=False))
        
        print("\n\nBranches where XGBoost forecasts higher on average:")
        higher_xgb = branch_df[branch_df['xgb_mean'] > branch_df['decomp_mean']]
        if len(higher_xgb) > 0:
            print(higher_xgb[['branch', 'decomp_mean', 'xgb_mean', 'mean_abs_diff']].to_string(index=False))
        
        print("\nBranches where Decomposition forecasts higher on average:")
        higher_decomp = branch_df[branch_df['decomp_mean'] > branch_df['xgb_mean']]
        if len(higher_decomp) > 0:
            print(higher_decomp[['branch', 'decomp_mean', 'xgb_mean', 'mean_abs_diff']].to_string(index=False))
    
    # Time-series comparison (first few forecast periods)
    print("\n\n📅 DETAILED COMPARISON (Sample Forecasts)")
    print("-" * 80)
    
    sample_df = comparison_df.dropna(subset=['decomp_forecast', 'xgb_forecast']).head(20)
    if len(sample_df) > 0:
        display_cols = ['branch', 'date', 'decomp_forecast', 'xgb_forecast', 
                       'difference', 'percentage_difference']
        print(sample_df[display_cols].to_string(index=False))
    
    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80 + "\n")


# --- Example usage ---
if __name__ == "__main__":
    csv_path = "./data/merged_filter_ingestion_2024.csv"
    
    # Compare forecasts from last data point
    print("Comparing forecasting methods (from_last_data_point)...")
    print("Filtering forecasts from 2025-04-01 onwards...")
    comparison_df = compare_forecast_methods(
        csv_path,
        xgb_n_trials=50,
        forecast_type="from_last_data_point",
        filter_date="2025-04-01"
    )
    
    # Print summary
    print_comparison_summary(comparison_df)
    
    # Optionally save comparison to CSV
    if not comparison_df.empty:
        output_path = "./data/forecast_comparison_2025_04_onwards.csv"
        comparison_df.to_csv(output_path, index=False)
        print(f"\n💾 Comparison saved to: {output_path}")
