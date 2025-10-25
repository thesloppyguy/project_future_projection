#!/usr/bin/env python3
"""
Phase 5: Regional Models Analysis

This script implements regional-level forecasting models for each state with two approaches:
- Approach A: Use only that state's weather data
- Approach B: Use all 5 states' weather data as features

Models: ARIMA, SARIMAX, Seasonal Decomposition, Holt-Winters, LSTM, GRU, Prophet
Total: 5 states × 6 models × 2 approaches = 60 models
"""

from datetime import datetime
import os
import joblib
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Time series and forecasting

# Handle pmdarima import gracefully
PMDARIMA_AVAILABLE = False
auto_arima = None


def safe_import_pmdarima():
    """Safely import pmdarima with error handling"""
    global PMDARIMA_AVAILABLE, auto_arima
    try:
        from pmdarima import auto_arima
        PMDARIMA_AVAILABLE = True
        print("pmdarima imported successfully")
        return auto_arima
    except Exception as e:
        print(
            f"Warning: pmdarima not available ({e}), will use alternative ARIMA parameter selection")
        PMDARIMA_AVAILABLE = False
        return None


# Try to import pmdarima
auto_arima = safe_import_pmdarima()

# Machine learning

# Utilities

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_regional_data():
    """
    Load weather and sales data for regional analysis
    """
    print("Loading regional data...")

    # Load weather data for all states
    weather_data = []
    states = ['AP', 'KA', 'KL', 'TL', 'TN']
    state_mapping = {
        'AP': 'Andhra Pradesh',
        'KA': 'Karnataka',
        'KL': 'Kerala',
        'TL': 'Telangana',
        'TN': 'Tamil Nadu'
    }

    for state_code in states:
        file_path = f'outputs/processed_weather_data/{state_code}_weather_timeseries.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df['State'] = state_mapping[state_code]
            weather_data.append(df)
            print(
                f"Loaded weather data for {state_mapping[state_code]}: {len(df)} records")
        else:
            print(f"Warning: Weather file not found for {state_code}")

    # Combine weather data
    weather_df = pd.concat(weather_data, ignore_index=True)
    weather_df = weather_df.sort_values(
        ['Date', 'State']).reset_index(drop=True)

    # Load sales data
    sales_df = pd.read_csv('data/monthly_sales_summary.csv')
    sales_df['Date'] = pd.to_datetime(
        sales_df['Year'].astype(str) + '-' +
        sales_df['Month'].astype(str) + '-01'
    )

    # Aggregate to regional level (State-Month)
    regional_sales = sales_df.groupby(['Date', 'State']).agg({
        'Monthly_Total_Sales': 'sum',
        'Number_of_Transactions': 'sum'
    }).reset_index()

    print(f"\nWeather data shape: {weather_df.shape}")
    print(f"Regional sales shape: {regional_sales.shape}")
    print(
        f"Date range: {weather_df['Date'].min()} to {weather_df['Date'].max()}")
    print(f"States: {weather_df['State'].unique()}")

    return weather_df, regional_sales


def prepare_modeling_data(weather_df, regional_sales):
    """
    Prepare data for regional modeling with train/validation splits
    """
    print("\nPreparing modeling data...")

    # Merge weather and sales data
    df_regional = regional_sales.merge(
        weather_df,
        on=['Date', 'State'],
        how='inner'
    )

    # Sort by date and state
    df_regional = df_regional.sort_values(
        ['Date', 'State']).reset_index(drop=True)

    # Define date ranges
    train_end = pd.to_datetime('2024-12-01')
    val_end = pd.to_datetime('2025-06-01')

    # Split data
    train_data = df_regional[df_regional['Date'] <= train_end].copy()
    val_data = df_regional[(df_regional['Date'] > train_end) & (
        df_regional['Date'] <= val_end)].copy()

    print(
        f"Training data: {len(train_data)} records ({train_data['Date'].min()} to {train_data['Date'].max()})")
    print(
        f"Validation data: {len(val_data)} records ({val_data['Date'].min()} to {val_data['Date'].max()})")

    # Create state-specific datasets
    state_data = {}
    states = df_regional['State'].unique()

    for state in states:
        state_train = train_data[train_data['State']
                                 == state].sort_values('Date')
        state_val = val_data[val_data['State'] == state].sort_values('Date')

        # Prepare time series
        train_ts = state_train.set_index('Date')['Monthly_Total_Sales']
        val_ts = state_val.set_index('Date')['Monthly_Total_Sales']

        # Prepare weather features
        weather_cols = ['Max_Temp', 'Min_Temp', 'Humidity', 'Wind_Speed']
        train_weather = state_train.set_index('Date')[weather_cols]
        val_weather = state_val.set_index('Date')[weather_cols]

        state_data[state] = {
            'train_ts': train_ts,
            'val_ts': val_ts,
            'train_weather': train_weather,
            'val_weather': val_weather,
            'train_data': state_train,
            'val_data': state_val,
            'weather_cols': weather_cols
        }

        print(f"{state}: Train {len(train_ts)} months, Val {len(val_ts)} months")

    return state_data, weather_df


def select_arima_parameters(train_ts):
    """Alternative ARIMA parameter selection when pmdarima is not available"""
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import itertools

    # Test for stationarity
    adf_result = adfuller(train_ts.dropna())
    is_stationary = adf_result[1] < 0.05

    # If not stationary, try differencing
    if not is_stationary:
        d = 1
        diff_ts = train_ts.diff().dropna()
    else:
        d = 0
        diff_ts = train_ts

    # Simple grid search for p and q
    best_aic = float('inf')
    best_order = (1, d, 1)
    best_seasonal_order = (1, 1, 1, 12)

    # Try common parameter combinations
    p_values = [0, 1, 2]
    q_values = [0, 1, 2]
    P_values = [0, 1]
    Q_values = [0, 1]

    for p, q, P, Q in itertools.product(p_values, q_values, P_values, Q_values):
        try:
            model = ARIMA(train_ts, order=(p, d, q),
                          seasonal_order=(P, 1, Q, 12))
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
                best_seasonal_order = (P, 1, Q, 12)
        except:
            continue

    return best_order, best_seasonal_order


def fit_arima_model(train_ts, val_ts, state_name, approach):
    """Fit ARIMA model with automatic parameter selection"""
    try:
        if PMDARIMA_AVAILABLE and auto_arima is not None:
            model = auto_arima(
                train_ts, seasonal=True, m=12,
                max_p=3, max_q=3, max_P=2, max_Q=2,
                suppress_warnings=True, stepwise=True, error_action='ignore'
            )
            order = model.order
            seasonal_order = model.seasonal_order
        else:
            # Use alternative parameter selection
            order, seasonal_order = select_arima_parameters(train_ts)

        fitted_model = ARIMA(train_ts, order=order,
                             seasonal_order=seasonal_order).fit()
        predictions = fitted_model.forecast(steps=len(val_ts))

        # Save model
        joblib.dump(
            fitted_model, f'outputs/models/regional_arima_{state_name.replace(" ", "_")}_approach_{approach}.pkl')

        return predictions, fitted_model
    except Exception as e:
        print(f"Error fitting ARIMA for {state_name}: {e}")
        return None, None


def fit_sarimax_model(train_ts, train_weather, val_ts, val_weather, state_name, approach):
    """Fit SARIMAX model with weather regressors"""
    try:
        if PMDARIMA_AVAILABLE and auto_arima is not None:
            model = auto_arima(
                train_ts, exogenous=train_weather, seasonal=True, m=12,
                max_p=3, max_q=3, max_P=2, max_Q=2,
                suppress_warnings=True, stepwise=True, error_action='ignore'
            )
            order = model.order
            seasonal_order = model.seasonal_order
        else:
            # Use alternative parameter selection
            order, seasonal_order = select_arima_parameters(train_ts)

        fitted_model = SARIMAX(train_ts, exog=train_weather, order=order,
                               seasonal_order=seasonal_order).fit(disp=False)
        predictions = fitted_model.forecast(
            steps=len(val_ts), exog=val_weather)

        # Save model
        joblib.dump(
            fitted_model, f'outputs/models/regional_sarimax_{state_name.replace(" ", "_")}_approach_{approach}.pkl')

        return predictions, fitted_model
    except Exception as e:
        print(f"Error fitting SARIMAX for {state_name}: {e}")
        return None, None


def seasonal_decomposition_forecast(train_ts, val_ts, state_name, approach):
    """Forecast using seasonal decomposition approach"""
    try:
        stl = STL(train_ts, seasonal=13)
        decomposition = stl.fit()

        # Forecast trend
        trend = decomposition.trend
        trend_dates = np.arange(len(trend)).reshape(-1, 1)
        trend_model = LinearRegression().fit(trend_dates, trend.values)

        val_trend_dates = np.arange(len(trend), len(
            trend) + len(val_ts)).reshape(-1, 1)
        trend_forecast = trend_model.predict(val_trend_dates)

        # Use last seasonal pattern
        seasonal = decomposition.seasonal
        last_year_seasonal = seasonal.iloc[-12:].values
        seasonal_forecast = np.tile(
            last_year_seasonal, (len(val_ts) // 12 + 1))[:len(val_ts)]

        forecast = trend_forecast + seasonal_forecast

        # Save model
        joblib.dump({
            'trend_model': trend_model,
            'seasonal_pattern': last_year_seasonal,
            'decomposition': decomposition
        }, f'outputs/models/regional_seasonal_decomp_{state_name.replace(" ", "_")}_approach_{approach}.pkl')

        return forecast
    except Exception as e:
        print(f"Error in seasonal decomposition for {state_name}: {e}")
        return None


def holt_winters_forecast(train_ts, val_ts, state_name, approach):
    """Fit Holt-Winters exponential smoothing model"""
    try:
        hw_model = ExponentialSmoothing(
            train_ts, trend='add', seasonal='add', seasonal_periods=12
        ).fit(optimized=True)

        forecast = hw_model.forecast(steps=len(val_ts))

        # Save model
        joblib.dump(
            hw_model, f'outputs/models/regional_holt_winters_{state_name.replace(" ", "_")}_approach_{approach}.pkl')

        return forecast
    except Exception as e:
        print(f"Error fitting Holt-Winters for {state_name}: {e}")
        return None


def prepare_lstm_data(train_ts, train_weather, val_ts, val_weather, lookback=12):
    """Prepare data for LSTM with lookback window"""
    train_combined = pd.concat([train_ts, train_weather], axis=1)
    val_combined = pd.concat([val_ts, val_weather], axis=1)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_combined)
    val_scaled = scaler.transform(val_combined)

    def create_sequences(data, lookback):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled, lookback)
    X_val, y_val = create_sequences(val_scaled, lookback)

    return X_train, y_train, X_val, y_val, scaler


def fit_lstm_model(train_ts, train_weather, val_ts, val_weather, state_name, approach):
    """Fit LSTM neural network model"""
    try:
        X_train, y_train, X_val, y_val, scaler = prepare_lstm_data(
            train_ts, train_weather, val_ts, val_weather, lookback=12
        )

        if len(X_train) == 0 or len(X_val) == 0:
            print(f"Insufficient data for LSTM training in {state_name}")
            return None

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(
                X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100, batch_size=16,
            callbacks=[early_stopping], verbose=0
        )

        lstm_pred = model.predict(X_val, verbose=0).flatten()

        # Inverse transform predictions
        dummy_array = np.zeros((len(lstm_pred), X_train.shape[2]))
        dummy_array[:, 0] = lstm_pred
        lstm_pred_original = scaler.inverse_transform(dummy_array)[:, 0]

        # Save model and scaler
        model.save(
            f'outputs/models/regional_lstm_{state_name.replace(" ", "_")}_approach_{approach}.h5')
        joblib.dump(
            scaler, f'outputs/models/regional_lstm_scaler_{state_name.replace(" ", "_")}_approach_{approach}.pkl')

        return lstm_pred_original
    except Exception as e:
        print(f"Error fitting LSTM for {state_name}: {e}")
        return None


def fit_gru_model(train_ts, train_weather, val_ts, val_weather, state_name, approach):
    """Fit GRU neural network model"""
    try:
        X_train, y_train, X_val, y_val, scaler = prepare_lstm_data(
            train_ts, train_weather, val_ts, val_weather, lookback=12
        )

        if len(X_train) == 0 or len(X_val) == 0:
            print(f"Insufficient data for GRU training in {state_name}")
            return None

        model = Sequential([
            GRU(64, return_sequences=True, input_shape=(
                X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100, batch_size=16,
            callbacks=[early_stopping], verbose=0
        )

        gru_pred = model.predict(X_val, verbose=0).flatten()

        # Inverse transform predictions
        dummy_array = np.zeros((len(gru_pred), X_train.shape[2]))
        dummy_array[:, 0] = gru_pred
        gru_pred_original = scaler.inverse_transform(dummy_array)[:, 0]

        # Save model and scaler
        model.save(
            f'outputs/models/regional_gru_{state_name.replace(" ", "_")}_approach_{approach}.h5')
        joblib.dump(
            scaler, f'outputs/models/regional_gru_scaler_{state_name.replace(" ", "_")}_approach_{approach}.pkl')

        return gru_pred_original
    except Exception as e:
        print(f"Error fitting GRU for {state_name}: {e}")
        return None


def fit_prophet_model(train_data, val_data, state_name, approach):
    """Fit Prophet model with weather regressors"""
    try:
        weather_cols = ['Max_Temp', 'Min_Temp', 'Humidity', 'Wind_Speed']
        prophet_train = train_data[[
            'Date', 'Monthly_Total_Sales'] + weather_cols].copy()
        prophet_train.columns = ['ds', 'y'] + weather_cols

        model = Prophet(
            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
            seasonality_mode='additive'
        )

        for col in weather_cols:
            model.add_regressor(col)

        model.fit(prophet_train)

        prophet_val = val_data[['Date'] + weather_cols].copy()
        prophet_val.columns = ['ds'] + weather_cols

        forecast = model.predict(prophet_val)
        prophet_pred = forecast['yhat'].values

        # Save model
        joblib.dump(
            model, f'outputs/models/regional_prophet_{state_name.replace(" ", "_")}_approach_{approach}.pkl')

        return prophet_pred
    except Exception as e:
        print(f"Error fitting Prophet for {state_name}: {e}")
        return None


def prepare_all_states_weather_data(state_data, all_weather_df):
    """Prepare weather data from all states as features for each state"""
    print("\nPreparing all states weather data for Approach B...")

    # Create wide format weather data (all states as columns)
    weather_wide = all_weather_df.pivot_table(
        index='Date',
        columns='State',
        values=['Max_Temp', 'Min_Temp', 'Humidity', 'Wind_Speed'],
        aggfunc='mean'
    )

    # Flatten column names
    weather_wide.columns = [
        f"{metric}_{state.replace(' ', '_')}" for metric, state in weather_wide.columns]
    weather_wide = weather_wide.reset_index()

    # Create state-specific datasets with all weather features
    all_states_data = {}

    for state in state_data.keys():
        train_data = state_data[state]['train_data']
        val_data = state_data[state]['val_data']

        # Merge with all weather data
        train_merged = train_data.merge(weather_wide, on='Date', how='inner')
        val_merged = val_data.merge(weather_wide, on='Date', how='inner')

        # Prepare time series
        train_ts = train_merged.set_index('Date')['Monthly_Total_Sales']
        val_ts = val_merged.set_index('Date')['Monthly_Total_Sales']

        # Prepare all weather features
        weather_cols = [col for col in weather_wide.columns if col != 'Date']
        train_weather = train_merged.set_index('Date')[weather_cols]
        val_weather = val_merged.set_index('Date')[weather_cols]

        all_states_data[state] = {
            'train_ts': train_ts,
            'val_ts': val_ts,
            'train_weather': train_weather,
            'val_weather': val_weather,
            'train_data': train_merged,
            'val_data': val_merged,
            'weather_cols': weather_cols
        }

        print(f"{state}: {len(weather_cols)} weather features from all states")

    return all_states_data


def train_approach_a_models(state_data):
    """Train all models for Approach A (own weather data only)"""
    print("\nTraining Approach A models (own weather data only)...")
    print("=" * 60)

    approach_a_results = {}

    for state in state_data.keys():
        print(f"\nTraining models for {state}...")

        data = state_data[state]
        train_ts = data['train_ts']
        val_ts = data['val_ts']
        train_weather = data['train_weather']
        val_weather = data['val_weather']
        train_data = data['train_data']
        val_data = data['val_data']

        state_results = {}

        # Train all 6 models
        print(f"  - ARIMA...")
        arima_pred, _ = fit_arima_model(train_ts, val_ts, state, 'A')
        state_results['ARIMA'] = arima_pred

        print(f"  - SARIMAX...")
        sarimax_pred, _ = fit_sarimax_model(
            train_ts, train_weather, val_ts, val_weather, state, 'A')
        state_results['SARIMAX'] = sarimax_pred

        print(f"  - Seasonal Decomposition...")
        decomp_pred = seasonal_decomposition_forecast(
            train_ts, val_ts, state, 'A')
        state_results['Seasonal_Decomp'] = decomp_pred

        print(f"  - Holt-Winters...")
        hw_pred = holt_winters_forecast(train_ts, val_ts, state, 'A')
        state_results['Holt_Winters'] = hw_pred

        print(f"  - LSTM...")
        lstm_pred = fit_lstm_model(
            train_ts, train_weather, val_ts, val_weather, state, 'A')
        state_results['LSTM'] = lstm_pred

        print(f"  - GRU...")
        gru_pred = fit_gru_model(
            train_ts, train_weather, val_ts, val_weather, state, 'A')
        state_results['GRU'] = gru_pred

        print(f"  - Prophet...")
        prophet_pred = fit_prophet_model(train_data, val_data, state, 'A')
        state_results['Prophet'] = prophet_pred

        approach_a_results[state] = {
            'predictions': state_results,
            'actual': val_ts.values,
            'dates': val_ts.index
        }

        print(f"  ✅ Completed {state}")

    print("\n" + "=" * 60)
    print("Approach A training completed!")
    return approach_a_results


def train_approach_b_models(all_states_data):
    """Train all models for Approach B (all states weather data)"""
    print("\nTraining Approach B models (all states weather data)...")
    print("=" * 60)

    approach_b_results = {}

    for state in all_states_data.keys():
        print(f"\nTraining models for {state}...")

        data = all_states_data[state]
        train_ts = data['train_ts']
        val_ts = data['val_ts']
        train_weather = data['train_weather']
        val_weather = data['val_weather']
        train_data = data['train_data']
        val_data = data['val_data']

        state_results = {}

        # Train all 6 models
        print(f"  - ARIMA...")
        arima_pred, _ = fit_arima_model(train_ts, val_ts, state, 'B')
        state_results['ARIMA'] = arima_pred

        print(f"  - SARIMAX...")
        sarimax_pred, _ = fit_sarimax_model(
            train_ts, train_weather, val_ts, val_weather, state, 'B')
        state_results['SARIMAX'] = sarimax_pred

        print(f"  - Seasonal Decomposition...")
        decomp_pred = seasonal_decomposition_forecast(
            train_ts, val_ts, state, 'B')
        state_results['Seasonal_Decomp'] = decomp_pred

        print(f"  - Holt-Winters...")
        hw_pred = holt_winters_forecast(train_ts, val_ts, state, 'B')
        state_results['Holt_Winters'] = hw_pred

        print(f"  - LSTM...")
        lstm_pred = fit_lstm_model(
            train_ts, train_weather, val_ts, val_weather, state, 'B')
        state_results['LSTM'] = lstm_pred

        print(f"  - GRU...")
        gru_pred = fit_gru_model(
            train_ts, train_weather, val_ts, val_weather, state, 'B')
        state_results['GRU'] = gru_pred

        print(f"  - Prophet...")
        prophet_pred = fit_prophet_model(train_data, val_data, state, 'B')
        state_results['Prophet'] = prophet_pred

        approach_b_results[state] = {
            'predictions': state_results,
            'actual': val_ts.values,
            'dates': val_ts.index
        }

        print(f"  ✅ Completed {state}")

    print("\n" + "=" * 60)
    print("Approach B training completed!")
    return approach_b_results


def evaluate_models(predictions_dict, actual_values, dates, state_name, approach):
    """Evaluate model performance and return metrics"""
    results = []

    for model_name, pred in predictions_dict.items():
        if pred is not None and len(pred) == len(actual_values):
            # Calculate metrics
            mae = np.mean(np.abs(pred - actual_values))
            rmse = np.sqrt(np.mean((pred - actual_values) ** 2))
            mape = np.mean(
                np.abs((actual_values - pred) / actual_values)) * 100

            results.append({
                'State': state_name,
                'Model': model_name,
                'Approach': approach,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            })

    return results


def create_comparison_visualizations(results_df):
    """Create comprehensive comparison visualizations"""
    print("\nCreating comparison visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        'Regional Models Performance Comparison: Approach A vs B', fontsize=16, y=0.98)

    # 1. MAE comparison by state
    mae_pivot = results_df.pivot_table(
        index=['State', 'Model'], columns='Approach', values='MAE')
    mae_pivot.plot(kind='bar', ax=axes[0, 0], width=0.8)
    axes[0, 0].set_title('MAE Comparison by State and Model')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].legend(['Approach A (Own Weather)', 'Approach B (All Weather)'])
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. RMSE comparison by state
    rmse_pivot = results_df.pivot_table(
        index=['State', 'Model'], columns='Approach', values='RMSE')
    rmse_pivot.plot(kind='bar', ax=axes[0, 1], width=0.8)
    axes[0, 1].set_title('RMSE Comparison by State and Model')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].legend(['Approach A (Own Weather)', 'Approach B (All Weather)'])
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. MAPE comparison by state
    mape_pivot = results_df.pivot_table(
        index=['State', 'Model'], columns='Approach', values='MAPE')
    mape_pivot.plot(kind='bar', ax=axes[0, 2], width=0.8)
    axes[0, 2].set_title('MAPE Comparison by State and Model')
    axes[0, 2].set_ylabel('MAPE (%)')
    axes[0, 2].legend(['Approach A (Own Weather)', 'Approach B (All Weather)'])
    axes[0, 2].tick_params(axis='x', rotation=45)

    # 4. Best model per state (by MAE)
    best_models = results_df.loc[results_df.groupby('State')['MAE'].idxmin()]
    best_models_pivot = best_models.pivot_table(
        index='State', columns='Approach', values='Model', aggfunc='first')
    best_models_pivot.plot(kind='bar', ax=axes[1, 0], width=0.8)
    axes[1, 0].set_title('Best Model per State (by MAE)')
    axes[1, 0].set_ylabel('Model')
    axes[1, 0].legend(['Approach A (Own Weather)', 'Approach B (All Weather)'])
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 5. Average performance by approach
    avg_performance = results_df.groupby(
        'Approach')[['MAE', 'RMSE', 'MAPE']].mean()
    avg_performance.plot(kind='bar', ax=axes[1, 1], width=0.8)
    axes[1, 1].set_title('Average Performance by Approach')
    axes[1, 1].set_ylabel('Metric Value')
    axes[1, 1].legend(['MAE', 'RMSE', 'MAPE'])
    axes[1, 1].tick_params(axis='x', rotation=0)

    # 6. Model performance heatmap
    model_performance = results_df.groupby(['Model', 'Approach'])[
        'MAE'].mean().unstack()
    sns.heatmap(model_performance, annot=True,
                fmt='.2f', cmap='YlOrRd', ax=axes[1, 2])
    axes[1, 2].set_title('Model Performance Heatmap (MAE)')
    axes[1, 2].set_xlabel('Approach')
    axes[1, 2].set_ylabel('Model')

    plt.tight_layout()
    plt.savefig('outputs/charts/regional_models_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print("Comparison chart saved to: outputs/charts/regional_models_comparison.png")


def main():
    """Main execution function"""
    print("=" * 80)
    print("PHASE 5: REGIONAL MODELS ANALYSIS")
    print("=" * 80)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Started at: {datetime.now()}")

    # Ensure output directories exist
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/charts', exist_ok=True)

    try:
        # 1. Load and prepare data
        weather_df, regional_sales = load_regional_data()
        state_data, all_weather_df = prepare_modeling_data(
            weather_df, regional_sales)

        # 2. Train Approach A models
        approach_a_results = train_approach_a_models(state_data)

        # 3. Prepare all states weather data and train Approach B models
        all_states_data = prepare_all_states_weather_data(
            state_data, all_weather_df)
        approach_b_results = train_approach_b_models(all_states_data)

        # 4. Evaluate all models
        print("\nEvaluating all models...")
        all_results = []

        print("Evaluating Approach A models...")
        for state, data in approach_a_results.items():
            results = evaluate_models(
                data['predictions'],
                data['actual'],
                data['dates'],
                state,
                'A'
            )
            all_results.extend(results)

        print("Evaluating Approach B models...")
        for state, data in approach_b_results.items():
            results = evaluate_models(
                data['predictions'],
                data['actual'],
                data['dates'],
                state,
                'B'
            )
            all_results.extend(results)

        # Create results dataframe
        results_df = pd.DataFrame(all_results)

        print("\nModel Performance Summary:")
        print("=" * 80)
        print(results_df.groupby(['State', 'Approach', 'Model'])[
              ['MAE', 'RMSE', 'MAPE']].mean().round(2))

        # 5. Create visualizations
        create_comparison_visualizations(results_df)

        # 6. Detailed analysis
        print("\nDetailed Performance Analysis:")
        print("=" * 80)

        # Best approach per state
        print("\nBest Approach per State (by average MAE):")
        state_avg = results_df.groupby(['State', 'Approach'])[
            'MAE'].mean().unstack()
        best_approach = state_avg.idxmin(axis=1)
        print(best_approach)

        # Best model per state
        print("\nBest Model per State (by MAE):")
        best_model_per_state = results_df.loc[results_df.groupby('State')[
            'MAE'].idxmin()]
        print(best_model_per_state[['State', 'Model',
              'Approach', 'MAE', 'RMSE', 'MAPE']])

        # Overall best models
        print("\nOverall Best Models (Top 10 by MAE):")
        top_models = results_df.nsmallest(10, 'MAE')
        print(
            top_models[['State', 'Model', 'Approach', 'MAE', 'RMSE', 'MAPE']])

        # Approach comparison summary
        print("\nApproach Comparison Summary:")
        approach_summary = results_df.groupby(
            'Approach')[['MAE', 'RMSE', 'MAPE']].agg(['mean', 'std']).round(3)
        print(approach_summary)

        # Save detailed results
        results_df.to_csv('outputs/regional_models_results.csv', index=False)
        print("\nDetailed results saved to: outputs/regional_models_results.csv")

        print("\n" + "=" * 80)
        print("PHASE 5 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Total models trained: {len(results_df)}")
        print(f"Models saved to: outputs/models/regional_*")
        print(f"Results saved to: outputs/regional_models_results.csv")
        print(f"Charts saved to: outputs/charts/regional_models_comparison.png")
        print(f"Completed at: {datetime.now()}")

    except Exception as e:
        print(f"\nError in regional models analysis: {e}")
        raise


if __name__ == "__main__":
    main()
