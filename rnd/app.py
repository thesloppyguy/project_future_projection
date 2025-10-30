# =============================================================================
# 1. DATA PREPROCESSING AND FEATURE ENGINEERING FOR ML MODELS
# =============================================================================

import time
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import warnings
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from prophet import Prophet
from tensorflow import keras
from tensorflow.keras import layers
from itertools import product
try:
    import keras_tuner as kt
    HAVE_KT = True
except ImportError:
    HAVE_KT = False

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

print("="*80)
print("MACHINE LEARNING MODELS IMPLEMENTATION")
print("="*80)

warnings.filterwarnings('ignore')

# Load the processed data
print("\nLoading processed data...")
refactored_df = pd.read_csv('./data/refactored_df.csv')
final_df = pd.read_csv('./data/Final Sales.csv')

# Convert date columns
refactored_df['Date'] = pd.to_datetime(refactored_df['Date'])
final_df['Date'] = pd.to_datetime(final_df['Date'])
refactored_df['Branch'].replace({'SBD1': 'VJW'}, inplace=True)
final_df['Branch'].replace({'BANGALORE': 'BLR'}, inplace=True)
final_df['Branch'].replace({'CHENNAI': 'MAA'}, inplace=True)
final_df['Branch'].replace({'HYDERABAD': 'SBD'}, inplace=True)
final_df['Branch'].replace({'COCHIN': 'COK'}, inplace=True)
final_df['Branch'].replace({'VIJAYAWADA': 'VJW'}, inplace=True)

print("Data loaded successfully!")
print(f"Sales data shape: {refactored_df.shape}")

main_df = (
    refactored_df.groupby([refactored_df['Date'].dt.to_period(
        'M').dt.to_timestamp().rename('MonthStart'), 'Branch'])
    .agg({'Qty': 'sum'})
    .reset_index()
)
main_df.columns = ['Date', 'Branch', 'Qty']
# main_df = main_df.drop(columns=['Month'])

final_df = (
    final_df.groupby([final_df['Date'].dt.to_period(
        'M').dt.to_timestamp().rename('MonthStart'), 'Branch'])
    .agg({'Sales Qty.': 'sum'})
    .reset_index()
)
final_df.columns = ['Date', 'Branch', 'Qty']

general_df = pd.concat([main_df, final_df])
general_df = general_df.sort_values(['Date']).reset_index(drop=True)


# =============================================================================
# 2. SPLIT DATASET INTO BRANCHES
# =============================================================================

branches = general_df['Branch'].unique()
branch_dfs = {}
for branch_name in branches:
    branch_dfs[branch_name] = general_df[general_df['Branch'] == branch_name]

# =============================================================================
# 3. COMPREHENSIVE FEATURE ENGINEERING
# =============================================================================


def create_ml_features(branch_df, anchor_month=3, season_years_back=1, lags=[1, 2, 3, 6, 12], rolling_windows=[3, 6, 12]):
    """
    Create features for ML models, including advanced time-series features:
     - Seasonal index (SI) per month (as before),
     - Lag features for Qty (e.g., 1, 2, 3, 6, 12 months),
     - Rolling means and stds (same windows as lags, back-filled),
     - Month-over-month delta feature,
     - Cumulative sum and mean (YTD by financial year, per branch).
    Keeps results per-branch/month, avoids smoothing across branches.
    """
    df = branch_df.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    # Financial year starts in April. FY = Year if Month >= 4, otherwise Year-1
    df['FinancialYear'] = df.apply(
        lambda r: r['Year'] if r['Month'] >= 4 else r['Year'] - 1, axis=1)
    # Group for SI
    fy_monthly = df.groupby(['FinancialYear', 'Month'])[
        'Qty'].sum().reset_index(name='FYMonthTotal')
    # SI as before
    fy_list = sorted(df['FinancialYear'].unique())
    si_records = []
    for fy in fy_list:
        season_fys = [fy - i for i in range(season_years_back, 0, -1)]
        if all(season_fys[si] in fy_monthly['FinancialYear'].unique() for si in range(season_years_back)):
            season_data = fy_monthly[fy_monthly['FinancialYear'].isin(
                season_fys)]
            month_means = season_data.groupby(
                'Month')['FYMonthTotal'].mean().reset_index()
            march_row = month_means[month_means['Month'] == anchor_month]
            if not march_row.empty and march_row['FYMonthTotal'].iloc[0] != 0:
                anchor_val = march_row['FYMonthTotal'].iloc[0]
            else:
                anchor_val = 1.0
            month_means['SeasonalIndex'] = month_means['FYMonthTotal'] / anchor_val
            month_means['FinancialYear'] = fy
            si_records.append(
                month_means[['FinancialYear', 'Month', 'SeasonalIndex']])
    si_all = pd.concat(si_records) if si_records else pd.DataFrame(
        columns=['FinancialYear', 'Month', 'SeasonalIndex'])
    df = df.merge(si_all, on=['FinancialYear', 'Month'], how='left')
    # ============== Advanced FE ==============
    df = df.sort_values('Date')  # for lag/rolling
    # Lags & rolling: group-wise per-branch if this were concat, but here 1 branch per call.
    for lag in lags:
        df[f'Qty_lag_{lag}'] = df['Qty'].shift(lag)
    for window in rolling_windows:
        df[f'Qty_roll_mean_{window}'] = df['Qty'].rolling(
            window).mean().shift(1)  # prevent leakage
        df[f'Qty_roll_std_{window}'] = df['Qty'].rolling(window).std().shift(1)
    # Month-over-month delta
    df['Qty_delta_1'] = df['Qty'] - df['Qty'].shift(1)
    # Cumulative sum/mean by FY
    df['Qty_ytd_sum'] = df.groupby('FinancialYear')['Qty'].cumsum()
    df['Qty_ytd_mean'] = df.groupby('FinancialYear')[
        'Qty'].expanding().mean().reset_index(level=0, drop=True)
    # ============== End Advanced FE ===========
    df.drop(['Branch'], axis=1, inplace=True)
    # Optionally, drop initial n rows if many NaNs from lag/rolling. Or just let ML/data prep handle it.
    return df


print(branches)
# Run all branches and models for multiple lookback configs
lookback_results = {}
for lookback in [1, 2, 3]:
    logger.info(f"Starting pipeline with season_years_back={lookback}")
    this_fe_branch_dfs = {}
    for branch_name, branch_df in branch_dfs.items():
        this_fe_branch_dfs[branch_name] = create_ml_features(
            branch_df, season_years_back=lookback)
    # Model results per branch
    lgb_results = {}
    xgb_results = {}
    catboost_results = {}
    prophet_results = {}
    keras_lstm_results = {}
    keras_gru_results = {}
    keras_rnn_results = {}
    keras_cnn_lstm_results = {}
    for branch_name, feature_branch in this_fe_branch_dfs.items():
        ml_data = prepare_ml_data(feature_branch)
        lgb_results[branch_name] = train_lightgbm_model(
            ml_data['X_train'], ml_data['y_train'],
            ml_data['X_val'], ml_data['y_val'],
            ml_data['X_test'], ml_data['y_test'],
            ml_data['feature_cols'])
        xgb_results[branch_name] = train_xgboost_model(
            ml_data['X_train'], ml_data['y_train'],
            ml_data['X_val'], ml_data['y_val'],
            ml_data['X_test'], ml_data['y_test'],
            ml_data['feature_cols'])
        catboost_results[branch_name] = train_catboost_model(
            ml_data['X_train'], ml_data['y_train'],
            ml_data['X_val'], ml_data['y_val'],
            ml_data['X_test'], ml_data['y_test'],
            ml_data['feature_cols'])

        # Prophet with extra regressors (all advanced features except target & date)
        reg_cols = [
            c for c in feature_branch.columns if c not in ['Date', 'Qty']]
        prophet_metrics = train_prophet_model(
            feature_branch, target_col='Qty', regressors=reg_cols, periods=12)
        prophet_results[branch_name] = prophet_metrics
        # Keras deep learning models
        keras_lstm_results[branch_name] = train_and_evaluate_keras_model(
            feature_branch, reg_cols, 'Qty', build_lstm_model)
        keras_gru_results[branch_name] = train_and_evaluate_keras_model(
            feature_branch, reg_cols, 'Qty', build_gru_model)
        keras_rnn_results[branch_name] = train_and_evaluate_keras_model(
            feature_branch, reg_cols, 'Qty', build_rnn_model)
        keras_cnn_lstm_results[branch_name] = train_and_evaluate_keras_model(
            feature_branch, reg_cols, 'Qty', build_cnn_lstm_model)

    lookback_results[lookback] = {
        'lgb': lgb_results,
        'xgb': xgb_results,
        'catboost': catboost_results,
        'prophet': prophet_results,
        'lstm': keras_lstm_results,
        'gru': keras_gru_results,
        'rnn': keras_rnn_results,
        'cnn_lstm': keras_cnn_lstm_results
    }
    logger.info(f"Results for season_years_back={lookback}")
    logger.info(f"LightGBM: {lgb_results}")
    logger.info(f"XGBoost: {xgb_results}")
    logger.info(f"CatBoost: {catboost_results}")
    logger.info(f"Prophet: {prophet_results}")
    logger.info(f"LSTM: {keras_lstm_results}")
    logger.info(f"GRU: {keras_gru_results}")
    logger.info(f"RNN: {keras_rnn_results}")
    logger.info(f"CNN-LSTM: {keras_cnn_lstm_results}")
    logger.info("-"*100)


# =============================================================================
# 4. LIGHTGBM MODEL IMPLEMENTATION
# =============================================================================
tscv = TimeSeriesSplit(n_splits=3)


def calculate_metrics(y_true, y_pred, model_name="Model"):
    """Calculate comprehensive evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def train_lightgbm_model(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    """Train LightGBM model with hyperparameter tuning"""
    baseline_params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train baseline model
    baseline_model = lgb.train(
        baseline_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )

    # Make predictions
    baseline_train_pred = baseline_model.predict(X_train)
    baseline_val_pred = baseline_model.predict(X_val)
    baseline_test_pred = baseline_model.predict(X_test)

    # Calculate metrics
    baseline_train_metrics = calculate_metrics(
        y_train, baseline_train_pred, "LightGBM Baseline Train")
    baseline_val_metrics = calculate_metrics(
        y_val, baseline_val_pred, "LightGBM Baseline Val")
    baseline_test_metrics = calculate_metrics(
        y_test, baseline_test_pred, "LightGBM Baseline Test")

    logger.info(
        f"LightGBM Baseline validation RMSE: {baseline_val_metrics['RMSE']:.4f}")
    logger.info(
        f"LightGBM Baseline validation R²: {baseline_val_metrics['R2']:.4f}")

    # Define parameter grid for tuning
    param_grid = {
        'num_leaves': [31, 50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'feature_fraction': [0.8, 0.9, 0.95, 1.0],
        'bagging_fraction': [0.8, 0.9, 0.95, 1.0],
        'bagging_freq': [5, 10, 15],
        'min_child_samples': [20, 30, 50],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0]
    }

    # Use RandomizedSearchCV for efficiency
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        boosting_type='gbdt',
        verbose=-1,
        random_state=42,
        n_estimators=1000
    )

    # Randomized search
    random_search = RandomizedSearchCV(
        lgb_model,
        param_grid,
        n_iter=50,  # Number of parameter settings sampled
        cv=tscv,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # Fit the model
    random_search.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

    # Get best parameters
    best_params = random_search.best_params_

    # Update parameters with best found
    final_params = baseline_params.copy()
    final_params.update(best_params)

    # Train final model
    final_model = lgb.train(
        final_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )

    # Make predictions with final model
    final_train_pred = final_model.predict(X_train)
    final_val_pred = final_model.predict(X_val)
    final_test_pred = final_model.predict(X_test)

    # Calculate final metrics
    final_train_metrics = calculate_metrics(
        y_train, final_train_pred, "LightGBM Final Train")
    final_val_metrics = calculate_metrics(
        y_val, final_val_pred, "LightGBM Final Val")
    final_test_metrics = calculate_metrics(
        y_test, final_test_pred, "LightGBM Final Test")

    logger.info(
        f"LightGBM Final validation RMSE: {final_val_metrics['RMSE']:.4f}")
    logger.info(f"LightGBM Final validation R²: {final_val_metrics['R2']:.4f}")

    # 4. Feature Importance Analysis
    logger.info("  Analyzing LightGBM feature importance...")
    feature_importance = final_model.feature_importance(importance_type='gain')
    feature_names = feature_cols

    # Create feature importance dataframe
    logger.debug(feature_importance)
    logger.debug(feature_names)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    logger.info(f"    Top 10 most important LightGBM features:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        logger.info(f"      {i+1}. {row['feature']}: {row['importance']:.4f}")

    # 5. Model Comparison
    logger.info("  LightGBM model performance comparison:")
    logger.info(f"    Baseline vs Final Model:")
    logger.info(
        f"      Validation RMSE: {baseline_val_metrics['RMSE']:.4f} → {final_val_metrics['RMSE']:.4f}")
    logger.info(
        f"      Validation R²: {baseline_val_metrics['R2']:.4f} → {final_val_metrics['R2']:.4f}")

    improvement_rmse = (
        (baseline_val_metrics['RMSE'] - final_val_metrics['RMSE']) / baseline_val_metrics['RMSE']) * 100
    improvement_r2 = (
        (final_val_metrics['R2'] - baseline_val_metrics['R2']) / abs(baseline_val_metrics['R2'])) * 100

    logger.info(f"      RMSE improvement: {improvement_rmse:.2f}%")
    logger.info(f"      R² improvement: {improvement_r2:.2f}%")

    return {
        'baseline_model': baseline_model,
        'final_model': final_model,
        'best_params': best_params,
        'baseline_metrics': {
            'train': baseline_train_metrics,
            'val': baseline_val_metrics,
            'test': baseline_test_metrics
        },
        'final_metrics': {
            'train': final_train_metrics,
            'val': final_val_metrics,
            'test': final_test_metrics
        },
        'feature_importance': importance_df,
        'predictions': {
            'baseline': {
                'train': baseline_train_pred,
                'val': baseline_val_pred,
                'test': baseline_test_pred
            },
            'final': {
                'train': final_train_pred,
                'val': final_val_pred,
                'test': final_test_pred
            }
        },
    }

# =============================================================================
# 5. XGBOOST MODEL IMPLEMENTATION
# =============================================================================


logger.info("\n5. XGBOOST MODEL IMPLEMENTATION")
logger.info("-" * 50)


def train_xgboost_model(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    """Train XGBoost model with hyperparameter tuning"""

    logger.info("Training XGBoost model...")

    # 1. Baseline XGBoost Model
    logger.info("  Training baseline XGBoost model...")
    start_time = time.time()

    baseline_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0
    }

    # Train baseline model
    baseline_model = xgb.XGBRegressor(**baseline_params)
    baseline_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Make predictions
    baseline_train_pred = baseline_model.predict(X_train)
    baseline_val_pred = baseline_model.predict(X_val)
    baseline_test_pred = baseline_model.predict(X_test)

    # Calculate metrics
    baseline_train_metrics = calculate_metrics(
        y_train, baseline_train_pred, "XGBoost Baseline Train")
    baseline_val_metrics = calculate_metrics(
        y_val, baseline_val_pred, "XGBoost Baseline Val")
    baseline_test_metrics = calculate_metrics(
        y_test, baseline_test_pred, "XGBoost Baseline Test")

    logger.info(
        f"    XGBoost Baseline validation RMSE: {baseline_val_metrics['RMSE']:.4f}")
    logger.info(
        f"    XGBoost Baseline validation R²: {baseline_val_metrics['R2']:.4f}")

    # 2. Hyperparameter Tuning
    logger.info("  Performing hyperparameter tuning...")
    start_time = time.time()

    # Define parameter grid for tuning
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'n_estimators': [500, 800, 1000, 1200],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bynode': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0, 2.0],
        'gamma': [0, 0.1, 0.5, 1.0],
        'min_child_weight': [1, 3, 5, 7]
    }

    # Use RandomizedSearchCV for efficiency
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='mae',
        random_state=42,
        verbosity=0
    )

    # Randomized search
    random_search = RandomizedSearchCV(
        xgb_model,
        param_grid,
        n_iter=50,  # Number of parameter settings sampled
        cv=tscv,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # Fit the model
    random_search.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)

    tuning_time = time.time() - start_time

    # Get best parameters
    best_params = random_search.best_params_
    logger.info(f"    Best parameters: {best_params}")
    logger.info(f"    Tuning time: {tuning_time:.2f} seconds")

    # 3. Train Final Model with Best Parameters
    logger.info("  Training final model with best parameters...")
    start_time = time.time()

    # Update parameters with best found
    final_params = baseline_params.copy()
    final_params.update(best_params)

    # Train final model
    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    final_time = time.time() - start_time

    # Make predictions with final model
    final_train_pred = final_model.predict(X_train)
    final_val_pred = final_model.predict(X_val)
    final_test_pred = final_model.predict(X_test)

    # Calculate final metrics
    final_train_metrics = calculate_metrics(
        y_train, final_train_pred, "XGBoost Final Train")
    final_val_metrics = calculate_metrics(
        y_val, final_val_pred, "XGBoost Final Val")
    final_test_metrics = calculate_metrics(
        y_test, final_test_pred, "XGBoost Final Test")

    logger.info(f"    Final XGBoost training time: {final_time:.2f} seconds")
    logger.info(
        f"    XGBoost Final validation RMSE: {final_val_metrics['RMSE']:.4f}")
    logger.info(
        f"    XGBoost Final validation R²: {final_val_metrics['R2']:.4f}")

    # 4. Feature Importance Analysis
    logger.info("  Analyzing XGBoost feature importance...")
    feature_importance = final_model.feature_importances_
    feature_names = feature_cols

    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    logger.info(f"    Top 10 most important XGBoost features:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        logger.info(f"      {i+1}. {row['feature']}: {row['importance']:.4f}")

    # 5. Model Comparison
    logger.info("  XGBoost model performance comparison:")
    logger.info(f"    Baseline vs Final Model:")
    logger.info(
        f"      Validation RMSE: {baseline_val_metrics['RMSE']:.4f} → {final_val_metrics['RMSE']:.4f}")
    logger.info(
        f"      Validation R²: {baseline_val_metrics['R2']:.4f} → {final_val_metrics['R2']:.4f}")

    improvement_rmse = (
        (baseline_val_metrics['RMSE'] - final_val_metrics['RMSE']) / baseline_val_metrics['RMSE']) * 100
    improvement_r2 = (
        (final_val_metrics['R2'] - baseline_val_metrics['R2']) / abs(baseline_val_metrics['R2'])) * 100

    logger.info(f"      RMSE improvement: {improvement_rmse:.2f}%")
    logger.info(f"      R² improvement: {improvement_r2:.2f}%")

    # 6. Advanced XGBoost Features
    logger.info("  Training advanced XGBoost model with additional features...")
    start_time = time.time()

    # Advanced parameters for better performance
    advanced_params = final_params.copy()
    advanced_params.update({
        'tree_method': 'hist',  # Use histogram-based algorithm
        'grow_policy': 'lossguide',  # Grow policy for better performance
        'max_leaves': 0,  # Let max_depth control tree size
        'max_bin': 256,  # Number of bins for histogram
        'predictor': 'cpu_predictor',  # Use CPU predictor
        'enable_categorical': False,  # Disable categorical features
        'interaction_constraints': None,  # No interaction constraints
        'monotone_constraints': None,  # No monotone constraints
    })

    # Train advanced model
    advanced_model = xgb.XGBRegressor(**advanced_params)
    advanced_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    advanced_time = time.time() - start_time

    # Make predictions with advanced model
    advanced_train_pred = advanced_model.predict(X_train)
    advanced_val_pred = advanced_model.predict(X_val)
    advanced_test_pred = advanced_model.predict(X_test)

    # Calculate advanced metrics
    advanced_train_metrics = calculate_metrics(
        y_train, advanced_train_pred, "XGBoost Advanced Train")
    advanced_val_metrics = calculate_metrics(
        y_val, advanced_val_pred, "XGBoost Advanced Val")
    advanced_test_metrics = calculate_metrics(
        y_test, advanced_test_pred, "XGBoost Advanced Test")

    logger.info(
        f"    Advanced XGBoost training time: {advanced_time:.2f} seconds")
    logger.info(
        f"    Advanced XGBoost validation RMSE: {advanced_val_metrics['RMSE']:.4f}")
    logger.info(
        f"    Advanced XGBoost validation R²: {advanced_val_metrics['R2']:.4f}")

    # Compare all XGBoost models
    logger.info("  All XGBoost models comparison:")
    logger.info(f"    Baseline RMSE: {baseline_val_metrics['RMSE']:.4f}")
    logger.info(f"    Final RMSE: {final_val_metrics['RMSE']:.4f}")
    logger.info(f"    Advanced RMSE: {advanced_val_metrics['RMSE']:.4f}")

    best_model = 'Advanced' if advanced_val_metrics['RMSE'] < final_val_metrics['RMSE'] else 'Final'
    logger.info(f"    Best XGBoost model: {best_model}")

    return {
        'baseline_model': baseline_model,
        'final_model': final_model,
        'advanced_model': advanced_model,
        'best_params': best_params,
        'baseline_metrics': {
            'train': baseline_train_metrics,
            'val': baseline_val_metrics,
            'test': baseline_test_metrics
        },
        'final_metrics': {
            'train': final_train_metrics,
            'val': final_val_metrics,
            'test': final_test_metrics
        },
        'advanced_metrics': {
            'train': advanced_train_metrics,
            'val': advanced_val_metrics,
            'test': advanced_test_metrics
        },
        'feature_importance': importance_df,
        'predictions': {
            'baseline': {
                'train': baseline_train_pred,
                'val': baseline_val_pred,
                'test': baseline_test_pred
            },
            'final': {
                'train': final_train_pred,
                'val': final_val_pred,
                'test': final_test_pred
            },
            'advanced': {
                'train': advanced_train_pred,
                'val': advanced_val_pred,
                'test': advanced_test_pred
            }
        },
    }

# =============================================================================
# 6. CATBOOST MODEL IMPLEMENTATION
# =============================================================================


def train_catboost_model(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    """Train CatBoost model with hyperparameter tuning"""

    logger.info("Training CatBoost model...")

    # 1. Baseline CatBoost Model
    logger.info("  Training baseline CatBoost model...")
    start_time = time.time()

    baseline_params = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 3,
        'bootstrap_type': 'Bayesian',
        'random_seed': 42,
        'od_type': 'Iter',
        'od_wait': 100,
        'verbose': False
    }

    # Train baseline model
    baseline_model = cb.CatBoostRegressor(**baseline_params)
    baseline_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=False
    )

    # Make predictions
    baseline_train_pred = baseline_model.predict(X_train)
    baseline_val_pred = baseline_model.predict(X_val)
    baseline_test_pred = baseline_model.predict(X_test)

    # Calculate metrics
    baseline_train_metrics = calculate_metrics(
        y_train, baseline_train_pred, "CatBoost Baseline Train")
    baseline_val_metrics = calculate_metrics(
        y_val, baseline_val_pred, "CatBoost Baseline Val")
    baseline_test_metrics = calculate_metrics(
        y_test, baseline_test_pred, "CatBoost Baseline Test")

    logger.info(
        f"    CatBoost Baseline validation RMSE: {baseline_val_metrics['RMSE']:.4f}")
    logger.info(
        f"    CatBoost Baseline validation R²: {baseline_val_metrics['R2']:.4f}")

    # 2. Hyperparameter Tuning
    logger.info("  Performing hyperparameter tuning...")
    start_time = time.time()

    # Define parameter grid for tuning
    param_grid = {
        'iterations': [500, 800, 1000, 1200],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'depth': [4, 5, 6, 7, 8],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'bootstrap_type': ['Bayesian', 'Bernoulli'],
        'bagging_temperature': [0, 0.5, 1.0],
        'random_strength': [0, 1, 2],
        'one_hot_max_size': [2, 10, 20],
        'leaf_estimation_method': ['Newton', 'Gradient'],
        'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
    }

    # Use RandomizedSearchCV for efficiency
    cb_model = cb.CatBoostRegressor(
        random_seed=42,
        od_type='Iter',
        od_wait=100,
        verbose=False
    )

    # Randomized search
    random_search = RandomizedSearchCV(
        cb_model,
        param_grid,
        n_iter=30,  # Number of parameter settings sampled
        cv=tscv,  # 3-fold cross-validation
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # Fit the model
    random_search.fit(X_train, y_train,
                      eval_set=(X_val, y_val),
                      early_stopping_rounds=100,
                      verbose=False)

    tuning_time = time.time() - start_time

    # Get best parameters
    best_params = random_search.best_params_
    logger.info(f"    Best parameters: {best_params}")
    logger.info(f"    Tuning time: {tuning_time:.2f} seconds")

    # 3. Train Final Model with Best Parameters
    logger.info("  Training final model with best parameters...")
    start_time = time.time()

    # Update parameters with best found
    final_params = baseline_params.copy()
    final_params.update(best_params)

    # Train final model
    final_model = cb.CatBoostRegressor(**final_params)
    final_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=False
    )

    final_time = time.time() - start_time

    # Make predictions with final model
    final_train_pred = final_model.predict(X_train)
    final_val_pred = final_model.predict(X_val)
    final_test_pred = final_model.predict(X_test)

    # Calculate final metrics
    final_train_metrics = calculate_metrics(
        y_train, final_train_pred, "CatBoost Final Train")
    final_val_metrics = calculate_metrics(
        y_val, final_val_pred, "CatBoost Final Val")
    final_test_metrics = calculate_metrics(
        y_test, final_test_pred, "CatBoost Final Test")

    logger.info(f"    Final CatBoost training time: {final_time:.2f} seconds")
    logger.info(
        f"    CatBoost Final validation RMSE: {final_val_metrics['RMSE']:.4f}")
    logger.info(
        f"    CatBoost Final validation R²: {final_val_metrics['R2']:.4f}")

    # 4. Feature Importance Analysis
    logger.info("  Analyzing CatBoost feature importance...")
    feature_importance = final_model.feature_importances_
    feature_names = feature_cols

    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    logger.info(f"    Top 10 most important CatBoost features:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        logger.info(f"      {i+1}. {row['feature']}: {row['importance']:.4f}")

    # 5. Model Comparison
    logger.info("  CatBoost model performance comparison:")
    logger.info(f"    Baseline vs Final Model:")
    logger.info(
        f"      Validation RMSE: {baseline_val_metrics['RMSE']:.4f} → {final_val_metrics['RMSE']:.4f}")
    logger.info(
        f"      Validation R²: {baseline_val_metrics['R2']:.4f} → {final_val_metrics['R2']:.4f}")

    improvement_rmse = (
        (baseline_val_metrics['RMSE'] - final_val_metrics['RMSE']) / baseline_val_metrics['RMSE']) * 100
    improvement_r2 = (
        (final_val_metrics['R2'] - baseline_val_metrics['R2']) / abs(baseline_val_metrics['R2'])) * 100

    logger.info(f"      RMSE improvement: {improvement_rmse:.2f}%")
    logger.info(f"      R² improvement: {improvement_r2:.2f}%")

    return {
        'baseline_model': baseline_model,
        'final_model': final_model,
        'best_params': best_params,
        'baseline_metrics': {
            'train': baseline_train_metrics,
            'val': baseline_val_metrics,
            'test': baseline_test_metrics
        },
        'final_metrics': {
            'train': final_train_metrics,
            'val': final_val_metrics,
            'test': final_test_metrics
        },
        'feature_importance': importance_df,
        'predictions': {
            'baseline': {
                'train': baseline_train_pred,
                'val': baseline_val_pred,
                'test': baseline_test_pred
            },
            'final': {
                'train': final_train_pred,
                'val': final_val_pred,
                'test': final_test_pred
            }
        },
    }


def prepare_ml_data(df, target_col='Qty'):
    """
    Prepare data for machine learning models with improved NaN handling
    """
    logger.info("Preparing data for ML models...")

    # Remove rows with missing target values
    df_clean = df[(df['Date'] >= '2020-04-01') & (df['Date'] <=
                                                  '2025-05-31')].dropna(subset=[target_col]).copy()
    logger.info(
        f"  After removing missing target values: {len(df_clean)} samples")

    # Check for any remaining missing values
    missing_before = df_clean.isnull().sum().sum()
    logger.info(f"  Missing values before cleaning: {missing_before}")

    # Fill any remaining missing values with appropriate strategies
    logger.info("  Filling remaining missing values...")

    # Fill numerical columns with median
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col != target_col and df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            logger.debug(
                f"    Filled {col} with median: {df_clean[col].median():.2f}")

    # Fill categorical columns with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode(
            )[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col] = df_clean[col].fillna(mode_val)
            logger.debug(f"    Filled {col} with mode: {mode_val}")

    # Define feature columns (exclude target and non-predictive columns)
    exclude_cols = [target_col, 'Date', 'Year',
                    'Month', 'Week', 'Branch', 'Segment', 'Rating']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

    logger.info(f"  Total features: {len(feature_cols)}")
    logger.info(f"  Total samples: {len(df_clean)}")

    # Check final missing values
    missing_after = df_clean.isnull().sum().sum()
    logger.info(f"  Missing values after cleaning: {missing_after}")

    if missing_after > 0:
        logger.warning("  Warning: Still have missing values!")
        missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
        logger.warning(f"  Columns with missing values: {missing_cols}")

    # Sort by date
    df_clean_sorted = df_clean.sort_values('Date')

    # Calculate split indices
    total_samples = len(df_clean_sorted)
    # Create splits
    train_data = df_clean_sorted[df_clean_sorted['Date'] < '2023-04-01']
    test_data = df_clean_sorted[df_clean_sorted['Date'] >= '2023-04-01']
    val_data = df_clean_sorted[(
        df_clean_sorted['Date'] >= '2023-11-01') & (df_clean_sorted['Date'] < '2024-04-01')]
    # train_data, val_data, test_data, branch_encoder = fit_label_encoder_on_train(train_data, val_data, test_data)

    # Extract features and targets for each split
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_val = val_data[feature_cols]
    y_val = val_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]

    logger.info(
        f"  Train set: {len(X_train)} samples ({len(X_train)/total_samples*100:.1f}%)")
    logger.info(
        f"  Validation set: {len(X_val)} samples ({len(X_val)/total_samples*100:.1f}%)")
    logger.info(
        f"  Test set: {len(X_test)} samples ({len(X_test)/total_samples*100:.1f}%)")

    # Date ranges for each split
    logger.info(
        f"  Train period: {train_data['Date'].min()} to {train_data['Date'].max()}")
    logger.info(
        f"  Validation period: {val_data['Date'].min()} to {val_data['Date'].max()}")
    logger.info(
        f"  Test period: {test_data['Date'].min()} to {test_data['Date'].max()}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_cols': feature_cols,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        # 'branch_encoder': branch_encoder
    }


# Prophet Training Function
def train_prophet_model(df, target_col='Qty', regressors=None, periods=12):
    """
    Trains Prophet model with optional additional regressors.
    Arguments:
        df: DataFrame with columns 'Date', target, and optional regressors
        target_col: target variable name
        regressors: list of columns to use as external regressors
        periods: forecast horizon for validation/test set
    Returns: Dict of Prophet metrics for splitting period
    """
    df = df.copy()
    prophet_df = df[['Date', target_col]].rename(
        columns={'Date': 'ds', target_col: 'y'})
    model = Prophet()
    if regressors:
        for reg in regressors:
            model.add_regressor(reg)
    if regressors:
        for reg in regressors:
            prophet_df[reg] = df[reg]
    # Train/val/test split: last 'periods' months for val/test, rest train
    split_idx = -periods if periods < len(df) else 0
    train_df = prophet_df.iloc[:split_idx]
    future_df = prophet_df.iloc[split_idx:]
    model.fit(train_df)
    forecast = model.predict(future_df)
    # Evaluate
    y_true = future_df['y'].values
    y_pred = forecast['yhat'].values
    metrics = calculate_metrics(y_true, y_pred, model_name="Prophet")
    return metrics

# Keras RNN/CNN-LSTM helpers


def create_timeseries_supervised(df, feature_cols, target_col='Qty', n_in=12, n_out=1):
    """
    Transform time series dataframe into supervised learning (sliding window) samples.
    """
    X, y = [], []
    values = df[feature_cols + [target_col]].values
    for i in range(len(values) - n_in - n_out + 1):
        X.append(values[i:i+n_in, :-1])  # all features except target
        y.append(values[i+n_in:i+n_in+n_out, -1])  # only target
    return np.array(X), np.array(y)

# DL Model Stubs (implementations filled in after working Prophet):


def build_lstm_model(input_shape):
    model = keras.Sequential([
        layers.LSTM(64, input_shape=input_shape),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def build_gru_model(input_shape):
    model = keras.Sequential([
        layers.GRU(64, input_shape=input_shape),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def build_rnn_model(input_shape):
    model = keras.Sequential([
        layers.SimpleRNN(64, input_shape=input_shape),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def build_cnn_lstm_model(input_shape):
    # Example for 1D conv input to LSTM
    model = keras.Sequential([
        layers.Conv1D(filters=32, kernel_size=3,
                      activation='relu', input_shape=input_shape),
        layers.LSTM(64),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def train_and_evaluate_keras_model(df, feature_cols, target_col, model_builder, n_in=12, epochs=20, batch_size=32):
    X, y = create_timeseries_supervised(
        df, feature_cols, target_col, n_in=n_in)
    n_samples = X.shape[0]
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size +
                     val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    model = model_builder(X_train.shape[1:])
    callbacks = [keras.callbacks.EarlyStopping(
        patience=4, restore_best_weights=True)]
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_val, y_val), callbacks=callbacks, verbose=0)
    y_pred = model.predict(X_test).flatten()
    y_true = y_test.flatten()
    metrics = calculate_metrics(
        y_true, y_pred, model_name=model_builder.__name__)
    return metrics

# (Integration into main loop + completed DL training functions next)


def detailed_monthly_report(lookback_results):
    for lookback, branch_results in lookback_results.items():
        logger.info(f"========== Lookback: {lookback} Year(s) Back ==========")
        for model_type, results in branch_results.items():
            logger.info(f"Model: {model_type.upper()}")
            for branch, metrics in results.items():
                logger.info(f"  Branch: {branch}")
                # If metrics is detailed per month (e.g., add monthly RMSE, MAPE if possible)
                # Here, just logs the default metrics unless models are adapted to output per-month
                if isinstance(metrics, dict):
                    for name, val in metrics.items():
                        logger.info(f"    {name}: {val}")
                else:
                    logger.info(f"    {metrics}")
                logger.info("-")
        logger.info("="*80)


# Example sweep configs (adjust as needed):
LGB_PARAM_GRID = {
    'num_leaves': [31, 50, 100],
    'learning_rate': [0.01, 0.05, 0.1],
    'feature_fraction': [0.8, 0.9, 1.0],
    'bagging_fraction': [0.8, 0.9, 1.0],
    'min_child_samples': [20, 30],
}
XGB_PARAM_GRID = {
    'max_depth': [5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'subsample': [0.7, 0.8, 1.0],
}
CAT_PARAM_GRID = {
    'iterations': [800, 1000],
    'learning_rate': [0.05, 0.1],
    'depth': [5, 6],
}
PROPHET_PARAM_GRID = {
    'seasonality_mode': ['additive', 'multiplicative'],
    'changepoint_prior_scale': [0.05, 0.5, 1.0],
}

# For each ML model, create a sweep wrapper (example for LightGBM):


def sweep_lightgbm(X_train, y_train, X_val, y_val, param_grid):
    best_score = float('inf')
    best_params = None
    tried_configs = []
    keys, values = zip(*param_grid.items())
    for vals in product(*values):
        params = dict(zip(keys, vals))
        model = lgb.LGBMRegressor(
            objective='regression', metric='mae', random_state=42, n_estimators=300, **params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[
                  lgb.early_stopping(30)], verbose=-1)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        tried_configs.append({'params': params, 'rmse': rmse})
        if rmse < best_score:
            best_score = rmse
            best_params = params
    return best_params, best_score, tried_configs
# (Do similarly for XGB, CatBoost, Prophet, Keras swept analogously)

# Prophet sweep:


def sweep_prophet(df, target_col, regressors, param_grid, periods=12):
    best_score = float('inf')
    best_params = None
    tried_configs = []
    keys, values = zip(*param_grid.items())
    for vals in product(*values):
        params = dict(zip(keys, vals))
        prophet_df = df[['Date', target_col]].rename(
            columns={'Date': 'ds', target_col: 'y'})
        model = Prophet(**params)
        for reg in regressors:
            model.add_regressor(reg)
            prophet_df[reg] = df[reg]
        split_idx = -periods if periods < len(df) else 0
        train_df = prophet_df.iloc[:split_idx]
        future_df = prophet_df.iloc[split_idx:]
        model.fit(train_df)
        forecast = model.predict(future_df)
        y_true = future_df['y'].values
        y_pred = forecast['yhat'].values
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        tried_configs.append({'params': params, 'rmse': rmse})
        if rmse < best_score:
            best_score = rmse
            best_params = params
    return best_params, best_score, tried_configs

# Keras tuner sweep (if keras-tuner installed):


def sweep_keras(X_train, y_train, X_val, y_val, build_model_fn, sweep_params):
    if not HAVE_KT:
        logger.warning('keras-tuner not installed')
        return None, None, []
    tuner = kt.RandomSearch(build_model_fn, objective='val_loss',
                            max_trials=8, overwrite=True, directory='kt_sweep')
    tuner.search(X_train, y_train, epochs=15,
                 validation_data=(X_val, y_val), verbose=0)
    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(X_train, y_train, epochs=15,
                   validation_data=(X_val, y_val), verbose=0)
    preds = best_model.predict(X_val).flatten()
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return best_hp.values, rmse, []  # full tried configs can be pulled from tuner

# In your main branch loop for each model, call the sweep wrapper and log best+all results
# ... Example for LightGBM ...
# bp, bs, tried = sweep_lightgbm(ml_data['X_train'], ml_data['y_train'], ml_data['X_val'], ml_data['y_val'], LGB_PARAM_GRID)
# logger.info(f"Best LightGBM params for {branch},{lookback}: {bp}, RMSE: {bs}")
# logger.debug(f"All LightGBM tried configs: {tried}")
# (Prophet, CatBoost, XGB, Keras swept analogously)


detailed_monthly_report(lookback_results)
