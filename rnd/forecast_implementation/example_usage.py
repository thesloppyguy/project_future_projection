"""
Example usage of the forecasting pipeline.
This script demonstrates how to use individual components.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from forecast_implementation.config import *
from forecast_implementation.data_preparation import DataPreparator
from forecast_implementation.feature_engineering import FeatureEngineer
from forecast_implementation.models import ModelTrainer
from forecast_implementation.rolling_forecast import RollingForecastCalibrator
from forecast_implementation.evaluation import Evaluator

def example_data_preparation():
    """Example: Data preparation."""
    print("\n=== Example: Data Preparation ===")
    
    preparator = DataPreparator(DATA_FILE, MISSING_MONTH)
    
    # Load data
    raw_data = preparator.load_data()
    print(f"Loaded {len(raw_data):,} rows")
    
    # Prepare aggregated data
    aggregated = preparator.prepare_aggregated_data()
    print(f"Monthly combined: {len(aggregated['monthly']['combined'])} rows")
    print(f"Monthly branch-wise: {len(aggregated['monthly']['branch_wise'])} rows")
    
    return aggregated

def example_feature_engineering():
    """Example: Feature engineering."""
    print("\n=== Example: Feature Engineering ===")
    
    # Create sample series
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2020-01-01', periods=24, freq='MS')
    values = np.random.randn(24).cumsum() + 100
    series = pd.Series(values, index=dates)
    
    # Create features
    engineer = FeatureEngineer()
    features_df = engineer.create_features(series)
    print(f"Created {len(features_df.columns)} features")
    print(f"Features: {list(features_df.columns)[:10]}...")
    
    # Prepare ML features
    X, y, feature_names = engineer.prepare_ml_features(series)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")

def example_model_training():
    """Example: Model training."""
    print("\n=== Example: Model Training ===")
    
    # Create sample data
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2020-01-01', periods=36, freq='MS')
    values = np.random.randn(36).cumsum() + 100
    series = pd.Series(values, index=dates)
    
    # Prepare features
    engineer = FeatureEngineer()
    X, y, feature_names = engineer.prepare_ml_features(series)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Train model
    trainer = ModelTrainer()
    
    if LIGHTGBM_AVAILABLE:
        print("Training LightGBM...")
        model = trainer.train_lightgbm(X_train, y_train, X_val, y_val)
        if model:
            print("LightGBM trained successfully")
            # Generate forecast
            forecast = trainer.forecast_lightgbm(model, X_val[:5])
            print(f"Sample forecast: {forecast[:3]}")

def example_rolling_forecast():
    """Example: Rolling forecast calibration."""
    print("\n=== Example: Rolling Forecast Calibration ===")
    
    # Create sample data
    import pandas as pd
    import numpy as np
    
    # Training data
    train_dates = pd.date_range('2020-01-01', periods=24, freq='MS')
    train_values = np.random.randn(24).cumsum() + 100
    train_series = pd.Series(train_values, index=train_dates)
    
    # Calibration data
    cal_dates = pd.date_range('2022-01-01', periods=12, freq='MS')
    cal_values = np.random.randn(12).cumsum() + 120
    cal_series = pd.Series(cal_values, index=cal_dates)
    
    # Setup
    trainer = ModelTrainer()
    engineer = FeatureEngineer()
    calibrator = RollingForecastCalibrator(trainer, engineer, rolling_window_months=1)
    
    print("Running rolling forecast calibration...")
    results = calibrator.rolling_forecast_calibration(
        train_series, cal_series, 'lightgbm', forecast_horizon=6
    )
    
    if len(results['metrics']) > 0:
        print(f"Generated {len(results['metrics'])} rolling forecasts")
        avg_rmse = np.mean([m['rmse'] for m in results['metrics'] if not np.isnan(m['rmse'])])
        print(f"Average RMSE: {avg_rmse:.2f}")

def example_evaluation():
    """Example: Evaluation."""
    print("\n=== Example: Evaluation ===")
    
    import pandas as pd
    import numpy as np
    
    # Create sample forecast and actuals
    dates = pd.date_range('2025-04-01', periods=6, freq='MS')
    forecast = pd.Series(np.random.randn(6) * 10 + 100, index=dates)
    actuals = pd.Series(np.random.randn(6) * 10 + 105, index=dates)
    
    # Evaluate
    evaluator = Evaluator()
    metrics = evaluator.evaluate_forecast(forecast, actuals, 'test_model')
    
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"R²: {metrics['r2']:.4f}")

def main():
    """Run all examples."""
    print("=" * 80)
    print("Forecasting Pipeline - Example Usage")
    print("=" * 80)
    
    try:
        example_data_preparation()
    except Exception as e:
        print(f"Error in data preparation example: {e}")
    
    try:
        example_feature_engineering()
    except Exception as e:
        print(f"Error in feature engineering example: {e}")
    
    try:
        example_model_training()
    except Exception as e:
        print(f"Error in model training example: {e}")
    
    try:
        example_rolling_forecast()
    except Exception as e:
        print(f"Error in rolling forecast example: {e}")
    
    try:
        example_evaluation()
    except Exception as e:
        print(f"Error in evaluation example: {e}")
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()

