# Comprehensive Forecasting Implementation

This implementation provides a complete forecasting pipeline following best practices for time series forecasting with proper train/calibration/blind evaluation splits.

## Overview

The pipeline implements the following approach:

1. **Data Preparation**: Loads data, handles missing months (2020-04), and creates monthly/weekly aggregates (combined and branch-wise)
2. **Model Development**: Trains multiple models on training data (2019 to March 2024)
3. **Rolling Forecast Calibration**: Performs rolling forecasts during calibration period (April 2024 to March 2025) with actuals to select best models
4. **Final Training**: Retrains selected models on full data (2019 to March 2025)
5. **Forecast Generation**: Generates 12-month forecasts from March 2025
6. **Blind Evaluation**: Evaluates forecasts on blind data (April 2025 onwards)

## Directory Structure

```
forecast_implementation/
├── __init__.py
├── config.py                 # Configuration settings
├── data_preparation.py       # Data loading and aggregation
├── feature_engineering.py    # Feature creation
├── models.py                 # Model training (LightGBM, XGBoost, CatBoost, Prophet)
├── rolling_forecast.py       # Rolling forecast calibration
├── evaluation.py             # Evaluation metrics and model comparison
├── main.py                   # Main orchestration pipeline
├── run_forecast.py          # Simple runner script
└── README.md                 # This file
```

## Key Features

- **Proper Data Splitting**: Train (2019-2024-03), Calibration (2024-04 to 2025-03), Blind (2025-04+)
- **Rolling Forecast Calibration**: Updates forecasts monthly during calibration period using actuals
- **Multiple Models**: Supports LightGBM, XGBoost, CatBoost, and Prophet
- **Ensemble Forecasting**: Creates weighted ensemble based on calibration performance
- **Anomaly Adaptation**: Models adapt to anomalies during calibration
- **Comprehensive Evaluation**: Multiple metrics (RMSE, MAE, MAPE, NRMSE, R²)
- **Monthly & Weekly Aggregates**: Works with both monthly and weekly data
- **Combined & Branch-wise**: Supports both company-level and branch-level forecasting

## Installation

Ensure you have the required packages:

```bash
pip install pandas numpy lightgbm xgboost catboost prophet scikit-learn
```

## Usage

### Basic Usage

```bash
cd /Users/sahil/Dev/cdro/rnd
python forecast_implementation/run_forecast.py
```

### Programmatic Usage

```python
from forecast_implementation.main import ForecastingPipeline

pipeline = ForecastingPipeline()
pipeline.run_full_pipeline()
```

## Configuration

Edit `config.py` to customize:

- **Date Splits**: Training, calibration, and blind evaluation periods
- **Models**: Which models to train
- **Forecast Horizon**: Number of months to forecast (default: 12)
- **Aggregation Levels**: Monthly/weekly, combined/branch-wise
- **Feature Engineering**: Lag periods, rolling windows, seasonality

## Output

Results are saved to `forecast_outputs/results/<timestamp>/`:

- **calibration/**: Rolling forecast results during calibration period
  - `*_metrics.csv`: Metrics for each rolling window
  - `*_forecasts.csv`: Forecasts vs actuals during calibration
- **forecasts/**: Final 12-month forecasts
  - `*_forecast.csv`: Forecasts for each model
- **evaluation/**: Blind evaluation results
  - `model_comparison.csv`: Detailed metrics for each model
  - `model_summary.csv`: Summary statistics by model
- **config.json**: Configuration used for the run

## Model Selection

The pipeline automatically:

1. Trains all specified models on training data
2. Evaluates each model during calibration period using rolling forecasts
3. Selects best model(s) based on calibration performance
4. Creates ensemble with weights based on calibration RMSE
5. Generates final forecasts using best model/ensemble

## Evaluation Metrics

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **NRMSE**: Normalized RMSE (as percentage)
- **R²**: Coefficient of Determination

## Data Requirements

Input CSV file should have columns:

- `Date`: Date column (will be parsed)
- `Quantity`: Target variable to forecast
- `Branch`: Branch identifier (for branch-wise forecasting)
- Other columns: `Item Code`, `Star Rating`, `Segment`, `Tonnage`, `Region`

## Missing Data Handling

The pipeline automatically handles missing month (2020-04) by:

1. Detecting missing month
2. Interpolating from surrounding months (previous and next)
3. Creating synthetic records for the missing period

## Notes

- The pipeline processes both monthly and weekly aggregates
- For branch-wise forecasting, each branch is processed separately
- Ensemble forecasts use weighted average based on calibration performance
- All models are saved to `forecast_outputs/models/` for future use

## Troubleshooting

1. **Insufficient data**: Ensure at least 12 months of training data
2. **Missing dependencies**: Install required packages (see Installation)
3. **Memory issues**: Process one aggregation level at a time by modifying the code
4. **Model training fails**: Check logs for specific error messages

## Future Enhancements

- Add LSTM/GRU deep learning models
- Implement automatic hyperparameter tuning
- Add more sophisticated anomaly detection
- Support for external regressors (weather, promotions, etc.)
- Real-time forecast updates as new data arrives
