# HVAC Forecasting ML Pipeline

A modular Python pipeline for forecasting HVAC quantity 1 year in advance using LightGBM. This pipeline implements a 6-step methodology designed to handle high volatility and categorical features.

## Overview

The pipeline processes raw transactional data and generates 52-week ahead forecasts for all combinations of Branch and Tonnage using a Gradient Boosted Tree model (LightGBM).

## Pipeline Steps

1. **Data Aggregation**: Aggregates raw data to weekly frequency and creates complete panel dataset
2. **Feature Engineering**: Creates date-based, lag, rolling window, and categorical features
3. **Outlier Detection**: Identifies and caps outliers using Isolation Forest/LOF with Winsorization
4. **Model Training**: Trains LightGBM model with optional hyperparameter optimization
5. **Validation**: Performs time series cross-validation and trains final model
6. **Forecast Generation**: Generates 52-week ahead forecasts using recursive autoregressive approach

## Installation

All required dependencies are already in `pyproject.toml`:
- pandas, numpy
- lightgbm
- scikit-learn
- optuna (optional, for hyperparameter optimization)
- matplotlib, seaborn (for visualization)

## Usage

### Running the Full Pipeline

```bash
# From the project root directory
cd forecasting_pipeline
python pipeline.py
```

### Running Specific Steps

```bash
# Run only steps 1-3
python pipeline.py --steps 1 2 3

# Run only forecast generation (requires previous steps)
python pipeline.py --steps 6

# Skip validation step
python pipeline.py --skip 5
```

### Running Individual Steps

Each step can be run independently:

```bash
python step1_data_aggregation.py
python step2_feature_engineering.py
python step3_outlier_detection.py
python step4_model_training.py
python step5_validation.py
python step6_forecast_generation.py
```

### Using as a Module

```python
from forecasting_pipeline.pipeline import run_full_pipeline

# Run full pipeline
results = run_full_pipeline()

# Skip certain steps
results = run_full_pipeline(skip_steps=[4])
```

## Configuration

Edit `config.py` to customize:

- **Data paths**: Source data file and output directories
- **Aggregation**: Weekly frequency settings
- **Features**: Peak season months, lag periods, rolling windows
- **Outlier detection**: Method (isolation_forest, lof, percentile) and parameters
- **Model**: LightGBM parameters and Optuna optimization settings
- **Validation**: Cross-validation folds and date ranges
- **Forecasting**: Forecast horizon and start date

## Output Files

The pipeline generates the following outputs in `forecasting_pipeline/outputs/`:

- `aggregated_weekly_data.parquet`: Weekly aggregated data
- `featured_data.parquet`: Data with engineered features
- `cleaned_data.parquet`: Data after outlier treatment
- `feature_list.txt`: List of feature columns
- `categorical_mappings.pkl`: Categorical encoding mappings
- `validation_results.json`: Validation metrics
- `validation_summary.csv`: Summary of validation results
- `validation_fold_*.png`: Validation plots
- `feature_importance.png`: Feature importance plot
- `forecast_52_weeks.csv`: Final 52-week forecasts

Models are saved in `forecasting_pipeline/models/`:
- `lightgbm_model_*.pkl`: Trained models with timestamps
- `latest_final_model.pkl`: Symlink to latest final model
- `model_metadata_*.json`: Model metadata

## Key Features

### Panel Data Completeness
The pipeline creates a complete panel by filling missing combinations with zeros, which is critical for the model to learn when demand is zero.

### Robust Feature Engineering
- **Date features**: Week of year, month, year, quarter, peak season indicator
- **Lag features**: 1 week, 4 weeks, and 52 weeks (captures seasonality)
- **Rolling features**: Mean, std, max over 4-week windows
- **Categorical encoding**: Label encoding for Branch and Tonnage

### Outlier Handling
Uses Winsorization (capping at 99th percentile) instead of deletion to preserve high-demand signals while preventing model skewing.

### Time Series Validation
Uses rolling forecast origin cross-validation to ensure proper time series validation (no data leakage).

### Recursive Forecasting
Generates multi-step ahead forecasts by iteratively using previous predictions to create lag/rolling features for future weeks.

## Logging

The pipeline logs to both console and `forecasting_pipeline/outputs/pipeline.log`. Set log level with:

```bash
python pipeline.py --log-level DEBUG
```

## Troubleshooting

1. **Import errors**: Ensure you're running from the correct directory or using the package import structure
2. **Missing files**: Run steps in order, or check that required intermediate files exist
3. **Memory issues**: The panel dataset can be large; consider filtering data in config if needed
4. **Model not found**: Ensure Step 5 (validation) has been run before Step 6 (forecast generation)

## Design Decisions

1. **Weekly aggregation**: Best balance for handling volatility while maintaining enough data points
2. **LightGBM**: Excels at learning complex, non-linear patterns and handles categorical features natively
3. **Lag 52 weeks**: Critical feature for capturing annual seasonality patterns
4. **Winsorization over deletion**: Preserves outlier information while preventing skewing
5. **Panel completeness**: Zero-filling is essential for intermittent demand patterns

