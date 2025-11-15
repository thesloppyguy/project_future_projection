# Production Forecasting Script

## Overview

The `production_forecast.py` script provides a comprehensive production forecasting solution that:

1. **Calculates SOB from Historical Full Financial Years**: Uses all complete FYs (April-March) to calculate historical SOB statistics
2. **Forecasts SOB**: Uses time series models (Prophet, Holt-Winters, or Auto-ARIMA) to forecast SOB for the next 12 months
3. **Forecasts Branch Sales**: Uses trained CatBoost models to forecast BLR and MAA sales
4. **Estimates Total Market Sales**: Uses forecasted SOB and weighted mean to estimate total market
5. **Calculates FY Sales**: Aggregates yearly and branch-wise sales by financial year

## Key Features

### 1. Dynamic SOB Calculation (`SOBCalculator`)

- **No Magic Numbers**: All SOB values are calculated from historical data
- **Complete FYs Only**: Only uses complete financial years (12 months) for statistics
- **Historical Statistics**: Calculates mean, std, min, max from all complete FYs
- **SOB Forecasting**: Forecasts SOB for future months using time series models

### 2. Production Forecasting (`ProductionForecaster`)

- **Model Loading**: Loads trained CatBoost models for BLR and MAA
- **Sales Forecasting**: Generates 12-month sales forecasts for each branch
- **Total Market Estimation**: Uses formula: `Total Market = Branch Sales / SOB`
- **Weighted Mean**: Uses SOB-weighted average for combining estimates from multiple branches
- **Variance Bounds**: Provides min/max bounds using historical SOB variance

### 3. Financial Year Sales Calculation

- **Yearly Sales**: Sums total market sales for each complete FY in the forecast
- **Branch-wise Sales**: Calculates BLR and MAA sales for each FY
- **Other Branches**: Estimates sales for other branches (COK, SBD, SBD1) using their historical average SOB
- **Summary Reports**: Creates comprehensive FY summary tables

## Usage

### Basic Usage

```python
from forecast_implementation.production_forecast import ProductionForecaster

# Initialize forecaster
forecaster = ProductionForecaster(
    model_name="catboost",  # Model for sales forecasting
    sob_model_type="prophet"  # Model for SOB forecasting
)

# Run production forecast
results = forecaster.run_production_forecast(
    n_periods=12,
    output_path=Path("forecast_outputs/production/production_forecast_20250101_120000")
)
```

### Command Line

```bash
cd /Users/sahil/Dev/cdro/rnd
python forecast_implementation/production_forecast.py
```

## Output Files

The script generates the following CSV files:

1. **`*_blr_forecast.csv`**: BLR branch sales forecast (monthly)
2. **`*_maa_forecast.csv`**: MAA branch sales forecast (monthly)
3. **`*_blr_sob_forecast.csv`**: BLR SOB forecast (monthly)
4. **`*_maa_sob_forecast.csv`**: MAA SOB forecast (monthly)
5. **`*_total_market.csv`**: Total market sales estimate with bounds (monthly)
6. **`*_fy_summary.csv`**: Financial year summary with yearly and branch-wise totals

## Output Structure

### Monthly Forecasts

Each monthly forecast file contains:

- `date`: Forecast date
- `forecast` or `sob_forecast`: Forecasted value

### Total Market Estimate

Contains:

- `date`: Forecast date
- `total_market_estimate`: Weighted mean estimate
- `total_market_min`: Lower bound
- `total_market_max`: Upper bound
- `blr_forecast`: BLR sales forecast
- `maa_forecast`: MAA sales forecast
- `blr_sob_forecast`: BLR SOB forecast
- `maa_sob_forecast`: MAA SOB forecast
- `total_from_blr`: Total market estimate from BLR
- `total_from_maa`: Total market estimate from MAA

### Financial Year Summary

Contains:

- `FinancialYear`: Financial year (e.g., "2025-2026")
- `TotalMarketSales`: Total market sales for the FY
- `TotalMarketMin`: Lower bound for total market
- `TotalMarketMax`: Upper bound for total market
- `BLR_Sales`: BLR sales for the FY
- `MAA_Sales`: MAA sales for the FY
- `COK_Sales`: COK sales (estimated) for the FY
- `SBD_Sales`: SBD sales (estimated) for the FY
- `SBD1_Sales`: SBD1 sales (estimated) for the FY
- `MonthCount`: Number of months in the FY (should be 12 for complete FY)

## Financial Year Calculation

Financial years are calculated as:

- **FY starts in April**: Month >= 4 means next year's FY
- Example: April 2025 = FY 2025-2026
- Example: March 2025 = FY 2024-2025

## SOB Calculation Methodology

1. **Load Market Share Data**: Reads `data/market_share_by_month_year.csv`
2. **Identify Complete FYs**: Filters to financial years with 12 months of data
3. **Calculate Statistics**: Computes mean, std, min, max from complete FYs
4. **Forecast SOB**: Uses Prophet (or other time series model) to forecast next 12 months
5. **Apply to Forecasts**: Uses forecasted SOB to estimate total market sales

## Total Market Estimation Formula

For each month:

```
Total Market (from BLR) = BLR Sales / BLR SOB
Total Market (from MAA) = MAA Sales / MAA SOB

Weighted Total Market = (Total from BLR × BLR Weight) + (Total from MAA × MAA Weight)

Where:
BLR Weight = BLR SOB / (BLR SOB + MAA SOB)
MAA Weight = MAA SOB / (BLR SOB + MAA SOB)
```

## Branch-wise Sales Calculation

1. **BLR and MAA**: Direct from forecasted sales
2. **Other Branches**: Estimated using formula:
   ```
   Branch Sales = Total Market Sales × Branch Average SOB
   ```
   Where Branch Average SOB is calculated from historical complete FYs

## Example Output

```
Financial Year Summary:
  FinancialYear: 2025-2026
  TotalMarketSales: 1,234,567 units
  TotalMarketMin: 1,100,000 units
  TotalMarketMax: 1,350,000 units
  BLR_Sales: 123,457 units
  MAA_Sales: 456,789 units
  COK_Sales: 82,305 units (estimated)
  SBD_Sales: 333,333 units (estimated)
  SBD1_Sales: 238,683 units (estimated)
```

## Configuration

The script uses configuration from `config.py`:

- `CALIBRATION_END_DATE`: Last date of training data
- `DATA_DIR`: Directory containing market share data
- `MODEL_DIR`: Directory containing trained models

## Dependencies

- pandas
- numpy
- Prophet (for SOB forecasting)
- CatBoost (for sales forecasting)
- Other models from `forecast_implementation.models`

## Notes

- The script automatically handles missing months in market share data
- SOB forecasts are clipped to [0, 1] range
- Sales forecasts are ensured to be non-negative
- Financial year calculations follow Indian FY convention (April-March)
