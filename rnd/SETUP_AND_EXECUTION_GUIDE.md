# AC Sales & Weather Analysis - Setup and Execution Guide

## Overview

This project implements a comprehensive AC sales forecasting system using weather data across multiple scopes:
- **Combined Scope**: National-level forecasting
- **Regional Scope**: State-level forecasting with two approaches
- **Segment Scope**: Product segment-level forecasting

## Current Status

✅ **Completed:**
- All Python scripts created (`regional_models_analysis.py`, `segment_models_analysis.py`, `forecasting_and_outputs.py`)
- Main runner script (`run_analysis.py`) with dependency checking
- Test script (`test_scripts.py`) for validation
- UV environment setup with dependencies
- Graceful handling of pmdarima compatibility issues

⚠️ **Known Issues:**
- pmdarima has numpy compatibility issues (scripts handle this gracefully)
- scikit-learn import issue in dependency check
- Need to run actual analysis phases

## File Structure

```
rnd/
├── data/
│   ├── monthly_sales_summary.csv          # Sales data
│   └── Final Sales.xlsx                   # Original sales data
├── outputs/
│   ├── processed_weather_data/            # Processed weather files
│   │   ├── AP_weather_timeseries.csv
│   │   ├── KA_weather_timeseries.csv
│   │   ├── KL_weather_timeseries.csv
│   │   ├── TL_weather_timeseries.csv
│   │   └── TN_weather_timeseries.csv
│   ├── models/                            # Trained models (to be created)
│   ├── charts/                            # Visualizations (to be created)
│   └── forecasts/                         # Forecast outputs (to be created)
├── *.ipynb                                # Jupyter notebooks (Phases 1-4)
├── *.py                                   # Python scripts (Phases 5-7)
├── pyproject.toml                         # UV project configuration
└── SETUP_AND_EXECUTION_GUIDE.md          # This file
```

## Dependencies

### Required Packages (via UV)
- pandas, numpy, matplotlib, seaborn
- statsmodels, prophet, tensorflow
- scikit-learn, joblib
- jupyter, nbconvert (for notebook execution)

### Optional Packages
- pmdarima (has compatibility issues, scripts use alternatives)

## Execution Options

### Option 1: Run All Phases with UV
```bash
cd /Users/sahil/Dev/cdro/rnd

# Check dependencies
uv run python run_analysis.py --check-deps

# Run all phases (1-7)
uv run python run_analysis.py

# Run specific phases
uv run python run_analysis.py --phases 5,6,7

# Skip notebook phases (1-4) and run only Python scripts
uv run python run_analysis.py --phases 5,6,7 --skip-notebook
```

### Option 2: Run Individual Scripts
```bash
cd /Users/sahil/Dev/cdro/rnd

# Phase 5: Regional Models Analysis
uv run python regional_models_analysis.py

# Phase 6: Segment Models Analysis  
uv run python segment_models_analysis.py

# Phase 7: Forecasting and Outputs
uv run python forecasting_and_outputs.py
```

### Option 3: Run Notebooks First, Then Scripts
```bash
cd /Users/sahil/Dev/cdro/rnd

# Run notebooks for Phases 1-4 (if needed)
jupyter nbconvert --to script --execute weather_sales_analysis.ipynb

# Then run Python scripts for Phases 5-7
uv run python run_analysis.py --phases 5,6,7 --skip-notebook
```

## Expected Outputs

### Forecast Files
- `outputs/forecasts/forecasts_combined_models.csv` - National forecasts
- `outputs/forecasts/forecasts_regional_models.csv` - Regional forecasts
- `outputs/forecasts/forecasts_segments.csv` - Segment forecasts

### Analysis Files
- `outputs/regional_models_results.csv` - Regional model performance
- `outputs/segment_models_results.csv` - Segment model performance
- `outputs/segment_aggregation_validation.csv` - Aggregation validation

### Visualization Files
- `outputs/charts/regional_models_comparison.png`
- `outputs/charts/segment_models_comparison.png`
- Additional EDA and decomposition charts

### Model Files
- `outputs/models/regional_*_approach_A.*` - 30 Approach A models
- `outputs/models/regional_*_approach_B.*` - 30 Approach B models
- `outputs/models/segment_*` - Segment models

### Documentation
- `outputs/results_summary.md` - Comprehensive analysis report

## Troubleshooting

### Issue 1: pmdarima Compatibility
**Problem:** `ValueError: numpy.dtype size changed, may indicate binary incompatibility`

**Solution:** Scripts handle this gracefully by:
- Using default ARIMA parameters when pmdarima fails
- Continuing execution with alternative implementations
- No action needed - scripts will work

### Issue 2: scikit-learn Import Error
**Problem:** scikit-learn import fails in dependency check

**Solution:** 
```bash
# Reinstall scikit-learn
uv add --upgrade scikit-learn

# Or rebuild environment
rm -rf .venv && uv sync
```

### Issue 3: Missing Data Files
**Problem:** Weather or sales data files not found

**Solution:** Ensure these files exist:
- `data/monthly_sales_summary.csv`
- `outputs/processed_weather_data/*_weather_timeseries.csv`

### Issue 4: Memory Issues
**Problem:** Out of memory during model training

**Solution:** 
- Reduce batch sizes in neural network training
- Process fewer models at once
- Use smaller lookback windows

## Model Details

### Regional Models (Phase 5)
- **60 models total**: 5 states × 6 models × 2 approaches
- **Approach A**: Own state's weather data only
- **Approach B**: All states' weather data as features
- **Models**: ARIMA, SARIMAX, Seasonal Decomp, Holt-Winters, LSTM, GRU, Prophet

### Segment Models (Phase 6)
- **30-45 models**: Top 10-15 segments × 3 models
- **Models**: ARIMA, SARIMAX, Prophet
- **Segments**: Top State-Star_Rating-Tonnage combinations (70-80% of sales)

### Forecasting (Phase 7)
- **12-month forecasts**: 2025-07 to 2026-06
- **Weather strategy**: Reuse 2025 weather pattern
- **Outputs**: 3 main forecast CSV files + comprehensive summary

## Performance Expectations

### Training Time
- **Regional models**: ~30-60 minutes (60 models)
- **Segment models**: ~15-30 minutes (30-45 models)
- **Forecasting**: ~5-10 minutes (generation only)

### Memory Usage
- **Peak usage**: ~2-4 GB during neural network training
- **Typical usage**: ~1-2 GB for most operations

## Validation and Quality Checks

### Data Validation
- Check date ranges and completeness
- Verify weather-sales data alignment
- Validate segment identification

### Model Validation
- Cross-validation on 2025-01 to 2025-06 period
- MAE, RMSE, MAPE metrics
- Segment aggregation validation

### Output Validation
- Forecast file completeness
- Model file persistence
- Visualization generation

## Next Steps

1. **Fix scikit-learn import issue** (if present)
2. **Run dependency check**: `uv run python run_analysis.py --check-deps`
3. **Execute analysis**: `uv run python run_analysis.py --phases 5,6,7`
4. **Review outputs** in `outputs/` directory
5. **Generate final report** from `outputs/results_summary.md`

## Support

If you encounter issues:
1. Check this guide for troubleshooting steps
2. Review error messages in terminal output
3. Verify data files are present and accessible
4. Check UV environment is properly set up

## Success Criteria

✅ **Analysis Complete When:**
- All 3 forecast CSV files are generated
- Model performance results are saved
- Visualizations are created
- Results summary is generated
- No critical errors in execution

The analysis is designed to be robust and handle common issues gracefully. Most problems are related to environment setup rather than the analysis logic itself.
