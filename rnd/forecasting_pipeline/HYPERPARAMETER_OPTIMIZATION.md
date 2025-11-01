# Hyperparameter Optimization Guide

## Overview

The pipeline supports hyperparameter optimization using **Optuna** for the following models:

- ✅ **LightGBM** - Full optimization
- ✅ **XGBoost** - Full optimization  
- ✅ **CatBoost** - Full optimization
- ✅ **Prophet** - Limited optimization (fewer trials)
- ⚠️ **LSTM/GRU/RNN** - Not optimized (use fixed architectures for speed)

## How to Enable

### Option 1: Enable in Config

Edit `config.py`:

```python
USE_OPTUNA = True  # Enable hyperparameter optimization
OPTUNA_N_TRIALS = 50  # Number of trials per model
```

### Option 2: Runtime Control

The optimization runs automatically during:
- **Step 4**: Training phase (if `USE_OPTUNA=True`)
- **Step 5**: Validation phase (if `USE_OPTUNA=True`, per fold)

## What Gets Optimized

### LightGBM
- `num_leaves`: 20-100
- `learning_rate`: 0.01-0.3 (log scale)
- `feature_fraction`: 0.6-1.0
- `bagging_fraction`: 0.6-1.0
- `bagging_freq`: 1-10
- `min_child_samples`: 5-100
- `max_depth`: 3-15

### XGBoost
- `max_depth`: 3-10
- `learning_rate`: 0.01-0.3 (log scale)
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0
- `min_child_weight`: 1-10
- `gamma`: 0-5

### CatBoost
- `depth`: 4-10
- `learning_rate`: 0.01-0.3 (log scale)
- `l2_leaf_reg`: 1-10
- `bagging_temperature`: 0-1

### Prophet
- `yearly_seasonality`: True/False
- `weekly_seasonality`: True/False
- `seasonality_mode`: additive/multiplicative
- `changepoint_prior_scale`: 0.001-0.5 (log scale)
- `seasonality_prior_scale`: 0.01-10 (log scale)

## Optimization Strategy

1. **For Training (Step 4)**: 
   - Optimizes on train/val split
   - Uses full `OPTUNA_N_TRIALS` trials

2. **For Validation (Step 5)**:
   - Optimizes per fold (separate optimization for each fold)
   - Uses `min(20, OPTUNA_N_TRIALS)` trials per fold (faster)
   - Ensures each fold has its own optimal hyperparameters

## Performance Impact

- **Time**: Adds significant time (50 trials × 3-4 models = 150-200+ model fits)
- **Accuracy**: Typically improves RMSE by 5-15%
- **Recommendation**: 
  - Use for final production models
  - Skip for quick prototyping/exploration

## Example Usage

```bash
# Enable in config.py first
# USE_OPTUNA = True

# Then run pipeline
python pipeline.py --steps 1 2 3 4 5

# Or just validation with optimization
python step5_multi_model_validation.py
```

## Best Practices

1. **Start without optimization** to get baseline results
2. **Enable optimization** once you've identified promising models
3. **Use fewer trials** (20-30) for faster iterations during development
4. **Use more trials** (50-100) for final production models
5. **Save optimized parameters** for future use (they're logged in the output)

## Output

Optimized hyperparameters are:
- Logged to console during optimization
- Used automatically in subsequent training
- Can be extracted from model metadata files

## Troubleshooting

- **Optuna not found**: Install with `pip install optuna` or `uv add optuna`
- **Too slow**: Reduce `OPTUNA_N_TRIALS` or disable for specific models
- **Memory issues**: Optimize one model at a time by modifying `hyperparameter_optimization.py`

