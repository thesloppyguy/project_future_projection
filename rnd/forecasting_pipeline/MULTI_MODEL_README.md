# Multi-Model Forecasting Pipeline

This pipeline now supports **7 different models** plus an **Ensemble**, with comprehensive validation and a "King of the Hill" ranking system.

## Supported Models

1. **LightGBM** - Gradient boosting framework
2. **XGBoost** - Extreme gradient boosting
3. **CatBoost** - Categorical boosting
4. **Prophet** - Facebook's time series forecasting
5. **LSTM** - Long Short-Term Memory neural network
6. **GRU** - Gated Recurrent Unit neural network
7. **Simple RNN** - Simple Recurrent Neural Network
8. **Ensemble** - Average of all successful models

## How to Run

### Full Pipeline with Multi-Model Validation

```bash
# Run full pipeline (Steps 1-3, then multi-model training & validation)
cd forecasting_pipeline
python pipeline.py --steps 1 2 3 4 5

# Exclude deep learning models (faster, uses only tree-based models + Prophet)
python pipeline.py --steps 1 2 3 4 5 --exclude-deep-learning

# Or run individual steps
python step4_multi_model_training.py  # Train all models
python step5_multi_model_validation.py  # Validate and rank all models
```

### View King of the Hill Results

```bash
python show_king_of_hill.py
```

## Validation Methodology

The validation uses **Time Series Cross-Validation with Rolling Forecast Origin**:

- **Fold 1**: Train 2019-2022, Validate 2023
- **Fold 2**: Train 2019-2023, Validate 2024

Each model is trained and evaluated on both folds, and metrics are averaged.

## Ranking System

Models are ranked using a weighted composite score:
- **50%** - RMSE (Root Mean Squared Error) rank
- **30%** - MAE (Mean Absolute Error) rank  
- **20%** - R² (Coefficient of Determination) rank

**Lower overall rank = Better performance**

## Output Files

After running validation, you'll get:

1. **`multi_model_validation_results.json`** - Detailed metrics for each model per fold
2. **`model_rankings.csv`** - Ranked table with all metrics
3. **`model_comparison.png`** - Visual comparison charts (RMSE, MAE, R², Overall Rank)

## Model-Specific Notes

### Tree-Based Models (LightGBM, XGBoost, CatBoost)
- Use tabular features directly
- Handle categorical features natively
- Fast training and prediction

### Prophet
- Aggregates data by date (univariate time series)
- Automatically detects seasonality
- Good for long-term trends

### Neural Networks (LSTM, GRU, RNN)
- Require sequence data (lookback window = 12 weeks)
- Need scaling (MinMaxScaler)
- More computationally intensive
- Can capture complex temporal patterns

### Ensemble
- Simple average of all successful model predictions
- Often performs better than individual models
- Robust to individual model failures

## Performance Tips

1. **For speed**: Use tree-based models (LightGBM, XGBoost, CatBoost)
2. **For accuracy**: Check the King of the Hill rankings
3. **For robustness**: Use the Ensemble model
4. **For interpretability**: Use Prophet or tree-based models

## Excluding Deep Learning Models

To speed up training/validation, you can exclude neural network models (LSTM, GRU, SimpleRNN):

### Option 1: CLI Argument
```bash
python pipeline.py --exclude-deep-learning
python step5_multi_model_validation.py  # Respects config setting
```

### Option 2: Config Setting
Edit `config.py`:
```python
INCLUDE_DEEP_LEARNING = False  # Excludes LSTM, GRU, SimpleRNN
```

When excluded, only these models run:
- LightGBM
- XGBoost
- CatBoost
- Prophet
- Ensemble (of the above)

## Troubleshooting

- **Neural network errors**: Ensure you have enough data (need at least 12 weeks) or exclude them with `--exclude-deep-learning`
- **Prophet errors**: Works best with aggregated data (already handled)
- **Memory issues**: Neural networks use more memory, exclude them with `--exclude-deep-learning`
- **Slow training**: Exclude deep learning models to speed up significantly

