# Normalized Root Mean Square Error (NRMSE)

## Overview

NRMSE is now calculated alongside RMSE in the validation pipeline. It provides a **scale-independent** error metric, making it easier to compare model performance across different datasets or contexts.

## Three Normalization Methods

The pipeline calculates **three variants** of NRMSE:

### 1. NRMSE (mean) - **Recommended**
```
NRMSE_mean = (RMSE / mean(y_true)) × 100%
```

**Interpretation**: Error as a percentage of the mean value
- **Example**: NRMSE = 187% means the RMSE is 1.87× the average quantity
- **Best for**: Understanding error relative to typical values
- **Use case**: When you want to know "how wrong are predictions relative to average demand?"

### 2. NRMSE (range)
```
NRMSE_range = (RMSE / (max(y_true) - min(y_true))) × 100%
```

**Interpretation**: Error as a percentage of the data range
- **Example**: NRMSE = 5% means the RMSE is 5% of the full data range
- **Best for**: Understanding error relative to data spread
- **Use case**: When you care about errors relative to the full scale of possible values

### 3. NRMSE (std)
```
NRMSE_std = (RMSE / std(y_true)) × 100%
```

**Interpretation**: Error in terms of standard deviations
- **Example**: NRMSE = 200% means the RMSE is 2× the standard deviation
- **Best for**: Statistical interpretation (coefficient of variation perspective)
- **Use case**: When comparing with statistical measures of variability

## Example Calculation

For validation data with:
- Actual values: Mean = 99.27, Std = 292.93, Range = 0 to 2023.20
- RMSE = 186

The three NRMSE values would be:
- **NRMSE_mean**: (186 / 99.27) × 100% ≈ **187%**
- **NRMSE_range**: (186 / 2023.20) × 100% ≈ **9.2%**
- **NRMSE_std**: (186 / 292.93) × 100% ≈ **63.5%**

## Interpretation Guidelines

### NRMSE_mean < 30%
- Excellent performance
- Predictions are very accurate relative to average values

### NRMSE_mean 30% - 100%
- Good performance
- Predictions are within 1× the average value

### NRMSE_mean 100% - 200%
- Moderate performance
- RMSE is 1-2× the average value
- This is typical for many forecasting problems

### NRMSE_mean > 200%
- Poor performance
- RMSE is more than 2× the average value
- Consider feature engineering or different models

## Why Use NRMSE?

1. **Scale Independence**: Compare models across different scales or datasets
2. **Intuitive Interpretation**: Percentage is easier to understand than absolute units
3. **Business Context**: Understand error relative to typical values (NRMSE_mean)
4. **Model Comparison**: Easier to compare when data scales differ

## Where It's Displayed

- **Validation logs**: Shown alongside RMSE, MAE, R²
- **Model rankings CSV**: All three variants included
- **King of the Hill**: NRMSE_mean shown in summary

## Which One to Use?

**For most cases, use NRMSE_mean** because:
- It's most intuitive (error relative to average)
- Most commonly used in literature
- Easiest to explain to stakeholders

Use the others for specific analysis:
- **NRMSE_range**: When you care about errors on the full scale
- **NRMSE_std**: For statistical analysis or when comparing with CV (coefficient of variation)

