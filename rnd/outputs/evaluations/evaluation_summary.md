# Comprehensive Sales Forecasting Evaluation Summary

Generated on: 2025-10-22 03:22:22

## Overview
- **Total Combinations Evaluated**: 65
- **Successful Evaluations**: 46
- **Trained (No Test Data)**: 0
- **Errors**: 19
- **Success Rate**: 70.8%

## Performance Summary (Successful Evaluations Only)

### Model Performance Metrics
- **Average MAE**: 0.00
- **Average RMSE**: 0.00
- **Average MAPE**: 0.00%
- **Best MAE**: 0.00
- **Worst MAE**: 0.00

### Top 10 Best Performing Combinations (by MAE)
| State | Star Rating | Tonnage | MAE | RMSE | MAPE |
|-------|-------------|---------|-----|------|------|
| Karnataka | 2 Star | 1.8 | 0.00 | 0.00 | 0.00% |
| Karnataka | 5 Star | 1.8 | 0.00 | 0.00 | 0.00% |
| Kerala | 5 Star | 1.8 | 0.00 | 0.00 | 0.00% |
| Andhra Pradesh | 1 Star | 1.8 | 0.00 | 0.00 | 0.00% |
| Andhra Pradesh | 5 Star | 1.8 | 0.00 | 0.00 | 0.00% |
| Kerala | 3 Star | 0.8 | 0.00 | 0.00 | 0.00% |
| Andhra Pradesh | 3 Star | 0.8 | 0.00 | 0.00 | 0.00% |
| Kerala | 1 Star | 1.8 | 0.00 | 0.00 | 0.00% |
| Andhra Pradesh | 3 Star | 1.0 | 0.00 | 0.00 | 0.00% |
| Karnataka | 3 Star | 1.8 | 0.00 | 0.00 | 0.00% |

### Performance by State
- **Andhra Pradesh**: MAE 0.00 ± 0.00 (n=9)
- **Karnataka**: MAE 0.00 ± 0.00 (n=10)
- **Kerala**: MAE 0.00 ± 0.00 (n=9)
- **Tamil Nadu**: MAE 0.00 ± 0.00 (n=9)
- **Telangana**: MAE 0.00 ± 0.00 (n=9)

### Performance by Star Rating
- **1 Star**: MAE 0.00 ± 0.00 (n=5)
- **2 Star**: MAE 0.00 ± nan (n=1)
- **3 Star**: MAE 0.00 ± 0.00 (n=25)
- **5 Star**: MAE 0.00 ± 0.00 (n=15)

## Error Analysis
- **Exception during evaluation: too many values to unpack (expected 3)**: 10 combinations
- **Insufficient data: 9 months**: 3 combinations
- **Insufficient data: 10 months**: 2 combinations
- **Insufficient data: 11 months**: 1 combinations
- **Insufficient data: 7 months**: 1 combinations
- **Insufficient data: 8 months**: 1 combinations
- **Insufficient data: 4 months**: 1 combinations

## Files Generated
- `outputs/evaluations/comprehensive_evaluation_results.csv` - Main results
- `outputs/evaluations/detailed_predictions.csv` - Detailed predictions
- `outputs/evaluations/evaluation_summary.md` - This summary

## Recommendations
1. **Focus on High-Performing Combinations**: Prioritize the top 10 combinations for business planning
2. **Address Data Quality Issues**: Review combinations with errors for data quality improvements
3. **Model Selection**: Consider ensemble methods for combinations with high MAPE
4. **Weather Integration**: Analyze weather correlation for better feature engineering

---
*This evaluation provides comprehensive insights into sales forecasting performance across all product combinations.*
