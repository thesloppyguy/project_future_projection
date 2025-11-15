"""
Evaluation utilities for time series forecasting models.
Calculates RMSE, MAE, MAPE, NRMSE, and R2 score metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys
import os

# Add parent directory to path to import from forecasting_pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from forecasting_pipeline.utils import calculate_metrics


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_nrmse(y_true: np.ndarray, y_pred: np.ndarray, normalization_method: str = "mean") -> float:
    """
    Calculate Normalized Root Mean Square Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        normalization_method: Method for normalization ('mean', 'range', or 'std')
    
    Returns:
        Normalized RMSE value (as percentage)
    """
    from sklearn.metrics import mean_squared_error
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if normalization_method == "mean":
        denominator = np.mean(y_true)
        if denominator == 0 or np.isnan(denominator):
            return np.nan
        nrmse = rmse / denominator
    elif normalization_method == "range":
        data_range = np.max(y_true) - np.min(y_true)
        if data_range == 0 or np.isnan(data_range):
            return np.nan
        nrmse = rmse / data_range
    elif normalization_method == "std":
        denominator = np.std(y_true)
        if denominator == 0 or np.isnan(denominator):
            return np.nan
        nrmse = rmse / denominator
    else:
        raise ValueError(f"Unknown normalization_method: {normalization_method}. Use 'mean', 'range', or 'std'")
    
    return nrmse * 100  # Return as percentage


def evaluate_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    branch: str,
    aggregation: str,
    anomaly_labels: Optional[pd.Series] = None,
    anomaly_severity: Optional[pd.Series] = None
) -> Dict[str, float]:
    """
    Evaluate forecast predictions against actual values.
    
    Args:
        y_true: Actual values (pandas Series with Date index)
        y_pred: Predicted values (pandas Series with Date index)
        model_name: Name of the model
        branch: Branch name
        aggregation: 'weekly' or 'monthly'
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Align indices
    common_index = y_true.index.intersection(y_pred.index)
    
    if len(common_index) == 0:
        print(f"Warning: No common dates between actual and predicted for {model_name} - {branch} - {aggregation}")
        return {
            'model': model_name,
            'branch': branch,
            'aggregation': aggregation,
            'rmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'nrmse': np.nan,
            'r2': np.nan,
            'n_samples': 0
        }
    
    y_true_aligned = y_true.loc[common_index]
    y_pred_aligned = y_pred.loc[common_index]
    
    # Convert to numpy arrays
    y_true_arr = y_true_aligned.values
    y_pred_arr = y_pred_aligned.values
    
    # Remove NaN and Inf values
    valid_mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr) | 
                   np.isinf(y_true_arr) | np.isinf(y_pred_arr))
    
    if valid_mask.sum() == 0:
        print(f"Warning: No valid values for {model_name} - {branch} - {aggregation}")
        return {
            'model': model_name,
            'branch': branch,
            'aggregation': aggregation,
            'rmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'nrmse': np.nan,
            'r2': np.nan,
            'n_samples': 0
        }
    
    y_true_valid = y_true_arr[valid_mask]
    y_pred_valid = y_pred_arr[valid_mask]
    
    # Calculate metrics
    metrics = calculate_metrics(y_true_valid, y_pred_valid)
    
    # Use NRMSE normalized by mean
    nrmse = metrics.get('nrmse_mean', np.nan)
    
    result = {
        'model': model_name,
        'branch': branch,
        'aggregation': aggregation,
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'mape': metrics['mape'],
        'nrmse': nrmse,
        'r2': metrics.get('r2', np.nan),
        'n_samples': len(y_true_valid)
    }
    
    # Add anomaly statistics if available
    if anomaly_labels is not None and len(anomaly_labels) > 0:
        # Align anomaly data with evaluation period
        anomaly_aligned = anomaly_labels.reindex(common_index, fill_value=0)
        anomaly_rate = anomaly_aligned.mean() if len(anomaly_aligned) > 0 else 0.0
        result['anomaly_rate'] = anomaly_rate
        
        if anomaly_severity is not None and len(anomaly_severity) > 0:
            severity_aligned = anomaly_severity.reindex(common_index, fill_value=0.0)
            anomaly_severities = severity_aligned[anomaly_aligned == 1]
            result['mean_anomaly_severity'] = anomaly_severities.mean() if len(anomaly_severities) > 0 else 0.0
            result['max_anomaly_severity'] = anomaly_severities.max() if len(anomaly_severities) > 0 else 0.0
        else:
            result['mean_anomaly_severity'] = np.nan
            result['max_anomaly_severity'] = np.nan
    else:
        result['anomaly_rate'] = np.nan
        result['mean_anomaly_severity'] = np.nan
        result['max_anomaly_severity'] = np.nan
    
    return result


def aggregate_forecast_to_match(
    forecast: pd.Series,
    target_freq: str,
    aggregation_method: str = 'sum'
) -> pd.Series:
    """
    Aggregate forecast to match target frequency.
    
    Args:
        forecast: Forecast series
        target_freq: Target frequency ('W-MON' for weekly, 'MS' for monthly)
        aggregation_method: 'sum' or 'mean'
        
    Returns:
        Aggregated series
    """
    if aggregation_method == 'sum':
        return forecast.resample(target_freq).sum()
    elif aggregation_method == 'mean':
        return forecast.resample(target_freq).mean()
    else:
        raise ValueError(f"Unknown aggregation_method: {aggregation_method}")


def save_evaluation_results(
    results: List[Dict[str, float]],
    output_path: str
) -> None:
    """
    Save evaluation results to CSV.
    
    Args:
        results: List of evaluation dictionaries
        output_path: Path to save CSV file
    """
    df = pd.DataFrame(results)
    
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Saved evaluation results to {output_path}")


def generate_summary_report(
    all_results: List[Dict[str, float]],
    output_path: str
) -> pd.DataFrame:
    """
    Generate summary report comparing all models.
    
    Args:
        all_results: List of all evaluation dictionaries
        output_path: Path to save summary CSV
        
    Returns:
        Summary dataframe
    """
    df = pd.DataFrame(all_results)
    
    # Create summary by model and aggregation
    summary = df.groupby(['model', 'aggregation']).agg({
        'rmse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'mape': ['mean', 'std'],
        'nrmse': ['mean', 'std'],
        'r2': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in summary.columns]
    
    # Save summary
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved summary report to {output_path}")
    
    return summary


def save_anomaly_detection_results(
    anomaly_labels: pd.Series,
    anomaly_severity: pd.Series,
    series: pd.Series,
    output_path: str,
    branch: str,
    aggregation: str
) -> None:
    """
    Save anomaly detection results to CSV.
    
    Args:
        anomaly_labels: Anomaly labels (0=normal, 1=anomaly)
        anomaly_severity: Anomaly severity scores (0-1)
        series: Original time series
        output_path: Path to save CSV file
        branch: Branch name
        aggregation: 'weekly' or 'monthly'
    """
    # Align all series to common index
    common_index = series.index.intersection(anomaly_labels.index).intersection(anomaly_severity.index)
    
    df = pd.DataFrame({
        'Date': common_index,
        'Value': series.reindex(common_index).values,
        'Anomaly_Label': anomaly_labels.reindex(common_index, fill_value=0).values,
        'Anomaly_Severity': anomaly_severity.reindex(common_index, fill_value=0.0).values,
        'Branch': branch,
        'Aggregation': aggregation
    })
    
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Saved anomaly detection results to {output_path}")


def generate_anomaly_summary_report(
    all_anomaly_results: List[Dict],
    output_path: str
) -> pd.DataFrame:
    """
    Generate summary report of anomaly detection across all branches and models.
    
    Args:
        all_anomaly_results: List of dictionaries with anomaly statistics
        output_path: Path to save summary CSV
        
    Returns:
        Summary dataframe
    """
    df = pd.DataFrame(all_anomaly_results)
    
    if len(df) == 0:
        print("Warning: No anomaly results to summarize")
        return pd.DataFrame()
    
    # Create summary by branch and aggregation
    summary = df.groupby(['branch', 'aggregation']).agg({
        'anomaly_rate': ['mean', 'std', 'min', 'max'],
        'mean_anomaly_severity': ['mean', 'std'],
        'max_anomaly_severity': ['mean', 'max']
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in summary.columns]
    
    # Save summary
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved anomaly summary report to {output_path}")
    
    return summary

