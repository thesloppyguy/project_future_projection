"""
Utility functions for the forecasting pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from . import config
except ImportError:
    import config


def ensure_quantity_non_negative(df: pd.DataFrame, quantity_col: str = "Quantity") -> pd.DataFrame:
    """Ensure quantity values are non-negative."""
    df = df.copy()
    df[quantity_col] = df[quantity_col].abs()
    return df


def create_complete_panel(
    df: pd.DataFrame,
    date_col: str,
    group_cols: List[str],
    value_col: str,
    freq: str = "W-MON"
) -> pd.DataFrame:
    """
    Create a complete panel dataset by filling missing combinations with zeros.
    
    Args:
        df: Input dataframe (should already be aggregated)
        date_col: Name of the date column
        group_cols: List of columns to group by (including date)
        value_col: Name of the value column to sum
        freq: Frequency for date resampling
        
    Returns:
        Complete panel dataframe with all combinations filled
    """
    # Ensure date is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Get unique values for each group column (excluding date)
    non_date_cols = [col for col in group_cols if col != date_col]
    unique_values = {}
    for col in non_date_cols:
        unique_values[col] = sorted(df[col].dropna().unique())
    
    # Use actual dates from the data (already resampled to weekly)
    # This ensures exact matching during merge
    unique_dates_list = sorted(df[date_col].unique().tolist())
    
    # Also create a complete date range for any missing weeks
    if len(unique_dates_list) > 0:
        min_date = unique_dates_list[0]
        max_date = unique_dates_list[-1]
        # Normalize dates to date only (remove time component if any)
        min_date = pd.to_datetime(min_date).normalize()
        max_date = pd.to_datetime(max_date).normalize()
        complete_date_range = pd.date_range(start=min_date, end=max_date, freq=freq)
        # Normalize all dates to ensure matching
        complete_date_range = [d.normalize() for d in complete_date_range]
        unique_dates_normalized = [pd.to_datetime(d).normalize() for d in unique_dates_list]
        # Combine actual dates with complete range, removing duplicates
        date_range = sorted(set(unique_dates_normalized + complete_date_range))
    else:
        date_range = []
    
    # Normalize dates in input dataframe for proper matching
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    
    # Create complete combinations of non-date columns
    import itertools
    group_combinations = list(itertools.product(*[unique_values[col] for col in non_date_cols]))
    
    # Create complete panel
    complete_panel = []
    for date in date_range:
        for combo in group_combinations:
            row = {date_col: date}
            for i, col in enumerate(non_date_cols):
                row[col] = combo[i]
            complete_panel.append(row)
    
    complete_df = pd.DataFrame(complete_panel)
    # Ensure date column is datetime
    complete_df[date_col] = pd.to_datetime(complete_df[date_col]).dt.normalize()
    
    # Prepare aggregated data (already aggregated, just ensure value_col exists)
    if value_col not in df.columns:
        # If value_col doesn't exist, create it by grouping
        agg_df = df.groupby(group_cols)[value_col].sum().reset_index()
    else:
        # Data is already aggregated, just select the columns we need
        agg_df = df[group_cols + [value_col]].copy()
        # In case there are duplicates, group by again
        agg_df = agg_df.groupby(group_cols)[value_col].sum().reset_index()
    
    # Ensure date column is normalized in agg_df
    agg_df[date_col] = pd.to_datetime(agg_df[date_col]).dt.normalize()
    
    # Merge with actual data - match on all group columns including date
    merged = complete_df.merge(
        agg_df,
        on=group_cols,
        how="left"
    )
    
    # Fill missing values with 0
    merged[value_col] = merged[value_col].fillna(0)
    
    return merged.sort_values(group_cols).reset_index(drop=True)


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
        normalization_method: Method for normalization
            - "mean": Normalize by mean of y_true (NRMSE = RMSE / mean)
            - "range": Normalize by range (max - min) of y_true
            - "std": Normalize by standard deviation of y_true
    
    Returns:
        Normalized RMSE value (as percentage)
    """
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
    
    return nrmse * 100  # Return as percentage for easier interpretation


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all evaluation metrics including NRMSE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "mape": calculate_mape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "nrmse_mean": calculate_nrmse(y_true, y_pred, "mean"),  # Normalized by mean
        "nrmse_range": calculate_nrmse(y_true, y_pred, "range"),  # Normalized by range
        "nrmse_std": calculate_nrmse(y_true, y_pred, "std"),  # Normalized by std
    }
    return metrics


def create_lag_features(
    df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
    lag_periods: List[int],
    date_col: str = "Date"
) -> pd.DataFrame:
    """Create lag features for time series data."""
    df = df.copy()
    df = df.sort_values(group_cols + [date_col])
    
    for lag in lag_periods:
        df[f"lag_{lag}_weeks"] = df.groupby(group_cols)[value_col].shift(lag)
    
    return df


def create_rolling_features(
    df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
    window: int,
    stats: List[str],
    date_col: str = "Date"
) -> pd.DataFrame:
    """Create rolling window features."""
    df = df.copy()
    df = df.sort_values(group_cols + [date_col])
    
    for stat in stats:
        if stat == "mean":
            df[f"rolling_{stat}_{window}_weeks"] = df.groupby(group_cols)[value_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
        elif stat == "std":
            df[f"rolling_{stat}_{window}_weeks"] = df.groupby(group_cols)[value_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
            )
        elif stat == "max":
            df[f"rolling_{stat}_{window}_weeks"] = df.groupby(group_cols)[value_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
            )
    
    # Fill NaN values
    for stat in stats:
        col = f"rolling_{stat}_{window}_weeks"
        if stat == "std":
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(df[value_col])
    
    return df


def create_date_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Create date-based features."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    df["week_of_year"] = df[date_col].dt.isocalendar().week
    df["month"] = df[date_col].dt.month
    df["year"] = df[date_col].dt.year
    df["quarter"] = df[date_col].dt.quarter
    df["day_of_week"] = df[date_col].dt.dayofweek
    
    return df


def create_peak_season_feature(
    df: pd.DataFrame,
    peak_months: List[int],
    month_col: str = "month"
) -> pd.DataFrame:
    """Create binary peak season feature."""
    df = df.copy()
    df["is_peak_season"] = df[month_col].isin(peak_months).astype(int)
    return df


def label_encode_categorical(
    df: pd.DataFrame,
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Label encode categorical columns.
    
    Returns:
        Encoded dataframe and mapping dictionary
    """
    df = df.copy()
    mappings = {}
    
    for col in categorical_cols:
        if col in df.columns:
            unique_vals = sorted(df[col].dropna().unique())
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            mappings[col] = mapping
            
            df[f"{col}_encoded"] = df[col].map(mapping)
            df[f"{col}_encoded"] = df[f"{col}_encoded"].astype("category")
    
    return df, mappings


def plot_feature_importance(model, feature_names: List[str], top_n: int = 20, save_path: Optional[Path] = None):
    """Plot feature importance from LightGBM model."""
    importance = model.feature_importance(importance_type="gain")
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df, y="feature", x="importance")
    plt.title(f"Top {top_n} Feature Importance")
    plt.xlabel("Importance (Gain)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_validation_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Validation Results",
    save_path: Optional[Path] = None
):
    """Plot validation predictions vs actuals."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    
    plt.subplot(1, 2, 2)
    residuals = y_pred - y_true
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    fold_idx: int = None,
    save_path: Optional[Path] = None
):
    """
    Plot predicted values against actual values (scatter plot with diagonal line).
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
        fold_idx: Fold index (optional)
        save_path: Path to save the plot
    """
    import logging
    logger = logging.getLogger(__name__)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot: Predicted vs Actual
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate R² for display
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    ax1.set_xlabel("Actual Quantity", fontsize=12)
    ax1.set_ylabel("Predicted Quantity", fontsize=12)
    title = f"Predicted vs Actual - {model_name.upper()}"
    if fold_idx:
        title += f" (Fold {fold_idx})"
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add text box with metrics
    textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Residual plot
    ax2 = axes[1]
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel("Predicted Quantity", fontsize=12)
    ax2.set_ylabel("Residuals (Predicted - Actual)", fontsize=12)
    ax2.set_title("Residual Plot", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved predicted vs actual plot to {save_path}")
    plt.close()


def plot_actual_vs_predicted_timeseries(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    fold_idx: int = None,
    save_path: Optional[Path] = None
):
    """
    Plot actual vs predicted values as a time series line chart.
    
    Args:
        dates: Date array for x-axis
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
        fold_idx: Fold index (optional)
        save_path: Path to save the plot
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Convert dates to pandas datetime if not already
    dates = pd.to_datetime(dates)
    
    # Calculate metrics for display
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot actual and predicted as lines
    ax.plot(dates, y_true, label='Actual', color='blue', linewidth=2, alpha=0.7, marker='o', markersize=3)
    ax.plot(dates, y_pred, label='Predicted', color='red', linewidth=2, alpha=0.7, linestyle='--', marker='s', markersize=3)
    
    # Fill area between actual and predicted to show error
    ax.fill_between(dates, y_true, y_pred, alpha=0.2, color='gray', label='Error')
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Quantity", fontsize=12)
    title = f"Actual vs Predicted Sales - {model_name.upper()}"
    if fold_idx:
        title += f" (Fold {fold_idx})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add text box with metrics
    textstr = f'RMSE = {rmse:.2f}\nMAE = {mae:.2f}\nR² = {r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved time series plot to {save_path}")
    plt.close()


def plot_forecast_vs_historical(
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
    date_col: str = "Date",
    value_col: str = "Quantity",
    save_path: Optional[Path] = None
):
    """Plot forecast alongside historical data."""
    plt.figure(figsize=(14, 6))
    
    # Plot historical - group by Branch and Tonnage only
    group_cols = ["Branch", "Tonnage"] if "Branch" in historical.columns and "Tonnage" in historical.columns else []
    if group_cols:
        for group, data in historical.groupby(group_cols):
            plt.plot(data[date_col], data[value_col], alpha=0.3, color="blue", label="Historical" if group == list(historical.groupby(group_cols))[0][0] else "")
        
        # Plot forecast
        for group, data in forecast.groupby(group_cols):
            plt.plot(data[date_col], data["forecast"], alpha=0.7, color="red", linestyle="--", label="Forecast" if group == list(forecast.groupby(group_cols))[0][0] else "")
    else:
        # Fallback if grouping columns don't exist
        plt.plot(historical[date_col], historical[value_col], alpha=0.3, color="blue", label="Historical")
        plt.plot(forecast[date_col], forecast["forecast"], alpha=0.7, color="red", linestyle="--", label="Forecast")
    
    plt.xlabel("Date")
    plt.ylabel("Quantity")
    plt.title("Historical Data vs Forecast")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

