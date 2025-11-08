"""
Train specific models for specific branches and calculate monthly forecasts with share percentages.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from typing import Dict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training.data_preprocessing import (
    filter_by_branch_quantity,
    create_monthly_aggregates,
    get_branch_series
)
from training.models import random_forest, catboost, quantile_regression, xgboost


# Model-Branch mappings
MODEL_BRANCH_MAPPINGS = {
    'random_forest': {
        'module': random_forest,
        'branch': 'BLR'
    },
    'catboost_cok': {
        'module': catboost,
        'branch': 'COK'
    },
    'quantile_regression': {
        'module': quantile_regression,
        'branch': 'MAA'
    },
    'catboost_sbd': {
        'module': catboost,
        'branch': 'SBD'
    },
    'xgboost': {
        'module': xgboost,
        'branch': 'SBD1'
    }
}


def get_financial_year(dt):
    """Calculate Financial Year from Date. FY starts in April (month >= 4 means next year."""
    if pd.isnull(dt):
        return None
    return dt.year + 1 if dt.month >= 4 else dt.year


def load_and_filter_data(data_path: str, cutoff_date: str = '2025-04-01') -> pd.DataFrame:
    """
    Load data and filter to dates before cutoff_date.
    
    Args:
        data_path: Path to CSV file
        cutoff_date: Date string to filter data (exclusive)
        
    Returns:
        Filtered dataframe
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter to dates before cutoff
    cutoff = pd.to_datetime(cutoff_date)
    df = df[df['Date'] < cutoff]
    
    print(f"Data loaded: {len(df)} rows, date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df


def prepare_monthly_series(df: pd.DataFrame, branch: str) -> pd.Series:
    """
    Prepare monthly time series for a specific branch.
    
    Args:
        df: Dataframe with Date, Branch, Quantity columns
        branch: Branch name
        
    Returns:
        Monthly time series with Date index
    """
    # Filter to Branch and Quantity
    branch_df = filter_by_branch_quantity(df)
    
    # Filter to specific branch
    branch_df = branch_df[branch_df['Branch'] == branch].copy()
    
    if len(branch_df) == 0:
        return pd.Series(dtype=float)
    
    # Create monthly aggregates
    monthly_df = create_monthly_aggregates(branch_df)
    
    # Extract series
    series = get_branch_series(monthly_df, branch)
    
    return series


def train_and_forecast(model_key: str, model_config: dict, train_series: pd.Series, 
                      branch: str, results_dir: Path) -> pd.Series:
    """
    Train model and generate forecast.
    
    Args:
        model_key: Model key identifier
        model_config: Model configuration dict with 'module' key
        train_series: Training time series
        branch: Branch name
        results_dir: Results directory path
        
    Returns:
        Forecast series with Date index
    """
    model_module = model_config['module']
    model_name = model_key
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} for branch {branch}")
    print(f"{'='*60}")
    
    if len(train_series) == 0:
        print(f"Warning: No training data for {branch}")
        return pd.Series(dtype=float)
    
    try:
        # Train model with hyperparameter optimization
        freq = 'MS'  # Monthly start
        model = model_module.train_model(
            train_series, 
            branch, 
            freq,
            use_optimization=True, 
            n_trials=20
        )
        
        if model is None:
            print(f"Warning: Model training failed for {model_name} - {branch}")
            return pd.Series(dtype=float)
        
        # Generate 12-month forecast
        forecast_series = model_module.forecast(model, n_periods=12, freq=freq, branch=branch)
        
        if len(forecast_series) == 0:
            print(f"Warning: Forecast generation failed for {model_name} - {branch}")
            return pd.Series(dtype=float)
        
        # Save model
        model_dir = results_dir / 'models' / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{branch}_monthly.pkl"
        try:
            model_module.save_model(model, str(model_path))
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
        
        # Save forecast
        forecast_dir = results_dir / 'monthly' / model_name
        forecast_dir.mkdir(parents=True, exist_ok=True)
        forecast_path = forecast_dir / f"{branch}_forecast.csv"
        forecast_df = pd.DataFrame({
            'Date': forecast_series.index,
            'Forecast': forecast_series.values
        })
        forecast_df.to_csv(forecast_path, index=False)
        print(f"Forecast saved to {forecast_path}")
        
        print(f"✓ {model_name} - {branch}: Forecast generated for {len(forecast_series)} months")
        
        return forecast_series
        
    except Exception as e:
        print(f"Error in {model_name} - {branch}: {e}")
        import traceback
        traceback.print_exc()
        return pd.Series(dtype=float)


def calculate_monthly_share_percentages(df: pd.DataFrame, cutoff_fy: int = 2025) -> pd.DataFrame:
    """
    Calculate monthly share percentages using Financial Year grouping and linear trend projection.
    
    Args:
        df: Dataframe with Date, Branch, Quantity columns
        cutoff_fy: Financial Year cutoff (exclusive)
        
    Returns:
        DataFrame with Branch, Month, Predicted_Monthly_Share_Trend columns
    """
    print("\nCalculating monthly share percentages...")
    
    # Filter to Branch and Quantity
    branch_df = filter_by_branch_quantity(df)
    
    # Calculate Financial Year
    branch_df['FinancialYear'] = branch_df['Date'].apply(get_financial_year)
    branch_df['Month'] = branch_df['Date'].dt.month
    
    # Group by Branch, FinancialYear, Month and sum Quantity
    fy_month_summary = branch_df.groupby(
        ['Branch', 'FinancialYear', 'Month'], 
        as_index=False
    )['Quantity'].sum()
    
    # Compute total qty for each (Branch, FinancialYear)
    fy_total = branch_df.groupby(
        ['Branch', 'FinancialYear'], 
        as_index=False
    )['Quantity'].sum().rename(columns={'Quantity': 'FY_Total'})
    
    # Merge total with monthly summary
    fy_month_summary = fy_month_summary.merge(fy_total, on=['Branch', 'FinancialYear'], how='left')
    
    # Calculate share in % for each month in FY
    fy_month_summary['Monthly_Percent_of_FY'] = (
        100 * fy_month_summary['Quantity'] / fy_month_summary['FY_Total']
    )
    
    # Filter out data for FY > cutoff_fy
    limited_fy_month_summary = fy_month_summary[fy_month_summary['FinancialYear'] <= cutoff_fy]
    
    # Project monthly share for next year using linear trend
    projected_monthly_share = []
    
    for branch in limited_fy_month_summary['Branch'].unique():
        branch_df = limited_fy_month_summary[limited_fy_month_summary['Branch'] == branch]
        for month in range(1, 13):
            df = branch_df[branch_df['Month'] == month].sort_values('FinancialYear')
            if len(df) >= 3:
                # Use last 3 years
                last3 = df.tail(3)
                years = last3['FinancialYear'].values
                shares = last3['Monthly_Percent_of_FY'].values
                # Fit a linear trend (year vs share) and predict for the next year
                x = years
                y = shares
                coeffs = np.polyfit(x, y, 1)  # Linear
                predicted_share = np.polyval(coeffs, cutoff_fy)  # Predict for next FY
            elif len(df) > 0:
                # Not enough years to fit trend, fall back to last available value
                predicted_share = df['Monthly_Percent_of_FY'].iloc[-1]
            else:
                predicted_share = np.nan
            
            projected_monthly_share.append({
                'Branch': branch,
                'Month': month,
                'Predicted_Monthly_Share_Trend': predicted_share
            })
    
    projected_monthly_share_df = pd.DataFrame(projected_monthly_share).sort_values(['Branch', 'Month'])
    
    print(f"Monthly share percentages calculated for {len(projected_monthly_share_df)} branch-month combinations")
    
    return projected_monthly_share_df


def calculate_monthly_values(forecasts: Dict[str, pd.Series], 
                            monthly_shares: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate actual monthly values using forecast totals and monthly share percentages.
    
    Args:
        forecasts: Dictionary mapping branch names to forecast series
        monthly_shares: DataFrame with Branch, Month, Predicted_Monthly_Share_Trend columns
        
    Returns:
        DataFrame with Branch, Month, Forecast_Total, Monthly_Share_Pct, Monthly_Value columns
    """
    print("\nCalculating monthly values...")
    
    results = []
    
    for branch, forecast_series in forecasts.items():
        if len(forecast_series) == 0:
            continue
        
        # Sum forecast values for next 12 months
        total_forecast = forecast_series.sum()
        
        # Get monthly shares for this branch
        branch_shares = monthly_shares[monthly_shares['Branch'] == branch]
        
        for _, row in branch_shares.iterrows():
            month = int(row['Month'])
            monthly_share_pct = row['Predicted_Monthly_Share_Trend']
            
            if pd.isna(monthly_share_pct):
                monthly_value = np.nan
            else:
                # Calculate: Monthly_Value = Total_Forecast * (Monthly_Share_Pct / 100)
                monthly_value = total_forecast * (monthly_share_pct / 100)
            
            results.append({
                'Branch': branch,
                'Month': month,
                'Forecast_Total': total_forecast,
                'Monthly_Share_Pct': monthly_share_pct,
                'Monthly_Value': monthly_value
            })
    
    results_df = pd.DataFrame(results).sort_values(['Branch', 'Month'])
    
    print(f"Monthly values calculated for {len(results_df)} branch-month combinations")
    
    return results_df


def main():
    """Main function to train models and calculate monthly forecasts."""
    print("="*60)
    print("Training Specific Models and Calculating Monthly Forecasts")
    print("="*60)
    
    # Paths
    data_path = Path('data/merged_filter_ingestion.csv')
    results_dir = Path('training/results')
    cutoff_date = '2025-04-01'
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and filter data
    df = load_and_filter_data(str(data_path), cutoff_date)
    
    # Store forecasts
    forecasts = {}
    
    # Train each model-branch pair
    for model_key, model_config in MODEL_BRANCH_MAPPINGS.items():
        branch = model_config['branch']
        
        # Prepare monthly series
        train_series = prepare_monthly_series(df, branch)
        
        if len(train_series) == 0:
            print(f"Warning: No data for branch {branch}")
            continue
        
        # Train and forecast
        forecast_series = train_and_forecast(
            model_key,
            model_config,
            train_series,
            branch,
            results_dir
        )
        
        if len(forecast_series) > 0:
            forecasts[branch] = forecast_series
    
    # Calculate monthly share percentages
    cutoff_fy = 2025  # FY2024-25 corresponds to dates < 2025-04-01
    monthly_shares = calculate_monthly_share_percentages(df, cutoff_fy)
    
    # Calculate actual monthly values
    monthly_values = calculate_monthly_values(forecasts, monthly_shares)
    
    # Save final results
    output_dir = results_dir / 'monthly_forecasts'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'final_monthly_values.csv'
    monthly_values.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Final monthly values saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Branches processed: {len(forecasts)}")
    print(f"  Total monthly values: {len(monthly_values)}")
    print(f"\nMonthly values preview:")
    print(monthly_values.head(20).to_string(index=False))


if __name__ == '__main__':
    main()

