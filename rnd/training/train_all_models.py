"""
Main orchestrator script for training all 16 time series forecasting models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from typing import Dict, List
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training.data_preprocessing import prepare_data_for_training
from training.evaluation import evaluate_forecast, save_evaluation_results, generate_summary_report

# Import all models
from training.models import auto_arima
from training.models import sarimax
from training.models import holt_winters
from training.models import prophet
from training.models import neural_prophet
from training.models import stl_decomposition
from training.models import xgboost
from training.models import lightgbm
from training.models import catboost
from training.models import random_forest
from training.models import lstm
from training.models import bayesian_deep_learning
from training.models import isolation_forest
from training.models import quantile_regression
from training.models import var
from training.models import kalman_filter


# Model configurations
MODELS = {
    'auto_arima': {
        'module': auto_arima,
        'name': 'Auto-ARIMA'
    },
    'sarimax': {
        'module': sarimax,
        'name': 'SARIMAX'
    },
    'holt_winters': {
        'module': holt_winters,
        'name': 'Holt-Winters'
    },
    'prophet': {
        'module': prophet,
        'name': 'Prophet'
    },
    'neural_prophet': {
        'module': neural_prophet,
        'name': 'Neural Prophet'
    },
    'stl_decomposition': {
        'module': stl_decomposition,
        'name': 'STL Decomposition'
    },
    'xgboost': {
        'module': xgboost,
        'name': 'XGBoost'
    },
    'lightgbm': {
        'module': lightgbm,
        'name': 'LightGBM'
    },
    'catboost': {
        'module': catboost,
        'name': 'CatBoost'
    },
    'random_forest': {
        'module': random_forest,
        'name': 'Random Forest'
    },
    'lstm': {
        'module': lstm,
        'name': 'LSTM'
    },
    'bayesian_deep_learning': {
        'module': bayesian_deep_learning,
        'name': 'Bayesian Deep Learning'
    },
    'isolation_forest': {
        'module': isolation_forest,
        'name': 'Isolation Forest'
    },
    'quantile_regression': {
        'module': quantile_regression,
        'name': 'Quantile Regression'
    },
    'var': {
        'module': var,
        'name': 'Vector Autoregression'
    },
    'kalman_filter': {
        'module': kalman_filter,
        'name': 'Kalman Filter'
    }
}

# Branches
BRANCHES = ['BLR', 'COK', 'MAA', 'SBD', 'SBD1']

# Forecast periods
WEEKLY_PERIODS = 52
MONTHLY_PERIODS = 12


def train_and_forecast_model(
    model_key: str,
    model_config: dict,
    train_data: Dict[str, pd.Series],
    test_data: Dict[str, pd.Series],
    branch: str,
    aggregation: str,
    results_dir: Path
) -> Dict:
    """
    Train a model and generate forecast for a specific branch and aggregation.
    
    Args:
        model_key: Model key (e.g., 'auto_arima')
        model_config: Model configuration dictionary
        train_data: Training data series
        test_data: Test data series
        branch: Branch name
        aggregation: 'weekly' or 'monthly'
        results_dir: Results directory path
        
    Returns:
        Evaluation dictionary
    """
    model_module = model_config['module']
    model_name = model_config['name']
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} - {branch} - {aggregation}")
    print(f"{'='*60}")
    
    try:
        # Get training and test data
        train_series = train_data.get(branch, pd.Series(dtype=float))
        test_series = test_data.get(branch, pd.Series(dtype=float))
        
        if len(train_series) == 0:
            print(f"Warning: No training data for {branch} - {aggregation}")
            return {
                'model': model_name,
                'branch': branch,
                'aggregation': aggregation,
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'nrmse': np.nan,
                'n_samples': 0,
                'status': 'no_data'
            }
        
        # Determine frequency and forecast periods
        if aggregation == 'weekly':
            freq = 'W-MON'
            n_periods = WEEKLY_PERIODS
        else:
            freq = 'MS'
            n_periods = MONTHLY_PERIODS
        
        # Special handling for VAR (needs all branches)
        if model_key == 'var':
            # Train VAR on all branches
            all_train_data = {}
            all_test_data = {}
            for b in BRANCHES:
                if b in train_data and len(train_data[b]) > 0:
                    all_train_data[b] = train_data[b]
                if b in test_data and len(test_data[b]) > 0:
                    all_test_data[b] = test_data[b]
            
            if len(all_train_data) < 2:
                print(f"Warning: VAR needs at least 2 branches, got {len(all_train_data)}")
                return {
                    'model': model_name,
                    'branch': branch,
                    'aggregation': aggregation,
                    'rmse': np.nan,
                    'mae': np.nan,
                    'mape': np.nan,
                    'nrmse': np.nan,
                    'n_samples': 0,
                    'status': 'insufficient_branches'
                }
            
            # Train model with hyperparameter optimization
            model = model_module.train_model(all_train_data, BRANCHES, freq, 
                                            use_optimization=True, n_trials=20)
        else:
            # Train model with hyperparameter optimization
            model = model_module.train_model(train_series, branch, freq,
                                            use_optimization=True, n_trials=20)
        
        if model is None:
            print(f"Warning: Model training failed for {model_name} - {branch} - {aggregation}")
            return {
                'model': model_name,
                'branch': branch,
                'aggregation': aggregation,
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'nrmse': np.nan,
                'r2': np.nan,
                'n_samples': 0,
                'status': 'training_failed'
            }
        
        # Generate forecast
        forecast_series = model_module.forecast(model, n_periods, freq, branch)
        
        if len(forecast_series) == 0:
            print(f"Warning: Forecast generation failed for {model_name} - {branch} - {aggregation}")
            return {
                'model': model_name,
                'branch': branch,
                'aggregation': aggregation,
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'nrmse': np.nan,
                'r2': np.nan,
                'n_samples': 0,
                'status': 'forecast_failed'
            }
        
        # Save forecast
        forecast_dir = results_dir / aggregation / model_key
        forecast_dir.mkdir(parents=True, exist_ok=True)
        forecast_path = forecast_dir / f"{branch}_forecast.csv"
        forecast_df = pd.DataFrame({
            'Date': forecast_series.index,
            'Forecast': forecast_series.values
        })
        forecast_df.to_csv(forecast_path, index=False)
        
        # Save model (if applicable)
        model_dir = results_dir / 'models' / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{branch}_{aggregation}.pkl"
        try:
            model_module.save_model(model, str(model_path))
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
        
        # Evaluate forecast
        if len(test_series) > 0:
            evaluation = evaluate_forecast(
                test_series,
                forecast_series,
                model_name,
                branch,
                aggregation
            )
            evaluation['status'] = 'success'
        else:
            print(f"Warning: No test data for {branch} - {aggregation}")
            evaluation = {
                'model': model_name,
                'branch': branch,
                'aggregation': aggregation,
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'nrmse': np.nan,
                'r2': np.nan,
                'n_samples': 0,
                'status': 'no_test_data'
            }
        
        print(f"✓ {model_name} - {branch} - {aggregation}: RMSE={evaluation.get('rmse', np.nan):.2f}, MAE={evaluation.get('mae', np.nan):.2f}, R2={evaluation.get('r2', np.nan):.4f}")
        
        return evaluation
        
    except Exception as e:
        print(f"Error in {model_name} - {branch} - {aggregation}: {e}")
        traceback.print_exc()
        return {
            'model': model_name,
            'branch': branch,
            'aggregation': aggregation,
            'rmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'nrmse': np.nan,
            'r2': np.nan,
            'n_samples': 0,
            'status': f'error: {str(e)}'
        }


def main():
    """Main function to train all models."""
    print("="*60)
    print("Multi-Model Time Series Forecasting Pipeline")
    print("="*60)
    
    # Paths
    train_path = Path('data/merged_filter_ingestion_2024.csv')
    test_path = Path('data/merged_filter_ingestion_2025.csv')
    results_dir = Path('training/results')
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    print("\nLoading and preprocessing data...")
    data = prepare_data_for_training(
        str(train_path),
        str(test_path),
        branches=BRANCHES
    )
    
    print(f"Data loaded successfully!")
    print(f"  Branches: {BRANCHES}")
    print(f"  Training periods (weekly): {[len(data['train']['weekly'][b]) for b in BRANCHES]}")
    print(f"  Training periods (monthly): {[len(data['train']['monthly'][b]) for b in BRANCHES]}")
    
    # Store all evaluation results
    all_evaluations = []
    
    # Train all models
    for model_key, model_config in MODELS.items():
        print(f"\n{'#'*60}")
        print(f"Processing Model: {model_config['name']}")
        print(f"{'#'*60}")
        
        # Process weekly aggregation
        for branch in BRANCHES:
            evaluation = train_and_forecast_model(
                model_key,
                model_config,
                data['train']['weekly'],
                data['test']['weekly'],
                branch,
                'weekly',
                results_dir
            )
            all_evaluations.append(evaluation)
        
        # Process monthly aggregation
        for branch in BRANCHES:
            evaluation = train_and_forecast_model(
                model_key,
                model_config,
                data['train']['monthly'],
                data['test']['monthly'],
                branch,
                'monthly',
                results_dir
            )
            all_evaluations.append(evaluation)
    
    # Save evaluation results
    print(f"\n{'='*60}")
    print("Saving evaluation results...")
    print(f"{'='*60}")
    
    # Save per-model evaluations
    for model_key, model_config in MODELS.items():
        model_name = model_config['name']
        
        # Weekly evaluations
        weekly_evals = [e for e in all_evaluations if e['model'] == model_name and e['aggregation'] == 'weekly']
        if weekly_evals:
            weekly_path = results_dir / 'evaluations' / f"{model_key}_weekly_metrics.csv"
            save_evaluation_results(weekly_evals, str(weekly_path))
        
        # Monthly evaluations
        monthly_evals = [e for e in all_evaluations if e['model'] == model_name and e['aggregation'] == 'monthly']
        if monthly_evals:
            monthly_path = results_dir / 'evaluations' / f"{model_key}_monthly_metrics.csv"
            save_evaluation_results(monthly_evals, str(monthly_path))
    
    # Generate summary report
    summary_path = results_dir / 'model_comparison_summary.csv'
    generate_summary_report(all_evaluations, str(summary_path))
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Results saved to: {results_dir}")
    print(f"Summary report: {summary_path}")


if __name__ == '__main__':
    main()

