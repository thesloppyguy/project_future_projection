"""
Diagnostic tools for analyzing forecast issues.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ForecastDiagnostics:
    """Diagnostic tools for forecast analysis."""
    
    def __init__(self, results_dir: Path):
        """
        Initialize diagnostics.
        
        Args:
            results_dir: Path to results directory
        """
        self.results_dir = results_dir
    
    def diagnose_weekly_forecasts(
        self,
        forecasts: Dict[str, pd.Series],
        actuals: pd.Series,
        series_key: str
    ) -> Dict:
        """
        Diagnose issues with weekly forecasts.
        
        Args:
            forecasts: Dictionary of forecasts {model_name: forecast_series}
            actuals: Actual values series
            series_key: Series identifier
            
        Returns:
            Dictionary with diagnostic information
        """
        logger.info(f"Diagnosing weekly forecasts for {series_key}")
        
        diagnostics = {
            'series_key': series_key,
            'models': {},
            'summary': {}
        }
        
        # Check actuals
        actuals_stats = {
            'count': len(actuals),
            'min': actuals.min(),
            'max': actuals.max(),
            'mean': actuals.mean(),
            'median': actuals.median(),
            'std': actuals.std(),
            'zeros': (actuals == 0).sum(),
            'negative': (actuals < 0).sum(),
            'very_small': (actuals < 1).sum() if actuals.min() >= 0 else None,
            'date_range': (actuals.index.min(), actuals.index.max()),
            'frequency': pd.infer_freq(actuals.index) if len(actuals) > 1 else None
        }
        diagnostics['actuals_stats'] = actuals_stats
        
        # Check each model's forecast
        for model_name, forecast in forecasts.items():
            if model_name == 'ensemble':
                continue
                
            model_diag = {
                'forecast_count': len(forecast),
                'forecast_min': forecast.min(),
                'forecast_max': forecast.max(),
                'forecast_mean': forecast.mean(),
                'forecast_median': forecast.median(),
                'forecast_std': forecast.std(),
                'forecast_zeros': (forecast == 0).sum(),
                'forecast_negative': (forecast < 0).sum(),
                'forecast_very_small': (forecast < 1).sum() if forecast.min() >= 0 else None,
                'date_range': (forecast.index.min(), forecast.index.max()),
                'frequency': pd.infer_freq(forecast.index) if len(forecast) > 1 else None
            }
            
            # Alignment check
            common_dates = forecast.index.intersection(actuals.index)
            model_diag['common_dates_count'] = len(common_dates)
            model_diag['alignment_ratio'] = len(common_dates) / len(actuals) if len(actuals) > 0 else 0
            
            # Date alignment issues
            if len(common_dates) > 0:
                forecast_aligned = forecast.loc[common_dates]
                actuals_aligned = actuals.loc[common_dates]
                
                # Check for extreme differences
                errors = forecast_aligned - actuals_aligned
                model_diag['error_mean'] = errors.mean()
                model_diag['error_std'] = errors.std()
                model_diag['error_max'] = errors.abs().max()
                
                # MAPE calculation issues
                non_zero_actuals = actuals_aligned[actuals_aligned != 0]
                if len(non_zero_actuals) > 0:
                    mape_values = np.abs(
                        (forecast_aligned[non_zero_actuals.index] - non_zero_actuals) / non_zero_actuals
                    ) * 100
                    model_diag['mape_mean'] = mape_values.mean()
                    model_diag['mape_median'] = mape_values.median()
                    model_diag['mape_max'] = mape_values.max()
                    model_diag['mape_extreme_count'] = (mape_values > 1000).sum()
                else:
                    model_diag['mape_mean'] = np.nan
                    model_diag['mape_median'] = np.nan
                    model_diag['mape_max'] = np.nan
                    model_diag['mape_extreme_count'] = 0
            else:
                model_diag['alignment_issue'] = 'NO_COMMON_DATES'
            
            diagnostics['models'][model_name] = model_diag
        
        # Summary
        diagnostics['summary'] = {
            'total_models': len(forecasts),
            'models_with_alignment_issues': sum(
                1 for m in diagnostics['models'].values() 
                if m.get('alignment_ratio', 0) < 0.5
            ),
            'models_with_negative_forecasts': sum(
                1 for m in diagnostics['models'].values()
                if m.get('forecast_negative', 0) > 0
            ),
            'models_with_zero_forecasts': sum(
                1 for m in diagnostics['models'].values()
                if m.get('forecast_zeros', 0) > 0
            )
        }
        
        return diagnostics
    
    def save_diagnostics_report(
        self,
        diagnostics: Dict,
        output_path: Path
    ):
        """Save diagnostics report to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, model_diag in diagnostics['models'].items():
            row = {'model': model_name}
            row.update(model_diag)
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, index=False)
        
        logger.info(f"Diagnostics report saved to {output_path}")
    
    def plot_forecast_comparison(
        self,
        forecasts: Dict[str, pd.Series],
        actuals: pd.Series,
        series_key: str,
        output_path: Optional[Path] = None,
        top_n: int = 5
    ):
        """
        Plot forecast vs actuals comparison.
        
        Args:
            forecasts: Dictionary of forecasts
            actuals: Actual values
            series_key: Series identifier
            output_path: Path to save plot
            top_n: Number of top models to plot
        """
        # Select top N models by RMSE
        model_rmse = {}
        for model_name, forecast in forecasts.items():
            if model_name == 'ensemble':
                continue
            common_dates = forecast.index.intersection(actuals.index)
            if len(common_dates) > 0:
                forecast_aligned = forecast.loc[common_dates]
                actuals_aligned = actuals.loc[common_dates]
                rmse = np.sqrt(np.mean((actuals_aligned - forecast_aligned) ** 2))
                model_rmse[model_name] = rmse
        
        # Sort by RMSE and take top N
        sorted_models = sorted(model_rmse.items(), key=lambda x: x[1])[:top_n]
        top_models = [m[0] for m in sorted_models]
        
        # Create plot
        fig, axes = plt.subplots(len(top_models) + 1, 1, figsize=(14, 4 * (len(top_models) + 1)))
        if len(top_models) == 0:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot actuals
        axes[0].plot(actuals.index, actuals.values, 'k-', linewidth=2, label='Actuals', marker='o')
        axes[0].set_title(f'{series_key} - Actuals', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Quantity')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot each top model
        for idx, model_name in enumerate(top_models, 1):
            forecast = forecasts[model_name]
            common_dates = forecast.index.intersection(actuals.index)
            
            if len(common_dates) > 0:
                forecast_aligned = forecast.loc[common_dates]
                actuals_aligned = actuals.loc[common_dates]
                
                axes[idx].plot(actuals_aligned.index, actuals_aligned.values, 
                              'k-', linewidth=2, label='Actuals', marker='o', markersize=4)
                axes[idx].plot(forecast_aligned.index, forecast_aligned.values,
                              'r--', linewidth=1.5, label=f'{model_name} Forecast', marker='s', markersize=3)
                axes[idx].set_title(f'{model_name} (RMSE: {model_rmse[model_name]:.2f})', 
                                   fontsize=11)
                axes[idx].set_ylabel('Quantity')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_residuals_analysis(
        self,
        forecasts: Dict[str, pd.Series],
        actuals: pd.Series,
        series_key: str,
        output_path: Optional[Path] = None,
        top_n: int = 5
    ):
        """Plot residuals analysis for top models."""
        # Select top N models
        model_rmse = {}
        for model_name, forecast in forecasts.items():
            if model_name == 'ensemble':
                continue
            common_dates = forecast.index.intersection(actuals.index)
            if len(common_dates) > 0:
                forecast_aligned = forecast.loc[common_dates]
                actuals_aligned = actuals.loc[common_dates]
                rmse = np.sqrt(np.mean((actuals_aligned - forecast_aligned) ** 2))
                model_rmse[model_name] = rmse
        
        sorted_models = sorted(model_rmse.items(), key=lambda x: x[1])[:top_n]
        top_models = [m[0] for m in sorted_models]
        
        if len(top_models) == 0:
            logger.warning("No models to plot")
            return
        
        fig, axes = plt.subplots(2, len(top_models), figsize=(5 * len(top_models), 10))
        if len(top_models) == 1:
            axes = axes.reshape(2, 1)
        
        for idx, model_name in enumerate(top_models):
            forecast = forecasts[model_name]
            common_dates = forecast.index.intersection(actuals.index)
            
            if len(common_dates) > 0:
                forecast_aligned = forecast.loc[common_dates]
                actuals_aligned = actuals.loc[common_dates]
                residuals = actuals_aligned - forecast_aligned
                
                # Residuals over time
                axes[0, idx].plot(actuals_aligned.index, residuals.values, 'o-', alpha=0.6)
                axes[0, idx].axhline(y=0, color='r', linestyle='--', linewidth=1)
                axes[0, idx].set_title(f'{model_name} - Residuals')
                axes[0, idx].set_ylabel('Residual')
                axes[0, idx].grid(True, alpha=0.3)
                
                # Residuals distribution
                axes[1, idx].hist(residuals.values, bins=20, alpha=0.7, edgecolor='black')
                axes[1, idx].axvline(x=0, color='r', linestyle='--', linewidth=1)
                axes[1, idx].set_title(f'{model_name} - Distribution')
                axes[1, idx].set_xlabel('Residual')
                axes[1, idx].set_ylabel('Frequency')
                axes[1, idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residuals plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()

