"""
Evaluation module for blind testing and model comparison.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate forecasts on blind data."""
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate_forecast(
        self,
        forecast: pd.Series,
        actuals: pd.Series,
        model_name: str = "unknown"
    ) -> Dict[str, float]:
        """
        Evaluate forecast against actuals.
        
        Args:
            forecast: Forecast time series
            actuals: Actual time series
            model_name: Name of model
            
        Returns:
            Dictionary of metrics
        """
        # Align on common dates
        common_dates = forecast.index.intersection(actuals.index)
        
        # If no exact matches, try to align with tolerance (for weekly data)
        if len(common_dates) == 0:
            # Try to find nearest dates within tolerance
            # For weekly: 3 days, for monthly: 7 days
            tolerance_days = 3 if len(forecast) > 0 and (forecast.index[-1] - forecast.index[0]).days / max(len(forecast), 1) < 10 else 7
            
            forecast_aligned = pd.Series(dtype=float)
            actuals_aligned = pd.Series(dtype=float)
            used_actual_indices = set()  # Avoid using same actual for multiple forecasts
            
            for fcst_date in forecast.index:
                # Find closest actual date within tolerance that hasn't been used
                available_actuals = actuals[~actuals.index.isin(used_actual_indices)]
                if len(available_actuals) == 0:
                    break
                    
                # Calculate absolute differences (TimedeltaIndex doesn't have .abs(), convert to Series first)
                date_diffs_timedelta = available_actuals.index - fcst_date
                date_diffs = pd.Series(
                    [abs(td.total_seconds() / 86400) for td in date_diffs_timedelta],
                    index=available_actuals.index
                )
                min_diff_idx = date_diffs.idxmin()
                min_diff_days = date_diffs.loc[min_diff_idx]
                min_diff = pd.Timedelta(days=min_diff_days)
                
                if min_diff <= pd.Timedelta(days=tolerance_days):
                    forecast_aligned.loc[fcst_date] = forecast.loc[fcst_date]
                    actuals_aligned.loc[fcst_date] = actuals.loc[min_diff_idx]
                    used_actual_indices.add(min_diff_idx)
            
            if len(forecast_aligned) == 0:
                logger.warning(
                    f"No common dates between forecast and actuals for {model_name}. "
                    f"Forecast dates: {forecast.index.min()} to {forecast.index.max()} ({len(forecast)} periods), "
                    f"Actual dates: {actuals.index.min()} to {actuals.index.max()} ({len(actuals)} periods)"
                )
                return self._empty_metrics()
            
            # Use forecast dates as index for alignment
            common_dates = forecast_aligned.index
            logger.info(
                f"Aligned {len(forecast_aligned)} periods for {model_name} using tolerance of {tolerance_days} days"
            )
        else:
            forecast_aligned = forecast.loc[common_dates]
            actuals_aligned = actuals.loc[common_dates]
        
        return self._calculate_metrics(actuals_aligned, forecast_aligned)
    
    def _calculate_metrics(
        self,
        actuals: pd.Series,
        forecast: pd.Series
    ) -> Dict[str, float]:
        """Calculate all evaluation metrics."""
        if len(actuals) == 0 or len(forecast) == 0:
            return self._empty_metrics()
        
        # RMSE
        rmse = np.sqrt(np.mean((actuals - forecast) ** 2))
        
        # MAE
        mae = np.mean(np.abs(actuals - forecast))
        
        # MAPE (handle zero actuals)
        non_zero_mask = actuals != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(
                np.abs((actuals[non_zero_mask] - forecast[non_zero_mask]) / 
                      actuals[non_zero_mask])
            ) * 100
        else:
            mape = np.nan
        
        # NRMSE (normalized by mean)
        if actuals.mean() != 0:
            nrmse = (rmse / actuals.mean()) * 100
        else:
            nrmse = np.nan
        
        # R²
        ss_res = np.sum((actuals - forecast) ** 2)
        ss_tot = np.sum((actuals - actuals.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        
        # Mean Error (bias)
        mean_error = np.mean(forecast - actuals)
        
        # Mean Absolute Percentage Error (alternative calculation)
        mape_alt = np.mean(np.abs((actuals - forecast) / (actuals + 1e-10))) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'nrmse': nrmse,
            'r2': r2,
            'mean_error': mean_error,
            'mape_alt': mape_alt,
            'n_samples': len(actuals)
        }
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'nrmse': np.nan,
            'r2': np.nan,
            'mean_error': np.nan,
            'mape_alt': np.nan,
            'n_samples': 0
        }
    
    def compare_models(
        self,
        model_results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            model_results: Dictionary with structure:
                {
                    'model_name': {
                        'metric_name': value,
                        ...
                    }
                }
        
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, metrics in model_results.items():
            row = {'model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by RMSE (lower is better)
        if 'rmse' in df.columns:
            df = df.sort_values('rmse')
        
        return df
    
    def create_ensemble_forecast(
        self,
        forecasts: Dict[str, pd.Series],
        method: str = 'weighted_average',
        weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Create ensemble forecast from multiple models.
        
        Args:
            forecasts: Dictionary of forecasts {model_name: forecast_series}
            method: Ensemble method ('average', 'weighted_average', 'median')
            weights: Weights for weighted average (if None, equal weights)
        
        Returns:
            Ensemble forecast series
        """
        if len(forecasts) == 0:
            logger.warning("No forecasts provided for ensemble")
            return pd.Series(dtype=float)
        
        # Get common dates
        all_dates = set()
        for forecast in forecasts.values():
            all_dates.update(forecast.index)
        all_dates = sorted(all_dates)
        
        # Align all forecasts
        aligned_forecasts = {}
        for model_name, forecast in forecasts.items():
            aligned = forecast.reindex(all_dates, method='ffill').fillna(0)
            aligned_forecasts[model_name] = aligned
        
        # Create ensemble
        if method == 'average':
            ensemble = pd.Series(
                np.mean([f.values for f in aligned_forecasts.values()], axis=0),
                index=all_dates
            )
        elif method == 'weighted_average':
            if weights is None:
                weights = {name: 1.0 / len(forecasts) for name in forecasts.keys()}
            
            weighted_sum = np.zeros(len(all_dates))
            total_weight = 0
            
            for model_name, forecast in aligned_forecasts.items():
                weight = weights.get(model_name, 1.0 / len(forecasts))
                weighted_sum += forecast.values * weight
                total_weight += weight
            
            ensemble = pd.Series(weighted_sum / total_weight, index=all_dates)
        elif method == 'median':
            ensemble = pd.Series(
                np.median([f.values for f in aligned_forecasts.values()], axis=0),
                index=all_dates
            )
        else:
            logger.warning(f"Unknown ensemble method: {method}, using average")
            ensemble = pd.Series(
                np.mean([f.values for f in aligned_forecasts.values()], axis=0),
                index=all_dates
            )
        
        return ensemble
    
    def save_evaluation_results(
        self,
        results: Dict,
        output_dir: Path
    ):
        """Save evaluation results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model comparison
        if 'model_comparison' in results:
            results['model_comparison'].to_csv(
                output_dir / 'model_comparison.csv',
                index=False
            )
        
        # Save detailed metrics
        if 'detailed_metrics' in results:
            results['detailed_metrics'].to_csv(
                output_dir / 'detailed_metrics.csv',
                index=False
            )
        
        # Save forecasts vs actuals
        if 'forecast_comparison' in results:
            results['forecast_comparison'].to_csv(
                output_dir / 'forecast_comparison.csv',
                index=False
            )
        
        logger.info(f"Evaluation results saved to {output_dir}")

