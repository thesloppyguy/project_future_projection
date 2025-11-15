"""
Improved ensemble strategy with model filtering and calibration-based weighting.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleOptimizer:
    """Optimize ensemble forecasts by filtering poor models and weighting by performance."""
    
    def __init__(
        self,
        max_mape_threshold: float = 100.0,
        min_models: int = 2,
        use_calibration_weights: bool = True
    ):
        """
        Initialize ensemble optimizer.
        
        Args:
            max_mape_threshold: Maximum MAPE allowed for models in ensemble (default: 100%)
            min_models: Minimum number of models required for ensemble
            use_calibration_weights: Whether to use calibration-based weights
        """
        self.max_mape_threshold = max_mape_threshold
        self.min_models = min_models
        self.use_calibration_weights = use_calibration_weights
    
    def filter_models_by_performance(
        self,
        forecasts: Dict[str, pd.Series],
        actuals: pd.Series,
        calibration_results: Optional[Dict] = None
    ) -> Dict[str, pd.Series]:
        """
        Filter out models with poor performance.
        
        Args:
            forecasts: Dictionary of forecasts {model_name: forecast_series}
            actuals: Actual values for evaluation
            calibration_results: Optional calibration results for filtering
            
        Returns:
            Filtered dictionary of forecasts
        """
        filtered_forecasts = {}
        model_metrics = {}
        
        # Evaluate each model
        for model_name, forecast in forecasts.items():
            if model_name == 'ensemble':
                continue
            
            # Align forecast with actuals
            common_dates = forecast.index.intersection(actuals.index)
            if len(common_dates) == 0:
                logger.warning(f"No common dates for {model_name}, excluding from ensemble")
                continue
            
            forecast_aligned = forecast.loc[common_dates]
            actuals_aligned = actuals.loc[common_dates]
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((actuals_aligned - forecast_aligned) ** 2))
            mae = np.mean(np.abs(actuals_aligned - forecast_aligned))
            
            # MAPE (handle zero actuals)
            non_zero_mask = actuals_aligned != 0
            if non_zero_mask.sum() > 0:
                mape = np.mean(
                    np.abs((actuals_aligned[non_zero_mask] - forecast_aligned[non_zero_mask]) /
                          actuals_aligned[non_zero_mask])
                ) * 100
            else:
                mape = np.inf
            
            model_metrics[model_name] = {
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            }
            
            # Filter by MAPE threshold
            if mape <= self.max_mape_threshold:
                filtered_forecasts[model_name] = forecast
                logger.info(
                    f"  {model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}% - INCLUDED"
                )
            else:
                logger.warning(
                    f"  {model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}% - EXCLUDED "
                    f"(MAPE > {self.max_mape_threshold}%)"
                )
        
        # Ensure minimum number of models
        if len(filtered_forecasts) < self.min_models:
            logger.warning(
                f"Only {len(filtered_forecasts)} models passed filter, but minimum is {self.min_models}. "
                f"Including top {self.min_models} models by RMSE."
            )
            # Sort by RMSE and take top N
            sorted_models = sorted(model_metrics.items(), key=lambda x: x[1]['rmse'])[:self.min_models]
            filtered_forecasts = {name: forecasts[name] for name, _ in sorted_models}
        
        return filtered_forecasts, model_metrics
    
    def calculate_calibration_weights(
        self,
        calibration_results: Dict,
        model_names: List[str]
    ) -> Dict[str, float]:
        """
        Calculate ensemble weights based on calibration performance.
        
        Args:
            calibration_results: Calibration results dictionary
            model_names: List of model names to weight
            
        Returns:
            Dictionary of weights {model_name: weight}
        """
        weights = {}
        model_rmse = {}
        
        # Calculate average RMSE for each model from calibration
        for model_name in model_names:
            if model_name in calibration_results:
                metrics = calibration_results[model_name].get('metrics', [])
                if len(metrics) > 0:
                    # Get RMSE values
                    rmse_values = [
                        m.get('rmse', np.nan) for m in metrics
                        if not np.isnan(m.get('rmse', np.nan))
                    ]
                    if len(rmse_values) > 0:
                        avg_rmse = np.mean(rmse_values)
                        if avg_rmse > 0:
                            model_rmse[model_name] = avg_rmse
        
        # Calculate inverse weights (lower RMSE = higher weight)
        if len(model_rmse) > 0:
            total_inv_rmse = sum(1.0 / rmse for rmse in model_rmse.values())
            for model_name, rmse in model_rmse.items():
                weights[model_name] = (1.0 / rmse) / total_inv_rmse
        else:
            # Equal weights if no calibration data
            weights = {name: 1.0 / len(model_names) for name in model_names}
        
        return weights
    
    def create_optimized_ensemble(
        self,
        forecasts: Dict[str, pd.Series],
        actuals: Optional[pd.Series] = None,
        calibration_results: Optional[Dict] = None,
        method: str = 'weighted_average'
    ) -> Tuple[pd.Series, Dict]:
        """
        Create optimized ensemble forecast.
        
        Args:
            forecasts: Dictionary of forecasts
            actuals: Optional actual values for filtering
            calibration_results: Optional calibration results for weighting
            method: Ensemble method ('average', 'weighted_average', 'median')
        
        Returns:
            Tuple of (ensemble_forecast, metadata_dict)
        """
        metadata = {
            'original_models': list(forecasts.keys()),
            'filtered_models': [],
            'weights': {},
            'method': method
        }
        
        # Filter models if actuals provided
        if actuals is not None:
            filtered_forecasts, model_metrics = self.filter_models_by_performance(
                forecasts, actuals, calibration_results
            )
            metadata['filtered_models'] = list(filtered_forecasts.keys())
            metadata['model_metrics'] = model_metrics
        else:
            # No filtering, use all models except ensemble
            filtered_forecasts = {
                name: f for name, f in forecasts.items()
                if name != 'ensemble'
            }
            metadata['filtered_models'] = list(filtered_forecasts.keys())
        
        if len(filtered_forecasts) == 0:
            logger.warning("No models available for ensemble")
            return pd.Series(dtype=float), metadata
        
        # Calculate weights
        if self.use_calibration_weights and calibration_results is not None:
            weights = self.calculate_calibration_weights(
                calibration_results, list(filtered_forecasts.keys())
            )
            metadata['weights'] = weights
        else:
            # Equal weights
            weights = {name: 1.0 / len(filtered_forecasts) for name in filtered_forecasts.keys()}
            metadata['weights'] = weights
        
        # Get common dates
        all_dates = set()
        for forecast in filtered_forecasts.values():
            all_dates.update(forecast.index)
        all_dates = sorted(all_dates)
        
        # Align all forecasts
        aligned_forecasts = {}
        for model_name, forecast in filtered_forecasts.items():
            aligned = forecast.reindex(all_dates, method='ffill').fillna(0)
            aligned_forecasts[model_name] = aligned
        
        # Create ensemble
        if method == 'average':
            ensemble = pd.Series(
                np.mean([f.values for f in aligned_forecasts.values()], axis=0),
                index=all_dates
            )
        elif method == 'weighted_average':
            weighted_sum = np.zeros(len(all_dates))
            total_weight = 0
            
            for model_name, forecast in aligned_forecasts.items():
                weight = weights.get(model_name, 1.0 / len(filtered_forecasts))
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
        
        logger.info(
            f"Created ensemble with {len(filtered_forecasts)} models: {', '.join(filtered_forecasts.keys())}"
        )
        
        return ensemble, metadata

