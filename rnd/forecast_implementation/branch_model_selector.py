"""
Branch-specific model selection based on calibration performance.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BranchModelSelector:
    """Select best models for each branch based on calibration performance."""
    
    def __init__(self, top_n: int = 3, min_models: int = 2):
        """
        Initialize branch model selector.
        
        Args:
            top_n: Number of top models to select per branch
            min_models: Minimum number of models to use
        """
        self.top_n = top_n
        self.min_models = min_models
    
    def select_models_for_branch(
        self,
        calibration_results: Dict,
        branch_key: str
    ) -> List[str]:
        """
        Select best models for a specific branch based on calibration performance.
        
        Args:
            calibration_results: Calibration results dictionary
            branch_key: Branch identifier (e.g., 'monthly_branch_wise_BLR')
            
        Returns:
            List of selected model names
        """
        if branch_key not in calibration_results:
            logger.warning(f"No calibration results for {branch_key}")
            return []
        
        branch_results = calibration_results[branch_key]
        model_scores = {}
        
        # Calculate average RMSE for each model
        for model_name, results in branch_results.items():
            if 'metrics' in results and len(results['metrics']) > 0:
                rmse_values = [
                    m.get('rmse', np.nan) for m in results['metrics']
                    if not np.isnan(m.get('rmse', np.nan))
                ]
                if len(rmse_values) > 0:
                    avg_rmse = np.mean(rmse_values)
                    model_scores[model_name] = avg_rmse
        
        if len(model_scores) == 0:
            logger.warning(f"No valid scores for {branch_key}")
            return []
        
        # Sort by RMSE (lower is better) and select top N
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1])
        selected_models = [name for name, _ in sorted_models[:self.top_n]]
        
        # Ensure minimum number of models
        if len(selected_models) < self.min_models and len(model_scores) >= self.min_models:
            selected_models = [name for name, _ in sorted_models[:self.min_models]]
        
        logger.info(
            f"Selected {len(selected_models)} models for {branch_key}: {', '.join(selected_models)}"
        )
        
        return selected_models
    
    def create_branch_model_mapping(
        self,
        calibration_results: Dict
    ) -> Dict[str, List[str]]:
        """
        Create mapping of branch keys to selected models.
        
        Args:
            calibration_results: Calibration results dictionary
            
        Returns:
            Dictionary mapping branch keys to selected model names
        """
        mapping = {}
        
        for branch_key in calibration_results.keys():
            if 'branch_wise' in branch_key:
                selected_models = self.select_models_for_branch(
                    calibration_results, branch_key
                )
                if len(selected_models) > 0:
                    mapping[branch_key] = selected_models
        
        return mapping
    
    def get_model_recommendations(
        self,
        calibration_results: Dict
    ) -> pd.DataFrame:
        """
        Get model recommendations for all branches.
        
        Args:
            calibration_results: Calibration results dictionary
            
        Returns:
            DataFrame with branch recommendations
        """
        recommendations = []
        
        for branch_key in calibration_results.keys():
            if 'branch_wise' in branch_key:
                selected_models = self.select_models_for_branch(
                    calibration_results, branch_key
                )
                
                # Get performance metrics
                branch_results = calibration_results[branch_key]
                for model_name in selected_models:
                    if model_name in branch_results:
                        results = branch_results[model_name]
                        if 'metrics' in results and len(results['metrics']) > 0:
                            rmse_values = [
                                m.get('rmse', np.nan) for m in results['metrics']
                                if not np.isnan(m.get('rmse', np.nan))
                            ]
                            mape_values = [
                                m.get('mape', np.nan) for m in results['metrics']
                                if not np.isnan(m.get('mape', np.nan))
                            ]
                            
                            recommendations.append({
                                'branch': branch_key,
                                'model': model_name,
                                'avg_rmse': np.mean(rmse_values) if len(rmse_values) > 0 else np.nan,
                                'avg_mape': np.mean(mape_values) if len(mape_values) > 0 else np.nan,
                                'rank': selected_models.index(model_name) + 1
                            })
        
        return pd.DataFrame(recommendations)

