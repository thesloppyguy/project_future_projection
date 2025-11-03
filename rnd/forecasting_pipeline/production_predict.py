"""
Production Prediction Script

This script is designed to be used in production to generate forecasts
for the next 12 months (or configurable horizon).

Usage:
    python -m forecasting_pipeline.production_predict --horizon 52
    python -m forecasting_pipeline.production_predict --model-path models/latest_final_model.pkl
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import json
from typing import Optional

try:
    from . import config
    from .utils import (
        create_date_features, create_peak_season_feature,
        label_encode_categorical
    )
except ImportError:
    import config
    from utils import (
        create_date_features, create_peak_season_feature,
        label_encode_categorical
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionForecaster:
    """
    Production-ready forecaster that generates predictions for future dates.
    """
    
    def __init__(self, model_path: Path, historical_data_path: Optional[Path] = None):
        """
        Initialize the production forecaster.
        
        Args:
            model_path: Path to the trained model pickle file
            historical_data_path: Path to historical data (for lag/rolling features)
                                 If None, uses config.CLEANED_DATA_FILE
        """
        self.model_path = model_path
        self.historical_data_path = historical_data_path or config.CLEANED_DATA_FILE
        
        # Load model and metadata
        self.model, self.mappings, self.feature_cols = self._load_model()
        
        # Load historical data
        logger.info(f"Loading historical data from {self.historical_data_path}")
        self.historical_df = pd.read_parquet(self.historical_data_path)
        self.historical_df[config.DATE_COL] = pd.to_datetime(self.historical_df[config.DATE_COL])
        
        # Determine actual group columns based on what's in the data
        self.group_cols = []
        for col in config.GROUP_BY_COLS[1:]:
            if col in self.historical_df.columns:
                self.group_cols.append(col)
            else:
                logger.warning(f"Column {col} not found in historical data, skipping")
        
        # Sort historical data by date
        self.historical_df = self.historical_df.sort_values([config.DATE_COL] + self.group_cols)
        
        logger.info(f"Loaded {len(self.historical_df)} historical rows")
        logger.info(f"Historical date range: {self.historical_df[config.DATE_COL].min()} to {self.historical_df[config.DATE_COL].max()}")
    
    def _load_model(self):
        """Load the trained model and its metadata."""
        logger.info(f"Loading model from {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Load model
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Handle different model storage formats
        if isinstance(model_data, dict):
            model = model_data.get('model')
            mappings = model_data.get('mappings', {})
            feature_cols = model_data.get('feature_cols', model.feature_name() if hasattr(model, 'feature_name') else None)
        else:
            model = model_data
            mappings = {}
            feature_cols = None
        
        # Try to load mappings from separate file
        mappings_path = self.model_path.parent / f"{self.model_path.stem}_mappings.json"
        if mappings_path.exists():
            with open(mappings_path, 'r') as f:
                mappings = json.load(f)
        
        # Try to load feature list
        feature_list_path = self.model_path.parent / f"{self.model_path.stem}_features.txt"
        if not feature_cols and feature_list_path.exists():
            with open(feature_list_path, 'r') as f:
                feature_cols = [line.strip() for line in f if line.strip()]
        
        # Get feature names from model if available
        if not feature_cols and hasattr(model, 'feature_name'):
            feature_cols = model.feature_name()
        
        if not feature_cols:
            raise ValueError("Could not determine feature columns. Please ensure feature_list.txt exists.")
        
        logger.info(f"Model loaded with {len(feature_cols)} features")
        return model, mappings, feature_cols
    
    def _get_latest_historical_values(self, group_cols: list) -> dict:
        """
        Get the latest historical values for each group (for lag features).
        
        Returns:
            Dictionary mapping group tuples to lists of recent values
        """
        latest_values = {}
        
        for group, group_df in self.historical_df.groupby(group_cols):
            # Get last 52 weeks of quantities for lag_52
            values = group_df[config.TARGET_COL].tail(52).values.tolist()
            latest_values[group] = values
        
        return latest_values
    
    def _create_lag_features(self, row: pd.Series, historical_values: list) -> dict:
        """Create lag features for a single row."""
        lag_features = {}
        
        if len(historical_values) == 0:
            for lag in config.LAG_PERIODS:
                lag_features[f"lag_{lag}_weeks"] = 0.0
            return lag_features
        
        for lag in config.LAG_PERIODS:
            if lag <= len(historical_values):
                lag_features[f"lag_{lag}_weeks"] = historical_values[-lag]
            else:
                lag_features[f"lag_{lag}_weeks"] = 0.0
        
        return lag_features
    
    def _create_rolling_features(self, historical_values: list) -> dict:
        """Create rolling window features for a single row."""
        rolling_features = {}
        
        if len(historical_values) == 0:
            for stat in config.ROLLING_STATS:
                rolling_features[f"rolling_{stat}_{config.ROLLING_WINDOW}_weeks"] = 0.0
            return rolling_features
        
        # Use recent values for rolling calculations
        window = config.ROLLING_WINDOW
        recent_values = historical_values[-window:] if len(historical_values) >= window else historical_values
        
        if len(recent_values) == 0:
            for stat in config.ROLLING_STATS:
                rolling_features[f"rolling_{stat}_{window}_weeks"] = 0.0
        else:
            recent_array = np.array(recent_values)
            for stat in config.ROLLING_STATS:
                if stat == "mean":
                    rolling_features[f"rolling_{stat}_{window}_weeks"] = recent_array.mean()
                elif stat == "std":
                    rolling_features[f"rolling_{stat}_{window}_weeks"] = recent_array.std() if len(recent_array) > 1 else 0.0
                elif stat == "max":
                    rolling_features[f"rolling_{stat}_{window}_weeks"] = recent_array.max()
        
        return rolling_features
    
    def _create_future_scaffold(self, forecast_start_date: pd.Timestamp, horizon_weeks: int) -> pd.DataFrame:
        """Create scaffold dataframe for future dates."""
        logger.info(f"Creating future scaffold: {horizon_weeks} weeks from {forecast_start_date}")
        
        # Get unique combinations using actual group columns
        unique_combos = self.historical_df[self.group_cols].drop_duplicates()
        
        logger.info(f"Found {len(unique_combos)} unique combinations")
        
        # Create date range (weekly, starting Monday)
        date_range = pd.date_range(
            start=forecast_start_date,
            periods=horizon_weeks,
            freq=config.AGGREGATION_FREQ
        )
        
        # Create all combinations
        future_rows = []
        for date in date_range:
            for _, combo in unique_combos.iterrows():
                row = {config.DATE_COL: date}
                for col in self.group_cols:
                    row[col] = combo[col]
                future_rows.append(row)
        
        future_df = pd.DataFrame(future_rows)
        logger.info(f"Created scaffold with {len(future_df)} rows")
        
        return future_df
    
    def predict(self, horizon_weeks: int = 52, forecast_start_date: Optional[pd.Timestamp] = None, save_features: bool = False):
        """
        Generate predictions for the specified horizon using iterative autoregressive method.
        
        Args:
            horizon_weeks: Number of weeks to forecast (default: 52)
            forecast_start_date: Start date for forecast. If None, uses next week after latest historical data
            save_features: If True, also returns feature dataframe
        
        Returns:
            Tuple of (forecast_df, features_df) if save_features=True, else just forecast_df
            forecast_df: DataFrame with forecasts including all group columns and 'forecast' column
        """
        # Determine forecast start date
        if forecast_start_date is None:
            latest_date = self.historical_df[config.DATE_COL].max()
            forecast_start_date = latest_date + timedelta(weeks=1)
            # Align to Monday
            forecast_start_date = forecast_start_date + timedelta(days=(7 - forecast_start_date.weekday()))
        
        logger.info(f"Forecast start date: {forecast_start_date}")
        logger.info(f"Forecast horizon: {horizon_weeks} weeks ({horizon_weeks / 52:.1f} months)")
        
        # Create future scaffold
        future_df = self._create_future_scaffold(forecast_start_date, horizon_weeks)
        
        # 1. Date features
        future_df = create_date_features(future_df, config.DATE_COL)
        
        # 2. Peak season feature
        future_df = create_peak_season_feature(future_df, config.PEAK_SEASON_MONTHS)
        
        # 3. Categorical encoding - only encode columns that exist in the data
        for col in config.CATEGORICAL_COLS:
            if col not in future_df.columns:
                logger.warning(f"Skipping encoding for {col} - column not in future dataframe")
                continue
            
            encoded_col = f"{col}_encoded"
            if col in self.mappings:
                future_df[encoded_col] = future_df[col].map(self.mappings[col])
                future_df[encoded_col] = future_df[encoded_col].fillna(0).astype("category")
            else:
                logger.warning(f"No mapping found for {col}. Using default encoding.")
                unique_vals = sorted(future_df[col].dropna().unique())
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                future_df[encoded_col] = future_df[col].map(mapping).astype("category")
        
        # Check if model expects encoded columns that we haven't created yet
        # This handles cases where model was trained with Tonnage but current data doesn't have it
        expected_encoded_cols = set([col for col in self.feature_cols if "_encoded" in col])
        created_encoded_cols = set([col for col in future_df.columns if "_encoded" in col])
        missing_encoded_cols = expected_encoded_cols - created_encoded_cols
        
        for encoded_col in missing_encoded_cols:
            original_col = encoded_col.replace("_encoded", "")
            logger.warning(
                f"Model expects {encoded_col} but {original_col} not in data. "
                f"Adding {encoded_col} with default value 0."
            )
            future_df[encoded_col] = 0
        
        # 4. Get historical values for lag/rolling features
        historical_values_dict = self._get_latest_historical_values(self.group_cols)
        
        # 5. Initialize group_predictions with historical values
        group_predictions = {}
        for group, values in historical_values_dict.items():
            group_predictions[group] = values.copy()
        
        # 6. Sort by date and group
        future_df = future_df.sort_values([config.DATE_COL] + self.group_cols).reset_index(drop=True)
        
        # 7. Generate predictions iteratively week by week
        logger.info("Generating predictions iteratively...")
        forecasts = []
        all_features = []
        
        unique_dates = sorted(future_df[config.DATE_COL].unique())
        
        for week_idx, forecast_date in enumerate(unique_dates):
            if (week_idx + 1) % 10 == 0:
                logger.info(f"  Forecasted {week_idx + 1}/{len(unique_dates)} weeks")
            
            # Get rows for this week
            week_df = future_df[future_df[config.DATE_COL] == forecast_date].copy()
            
            # Prepare features for this week
            week_features = []
            
            for _, row in week_df.iterrows():
                group = tuple(row[col] for col in self.group_cols)
                
                # Get historical values for this group
                if group in group_predictions:
                    hist_values = group_predictions[group]
                else:
                    hist_values = []
                
                # Create lag features
                lag_features = self._create_lag_features(row, hist_values)
                
                # Create rolling features
                rolling_features = self._create_rolling_features(hist_values)
                
                # Combine all features
                exclude_cols = [config.DATE_COL] + ["Branch", "Tonnage"]
                feature_dict = {
                    **row[[col for col in row.index if col not in exclude_cols]].to_dict(),
                    **lag_features,
                    **rolling_features
                }
                
                week_features.append(feature_dict)
            
            # Convert to dataframe
            week_features_df = pd.DataFrame(week_features)
            
            # Ensure all feature columns are present and in correct order
            feature_data = []
            for _, row in week_features_df.iterrows():
                feature_vector = []
                for col in self.feature_cols:
                    if col in row:
                        feature_vector.append(row[col])
                    elif col in week_features_df.columns:
                        feature_vector.append(week_features_df[col].iloc[0] if len(week_features_df) > 0 else 0)
                    else:
                        feature_vector.append(0.0)
                feature_data.append(feature_vector)
            
            X_week = np.array(feature_data)
            
            # Make predictions
            predictions = self.model.predict(X_week)
            predictions = np.maximum(predictions, 0)  # Ensure non-negative
            
            # Update group_predictions with new forecasts
            for idx, (_, row) in enumerate(week_df.iterrows()):
                group = tuple(row[col] for col in self.group_cols)
                pred_value = predictions[idx]
                
                if group not in group_predictions:
                    group_predictions[group] = []
                
                group_predictions[group].append(pred_value)
                # Keep only last 52 weeks for lag_52
                if len(group_predictions[group]) > 52:
                    group_predictions[group] = group_predictions[group][-52:]
                
                # Store forecast
                forecasts.append({
                    config.DATE_COL: forecast_date,
                    **{col: row[col] for col in self.group_cols},
                    "forecast": pred_value
                })
                
                # Store features if requested
                if save_features:
                    all_features.append({
                        config.DATE_COL: forecast_date,
                        **{col: row[col] for col in self.group_cols},
                        **week_features_df.iloc[idx].to_dict(),
                        "forecast": pred_value
                    })
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame(forecasts)
        
        logger.info(f"Generated {len(forecast_df)} forecasts")
        logger.info(f"Forecast statistics:")
        logger.info(f"  Mean: {forecast_df['forecast'].mean():.2f}")
        logger.info(f"  Std: {forecast_df['forecast'].std():.2f}")
        logger.info(f"  Min: {forecast_df['forecast'].min():.2f}")
        logger.info(f"  Max: {forecast_df['forecast'].max():.2f}")
        
        if save_features:
            features_df = pd.DataFrame(all_features)
            return forecast_df, features_df
        else:
            return forecast_df


def main():
    """Main entry point for production predictions."""
    parser = argparse.ArgumentParser(
        description="Production forecasting script for HVAC demand",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Forecast next 12 months (default)
  python -m forecasting_pipeline.production_predict
  
  # Forecast next 6 months (26 weeks)
  python -m forecasting_pipeline.production_predict --horizon 26
  
  # Use specific model
  python -m forecasting_pipeline.production_predict --model-path models/branch_A_model.pkl
  
  # Specify output file
  python -m forecasting_pipeline.production_predict --output forecasts/production_forecast.csv
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=Path,
        default=config.MODEL_DIR / "latest_final_model.pkl",
        help=f'Path to trained model (default: {config.MODEL_DIR / "latest_final_model.pkl"})'
    )
    
    parser.add_argument(
        '--historical-data',
        type=Path,
        default=None,
        help=f'Path to historical data (default: {config.CLEANED_DATA_FILE})'
    )
    
    parser.add_argument(
        '--horizon',
        type=int,
        default=52,
        help='Forecast horizon in weeks (default: 52 = 12 months)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Forecast start date (YYYY-MM-DD). If not provided, uses next week after latest historical data'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output file path (default: outputs/production_forecast_YYYYMMDD.csv)'
    )
    
    parser.add_argument(
        '--save-features',
        action='store_true',
        help='Also save the generated features for debugging'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize forecaster
        forecaster = ProductionForecaster(
            model_path=args.model_path,
            historical_data_path=args.historical_data
        )
        
        # Parse start date if provided
        forecast_start = None
        if args.start_date:
            forecast_start = pd.to_datetime(args.start_date)
        
        # Generate predictions
        result = forecaster.predict(
            horizon_weeks=args.horizon,
            forecast_start_date=forecast_start,
            save_features=args.save_features
        )
        
        if args.save_features:
            forecast_df, features_df = result
        else:
            forecast_df = result
            features_df = None
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = config.OUTPUT_DIR / f"production_forecast_{timestamp}.csv"
        
        # Save forecast
        output_path.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Forecast saved to: {output_path}")
        
        # Save features if requested
        if args.save_features and features_df is not None:
            features_path = output_path.with_suffix('.features.parquet')
            features_df.to_parquet(features_path, index=False)
            logger.info(f"✓ Features saved to: {features_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("FORECAST SUMMARY")
        print("="*70)
        print(f"Total forecasts: {len(forecast_df)}")
        print(f"Date range: {forecast_df[config.DATE_COL].min()} to {forecast_df[config.DATE_COL].max()}")
        # Get group columns from forecaster
        group_cols_for_summary = [col for col in config.GROUP_BY_COLS[1:] if col in forecast_df.columns]
        if group_cols_for_summary:
            print(f"Unique combinations: {forecast_df.groupby(group_cols_for_summary).ngroups}")
        else:
            print(f"Total forecast rows: {len(forecast_df)}")
        print(f"\nForecast statistics:")
        print(f"  Mean: {forecast_df['forecast'].mean():.2f}")
        print(f"  Median: {forecast_df['forecast'].median():.2f}")
        print(f"  Std: {forecast_df['forecast'].std():.2f}")
        print(f"  Min: {forecast_df['forecast'].min():.2f}")
        print(f"  Max: {forecast_df['forecast'].max():.2f}")
        print(f"\nOutput file: {output_path}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Production prediction failed: {e}")
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()

