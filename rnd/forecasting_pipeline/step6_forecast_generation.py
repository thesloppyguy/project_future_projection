"""
Step 6: Forecast Generation

Generates 52-week ahead forecasts using recursive autoregressive approach.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
from datetime import timedelta

import lightgbm as lgb

try:
    from . import config
    from .utils import (
        create_date_features,
        create_peak_season_feature,
        label_encode_categorical
    )
except ImportError:
    import config
    from utils import (
        create_date_features,
        create_peak_season_feature,
        label_encode_categorical
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_mappings(model_path: Path):
    """Load trained model and categorical mappings."""
    logger.info(f"Loading model from {model_path}")
    
    with open(model_path, "rb") as pickle_file:
        model = pickle.load(pickle_file)
    
    # Load categorical mappings
    mappings_path = config.OUTPUT_DIR / "categorical_mappings.pkl"
    if mappings_path.exists():
        with open(mappings_path, "rb") as f:
            mappings = pickle.load(f)
    else:
        mappings = {}
        logger.warning("Categorical mappings not found. Will infer from data.")
    
    # Load feature list
    feature_list_path = config.OUTPUT_DIR / "feature_list.txt"
    if feature_list_path.exists():
        with open(feature_list_path, "r") as f:
            feature_cols = [line.strip() for line in f if line.strip()]
    else:
        feature_cols = None
        logger.warning("Feature list not found. Will infer from model.")
    
    return model, mappings, feature_cols


def create_future_scaffold(
    historical_df: pd.DataFrame,
    forecast_start_date: pd.Timestamp,
    horizon_weeks: int = 52,
    required_cols: list = None
) -> pd.DataFrame:
    """
    Create scaffold dataframe for future forecasts.
    
    Creates all combinations of Branch and Tonnage for future weeks.
    If model expects Tonnage but current config excludes it, we need to add it.
    """
    logger.info(f"Creating future scaffold: {horizon_weeks} weeks from {forecast_start_date}")
    
    # Get unique combinations based on current config
    group_cols = list(config.GROUP_BY_COLS[1:])  # Branch, Tonnage (or just Branch)
    
    # Check if we need additional columns that model expects but config doesn't have
    if required_cols:
        # Check for encoded columns in required_cols to infer what categoricals are needed
        for encoded_col in required_cols:
            if "_encoded" in encoded_col:
                original_col = encoded_col.replace("_encoded", "")
                if original_col not in group_cols:
                    # Model expects this column - add it even if config excludes it
                    if original_col in historical_df.columns:
                        logger.info(f"Model expects {original_col} (for {encoded_col}) but not in current config. Adding it to scaffold.")
                        group_cols = group_cols + [original_col]
                    else:
                        logger.warning(f"Model expects {original_col} but it's not in historical data. Will use default encoding.")
    
    # Get unique combinations
    available_cols = [col for col in group_cols if col in historical_df.columns]
    if not available_cols:
        available_cols = group_cols  # Use config columns even if not in historical
    
    unique_combos = historical_df[available_cols].drop_duplicates()
    
    # If we need columns that aren't in historical (shouldn't happen, but handle it)
    for col in group_cols:
        if col not in unique_combos.columns:
            logger.warning(f"Column {col} not in historical data. Creating default values.")
            unique_combos[col] = unique_combos.iloc[0][available_cols[0]] if len(unique_combos) > 0 else "Unknown"
    
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
            for col in group_cols:
                if col in combo:
                    row[col] = combo[col]
                else:
                    # Default value if column doesn't exist
                    row[col] = unique_combos[col].iloc[0] if len(unique_combos) > 0 and col in unique_combos.columns else "Unknown"
            future_rows.append(row)
    
    future_df = pd.DataFrame(future_rows)
    
    logger.info(f"Created scaffold with {len(future_df)} rows")
    return future_df


def encode_future_categoricals(future_df: pd.DataFrame, mappings: dict, feature_cols: list = None) -> pd.DataFrame:
    """Encode categorical features for future dataframe."""
    future_df = future_df.copy()
    
    # Get all encoded columns that might be expected from feature_cols
    expected_encoded_cols = set()
    if feature_cols:
        expected_encoded_cols = set([col for col in feature_cols if "_encoded" in col])
    
    # Encode categoricals from current config
    for col in config.CATEGORICAL_COLS:
        encoded_col = f"{col}_encoded"
        if col in mappings:
            # Map using saved mappings
            future_df[encoded_col] = future_df[col].map(mappings[col])
            # Handle new categories not in training (shouldn't happen, but safe)
            future_df[encoded_col] = future_df[encoded_col].fillna(0)
            future_df[encoded_col] = future_df[encoded_col].astype("category")
        else:
            logger.warning(f"No mapping found for {col}. Creating new encoding.")
            # Fallback: create new mapping
            unique_vals = sorted(future_df[col].dropna().unique())
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            future_df[encoded_col] = future_df[col].map(mapping).astype("category")
    
    # Check if we need to add encoded columns that were in training but not in current config
    # This can happen if model was trained with Tonnage but we're forecasting without it
    for encoded_col in expected_encoded_cols:
        if encoded_col not in future_df.columns:
            original_col = encoded_col.replace("_encoded", "")
            if original_col not in config.CATEGORICAL_COLS:
                logger.warning(
                    f"Model expects {encoded_col} but current config excludes {original_col}. "
                    f"Trying to encode it from {original_col} column..."
                )
                # First check if the original column exists (it should if scaffold was created correctly)
                if original_col in future_df.columns:
                    # We have the original column, create mapping
                    if original_col in mappings:
                        future_df[encoded_col] = future_df[original_col].map(mappings[original_col])
                        future_df[encoded_col] = future_df[encoded_col].fillna(0)
                        future_df[encoded_col] = future_df[encoded_col].astype("category")
                        logger.info(f"Successfully encoded {encoded_col} from {original_col}")
                    else:
                        # No mapping available, use default
                        logger.warning(f"No mapping for {original_col}, using default value 0 for {encoded_col}")
                        future_df[encoded_col] = 0
                else:
                    # Original column doesn't exist, use default
                    logger.warning(
                        f"Original column {original_col} not found in future_df. "
                        f"Adding {encoded_col} with default value 0."
                    )
                    future_df[encoded_col] = 0
    
    return future_df


def get_latest_historical_values(
    historical_df: pd.DataFrame,
    group_cols: list,
    value_col: str,
    date_col: str
) -> dict:
    """Get the latest values for each group for lag feature initialization."""
    historical_df = historical_df.copy()
    historical_df = historical_df.sort_values(group_cols + [date_col])
    
    latest_values = {}
    for group, group_df in historical_df.groupby(group_cols):
        latest = group_df[value_col].tail(52).values.tolist()  # Keep last 52 weeks for lag_52
        latest_values[group] = {
            "values": latest,
            "max_date": group_df[date_col].max()
        }
    
    return latest_values


def create_lag_features_for_forecast(
    forecast_row: pd.Series,
    historical_values: list,
    lag_periods: list
) -> dict:
    """Create lag features for a single forecast row."""
    lag_features = {}
    
    # historical_values should be a list of recent values (most recent last)
    if len(historical_values) == 0:
        # No history, use zeros
        for lag in lag_periods:
            lag_features[f"lag_{lag}_weeks"] = 0.0
        return lag_features
    
    # Extract lag values
    for lag in lag_periods:
        if lag <= len(historical_values):
            lag_features[f"lag_{lag}_weeks"] = historical_values[-lag]
        else:
            lag_features[f"lag_{lag}_weeks"] = 0.0
    
    return lag_features


def create_rolling_features_for_forecast(
    historical_values: list,
    window: int,
    stats: list
) -> dict:
    """Create rolling window features for a single forecast row."""
    rolling_features = {}
    
    if len(historical_values) == 0:
        for stat in stats:
            rolling_features[f"rolling_{stat}_{window}_weeks"] = 0.0
        return rolling_features
    
    # Use recent values for rolling calculations
    recent_values = historical_values[-window:] if len(historical_values) >= window else historical_values
    
    if len(recent_values) == 0:
        for stat in stats:
            rolling_features[f"rolling_{stat}_{window}_weeks"] = 0.0
    else:
        recent_array = np.array(recent_values)
        for stat in stats:
            if stat == "mean":
                rolling_features[f"rolling_{stat}_{window}_weeks"] = recent_array.mean()
            elif stat == "std":
                rolling_features[f"rolling_{stat}_{window}_weeks"] = recent_array.std() if len(recent_array) > 1 else 0.0
            elif stat == "max":
                rolling_features[f"rolling_{stat}_{window}_weeks"] = recent_array.max()
    
    return rolling_features


def generate_forecast(
    model: lgb.Booster,
    historical_df: pd.DataFrame,
    feature_cols: list,
    mappings: dict,
    forecast_start_date: pd.Timestamp = None,
    horizon_weeks: int = 52
) -> pd.DataFrame:
    """
    Generate forecasts using recursive autoregressive approach.
    """
    logger.info("Generating forecasts using recursive autoregressive method")
    
    # Determine forecast start date
    if forecast_start_date is None:
        if config.FORECAST_START_DATE:
            forecast_start_date = pd.to_datetime(config.FORECAST_START_DATE)
        else:
            # Use latest date in historical data
            historical_df[config.DATE_COL] = pd.to_datetime(historical_df[config.DATE_COL])
            forecast_start_date = historical_df[config.DATE_COL].max() + timedelta(weeks=1)
            # Align to Monday
            forecast_start_date = forecast_start_date + timedelta(days=(7 - forecast_start_date.weekday()))
    
    logger.info(f"Forecast start date: {forecast_start_date}")
    
    # Create future scaffold - pass feature_cols so it knows if Tonnage is needed
    future_df = create_future_scaffold(historical_df, forecast_start_date, horizon_weeks, feature_cols)
    
    # Encode categoricals - pass feature_cols to ensure all expected encoded columns are created
    future_df = encode_future_categoricals(future_df, mappings, feature_cols)
    
    # Create date features
    future_df = create_date_features(future_df, config.DATE_COL)
    future_df = create_peak_season_feature(future_df, config.PEAK_SEASON_MONTHS)
    
    # Get group columns - use what's in future_df (may include Tonnage even if config excludes it)
    # This ensures we group correctly based on what the scaffold contains
    scaffold_group_cols = [col for col in future_df.columns 
                          if col not in [config.DATE_COL] and col not in [f"{c}_encoded" for c in config.CATEGORICAL_COLS]]
    # Filter to only include known grouping columns (Branch, Tonnage)
    known_group_cols = ["Branch", "Tonnage"]
    group_cols = [col for col in scaffold_group_cols if col in known_group_cols]
    
    # Fallback to config if nothing found
    if not group_cols:
        group_cols = config.GROUP_BY_COLS[1:]
    
    logger.debug(f"Using group columns for lag features: {group_cols}")
    
    # Get latest historical values for each group
    # Only use columns that exist in historical data
    hist_group_cols = [col for col in group_cols if col in historical_df.columns]
    historical_values_dict = get_latest_historical_values(
        historical_df, hist_group_cols, config.TARGET_COL, config.DATE_COL
    )
    
    # Track predictions for each group (for lag features)
    group_predictions = {}
    for group, info in historical_values_dict.items():
        group_predictions[group] = info["values"].copy()
    
    # Sort future dataframe by date and group
    sort_cols = [config.DATE_COL] + [col for col in group_cols if col in future_df.columns]
    future_df = future_df.sort_values(sort_cols).reset_index(drop=True)
    
    # Generate forecasts week by week
    logger.info("Generating forecasts iteratively...")
    forecasts = []
    
    unique_dates = sorted(future_df[config.DATE_COL].unique())
    
    for week_idx, forecast_date in enumerate(unique_dates):
        if (week_idx + 1) % 10 == 0:
            logger.info(f"  Forecasted {week_idx + 1}/{len(unique_dates)} weeks")
        
        # Get rows for this week
        week_df = future_df[future_df[config.DATE_COL] == forecast_date].copy()
        
        # Prepare features for this week
        week_features = []
        
        for _, row in week_df.iterrows():
            group = tuple(row[col] for col in group_cols)
            
            # Get historical values for this group
            if group in group_predictions:
                hist_values = group_predictions[group]
            else:
                hist_values = []
            
            # Create lag features
            lag_features = create_lag_features_for_forecast(
                row, hist_values, config.LAG_PERIODS
            )
            
            # Create rolling features
            rolling_features = create_rolling_features_for_forecast(
                hist_values, config.ROLLING_WINDOW, config.ROLLING_STATS
            )
            
            # Combine all features - include encoded categorical columns but exclude original categorical columns
            # Exclude DATE_COL and all categorical columns (even if they're not in current config)
            # This handles cases where model expects Tonnage but config excludes it
            exclude_cols = [config.DATE_COL] + ["Branch", "Tonnage"]  # Always exclude original categoricals
            feature_dict = {
                **row[[col for col in week_df.columns if col not in exclude_cols]].to_dict(),
                **lag_features,
                **rolling_features
            }
            
            week_features.append(feature_dict)
        
        # Convert to dataframe
        week_features_df = pd.DataFrame(week_features)
        
        # Debug: Log feature mismatch if any
        missing_features = set(feature_cols) - set(week_features_df.columns)
        extra_features = set(week_features_df.columns) - set(feature_cols)
        
        if missing_features:
            logger.warning(f"Missing features in forecast data: {missing_features}")
            logger.info(f"Expected features ({len(feature_cols)}): {feature_cols}")
            logger.info(f"Available features ({len(week_features_df.columns)}): {list(week_features_df.columns)}")
        if extra_features:
            logger.warning(f"Extra features in forecast data (will be ignored): {extra_features}")
        
        # Ensure all feature columns are present and in correct order
        feature_data = []
        for _, row in week_features_df.iterrows():
            feature_row = []
            for col in feature_cols:
                if col in row:
                    feature_row.append(row[col])
                elif col in week_features_df.columns:
                    # Column exists but not in this row
                    feature_row.append(week_features_df[col].iloc[0] if len(week_features_df) > 0 else 0)
                else:
                    # Missing feature, use default
                    if "lag" in col or "rolling" in col:
                        feature_row.append(0.0)
                    elif "_encoded" in col:
                        # Missing encoded feature - might be due to exclude_tonnage mismatch
                        logger.warning(f"Missing encoded feature: {col}. Using 0 as default.")
                        feature_row.append(0)
                    else:
                        feature_row.append(0)
            feature_data.append(feature_row)
        
        # Verify feature count matches BEFORE creating array
        if len(feature_data) == 0:
            raise ValueError("No feature data generated for this week")
        
        if len(feature_data[0]) != len(feature_cols):
            error_msg = (
                f"Feature count mismatch: Expected {len(feature_cols)} features, "
                f"got {len(feature_data[0])}. "
                f"\nModel expects: {feature_cols} "
                f"\nForecast generated: {list(week_features_df.columns)} "
                f"\nMissing: {missing_features}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Make predictions
        X_week = np.array(feature_data)
        
        # Final verification of array shape
        if X_week.shape[1] != len(feature_cols):
            raise ValueError(
                f"Array shape mismatch: Expected {len(feature_cols)} features, "
                f"got array with shape {X_week.shape} (features={X_week.shape[1]})"
            )
        
        logger.debug(f"Predicting with array shape: {X_week.shape}, expected {len(feature_cols)} features")
        predictions = model.predict(X_week)
        
        # Update group_predictions with new forecasts
        for idx, (_, row) in enumerate(week_df.iterrows()):
            group = tuple(row[col] for col in group_cols)
            pred_value = max(0, predictions[idx])  # Ensure non-negative
            
            if group not in group_predictions:
                group_predictions[group] = []
            
            group_predictions[group].append(pred_value)
            # Keep only last 52 weeks for lag_52
            if len(group_predictions[group]) > 52:
                group_predictions[group] = group_predictions[group][-52:]
            
            # Store forecast
            forecasts.append({
                config.DATE_COL: forecast_date,
                **{col: row[col] for col in group_cols},
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
    
    return forecast_df


def main():
    """Main function to run Step 6."""
    logger.info("=" * 50)
    logger.info("STEP 6: Forecast Generation")
    logger.info("=" * 50)
    
    # Load cleaned historical data
    logger.info(f"Loading historical data from {config.CLEANED_DATA_FILE}")
    historical_df = pd.read_parquet(config.CLEANED_DATA_FILE)
    logger.info(f"Loaded {len(historical_df)} historical rows")
    
    # Load model
    model_path = config.MODEL_DIR / "latest_final_model.pkl"
    if not model_path.exists():
        # Try alternative path
        model_path = config.MODEL_DIR / "latest_model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found. Please run Step 5 first.")
    
    model, mappings, feature_cols = load_model_and_mappings(model_path)
    
    # Always get feature names from the model itself (most reliable)
    # The model knows exactly how many features it was trained with
    model_feature_names = model.feature_name()
    
    # Use model's feature names if they differ from file
    if feature_cols is None or len(feature_cols) != len(model_feature_names):
        logger.warning(
            f"Feature list mismatch: file has {len(feature_cols) if feature_cols else 0} features, "
            f"model expects {len(model_feature_names)}. Using model's feature names."
        )
        feature_cols = model_feature_names
    elif set(feature_cols) != set(model_feature_names):
        logger.warning(
            f"Feature names differ between file and model. Using model's feature names."
        )
        feature_cols = model_feature_names
    
    logger.info(f"Model loaded with {len(feature_cols)} features (verified from model)")
    logger.info(f"Model expects these features: {feature_cols}")
    
    # Check if current config matches model training config
    expected_cat_cols = set([col.replace("_encoded", "") for col in feature_cols if "_encoded" in col])
    current_cat_cols = set(config.CATEGORICAL_COLS)
    
    if expected_cat_cols != current_cat_cols:
        logger.warning(
            f"Config mismatch detected! "
            f"Model was trained with categoricals: {expected_cat_cols}, "
            f"but current config has: {current_cat_cols}. "
            f"This may cause feature mismatch errors."
        )
    
    # Generate forecasts
    forecast_df = generate_forecast(
        model,
        historical_df,
        feature_cols,
        mappings,
        horizon_weeks=config.FORECAST_HORIZON_WEEKS
    )
    
    # Save forecasts
    logger.info(f"Saving forecasts to {config.FORECAST_FILE}")
    forecast_df.to_csv(config.FORECAST_FILE, index=False)
    
    logger.info("Step 6 completed successfully!")
    return forecast_df


if __name__ == "__main__":
    main()

