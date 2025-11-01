"""
Step 2: Feature Engineering

Creates date-based, lag, rolling window, and categorical features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle

try:
    from . import config
    from .utils import (
        create_date_features,
        create_peak_season_feature,
        create_lag_features,
        create_rolling_features,
        label_encode_categorical
    )
except ImportError:
    import config
    from utils import (
        create_date_features,
        create_peak_season_feature,
        create_lag_features,
        create_rolling_features,
        label_encode_categorical
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_date_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create date-based features."""
    logger.info("Creating date-based features")
    
    df = create_date_features(df, config.DATE_COL)
    df = create_peak_season_feature(df, config.PEAK_SEASON_MONTHS)
    
    logger.info("Date features created: week_of_year, month, year, quarter, day_of_week, is_peak_season")
    return df


def create_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag and rolling window features."""
    logger.info("Creating lag features")
    
    # Group columns for lag/rolling features (exclude date)
    group_cols = config.GROUP_BY_COLS[1:]  # Branch, Tonnage
    
    # Create lag features
    df = create_lag_features(
        df,
        value_col=config.TARGET_COL,
        group_cols=group_cols,
        lag_periods=config.LAG_PERIODS,
        date_col=config.DATE_COL
    )
    
    logger.info(f"Lag features created: {[f'lag_{p}_weeks' for p in config.LAG_PERIODS]}")
    
    logger.info("Creating rolling window features")
    df = create_rolling_features(
        df,
        value_col=config.TARGET_COL,
        group_cols=group_cols,
        window=config.ROLLING_WINDOW,
        stats=config.ROLLING_STATS,
        date_col=config.DATE_COL
    )
    
    logger.info(f"Rolling features created: {[f'rolling_{s}_{config.ROLLING_WINDOW}_weeks' for s in config.ROLLING_STATS]}")
    
    return df


def encode_categorical_features(df: pd.DataFrame, save_mappings: bool = True) -> tuple:
    """
    Encode categorical features using label encoding.
    
    Returns:
        Encoded dataframe and mappings dictionary
    """
    logger.info("Encoding categorical features")
    
    df_encoded, mappings = label_encode_categorical(df, config.CATEGORICAL_COLS)
    
    # Log encoding info
    for col in config.CATEGORICAL_COLS:
        if col in df_encoded.columns:
            encoded_col = f"{col}_encoded"
            logger.info(f"{col}: {df[col].nunique()} unique values -> {df_encoded[encoded_col].nunique()} categories")
    
    # Save mappings if requested
    if save_mappings:
        mappings_path = config.OUTPUT_DIR / "categorical_mappings.pkl"
        with open(mappings_path, "wb") as f:
            pickle.dump(mappings, f)
        logger.info(f"Saved categorical mappings to {mappings_path}")
    
    return df_encoded, mappings


def get_feature_columns(df: pd.DataFrame, exclude_cols: list = None) -> list:
    """
    Get list of feature columns for model training.
    
    Excludes target, date, and original categorical columns (keeps encoded versions).
    """
    if exclude_cols is None:
        exclude_cols = [config.DATE_COL, config.TARGET_COL] + config.CATEGORICAL_COLS
    
    # Keep encoded categorical columns
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols 
                   and not col.endswith("_mapping")]
    
    return sorted(feature_cols)


def main():
    """Main function to run Step 2."""
    logger.info("=" * 50)
    logger.info("STEP 2: Feature Engineering")
    logger.info("=" * 50)
    
    # Load aggregated data
    logger.info(f"Loading aggregated data from {config.AGGREGATED_DATA_FILE}")
    df = pd.read_parquet(config.AGGREGATED_DATA_FILE)
    logger.info(f"Loaded {len(df)} rows")
    
    # Create date-based features
    df = create_date_based_features(df)
    
    # Create lag and rolling features
    df = create_lag_and_rolling_features(df)
    
    # Encode categorical features
    df, mappings = encode_categorical_features(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    logger.info(f"Total features created: {len(feature_cols)}")
    logger.info(f"Feature columns: {', '.join(feature_cols[:10])}..." if len(feature_cols) > 10 else f"Feature columns: {', '.join(feature_cols)}")
    
    # Save featured data
    logger.info(f"Saving featured data to {config.FEATURED_DATA_FILE}")
    df.to_parquet(config.FEATURED_DATA_FILE, index=False)
    
    # Save feature list
    feature_list_path = config.OUTPUT_DIR / "feature_list.txt"
    with open(feature_list_path, "w") as f:
        f.write("\n".join(feature_cols))
    logger.info(f"Saved feature list to {feature_list_path}")
    
    logger.info("Step 2 completed successfully!")
    return df, feature_cols


if __name__ == "__main__":
    main()

