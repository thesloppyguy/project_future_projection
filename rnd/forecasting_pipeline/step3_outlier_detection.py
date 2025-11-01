"""
Step 3: Outlier Detection & Treatment

Identifies outliers using Isolation Forest or LOF and caps them using Winsorization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

try:
    from . import config
except ImportError:
    import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_outliers_isolation_forest(
    values: np.ndarray,
    contamination: float = 0.05
) -> np.ndarray:
    """
    Detect outliers using Isolation Forest.
    
    Returns:
        Boolean array where True indicates an outlier
    """
    logger.info(f"Detecting outliers using Isolation Forest (contamination={contamination})")
    
    # Reshape for sklearn
    values_reshaped = values.reshape(-1, 1)
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=config.RANDOM_STATE
    )
    outlier_labels = iso_forest.fit_predict(values_reshaped)
    
    # Convert to boolean (outliers are -1, inliers are 1)
    is_outlier = outlier_labels == -1
    
    logger.info(f"Detected {is_outlier.sum()} outliers ({is_outlier.sum() / len(is_outlier) * 100:.2f}%)")
    return is_outlier


def detect_outliers_lof(
    values: np.ndarray,
    contamination: float = 0.05
) -> np.ndarray:
    """
    Detect outliers using Local Outlier Factor.
    
    Returns:
        Boolean array where True indicates an outlier
    """
    logger.info(f"Detecting outliers using Local Outlier Factor (contamination={contamination})")
    
    # Reshape for sklearn
    values_reshaped = values.reshape(-1, 1)
    
    # Fit LOF
    lof = LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=20
    )
    outlier_labels = lof.fit_predict(values_reshaped)
    
    # Convert to boolean (outliers are -1, inliers are 1)
    is_outlier = outlier_labels == -1
    
    logger.info(f"Detected {is_outlier.sum()} outliers ({is_outlier.sum() / len(is_outlier) * 100:.2f}%)")
    return is_outlier


def detect_outliers_percentile(
    values: np.ndarray,
    percentile: float = 99
) -> np.ndarray:
    """
    Detect outliers using percentile threshold.
    
    Returns:
        Boolean array where True indicates an outlier
    """
    logger.info(f"Detecting outliers using {percentile}th percentile")
    
    threshold = np.percentile(values, percentile)
    is_outlier = values > threshold
    
    logger.info(f"Detected {is_outlier.sum()} outliers above {threshold:.2f} ({is_outlier.sum() / len(is_outlier) * 100:.2f}%)")
    return is_outlier


def detect_outliers(df: pd.DataFrame, target_col: str = "Quantity") -> pd.DataFrame:
    """
    Detect outliers in the target column.
    
    Adds an 'is_outlier' column to the dataframe.
    """
    df = df.copy()
    values = df[target_col].values
    
    # Select method based on config
    if config.OUTLIER_METHOD == "isolation_forest":
        is_outlier = detect_outliers_isolation_forest(
            values,
            contamination=config.OUTLIER_CONTAMINATION
        )
    elif config.OUTLIER_METHOD == "lof":
        is_outlier = detect_outliers_lof(
            values,
            contamination=config.OUTLIER_CONTAMINATION
        )
    elif config.OUTLIER_METHOD == "percentile":
        is_outlier = detect_outliers_percentile(
            values,
            percentile=config.WINSORIZE_PERCENTILE
        )
    else:
        logger.warning(f"Unknown outlier method: {config.OUTLIER_METHOD}. Using percentile method.")
        is_outlier = detect_outliers_percentile(values, percentile=config.WINSORIZE_PERCENTILE)
    
    df["is_outlier"] = is_outlier
    return df


def winsorize_values(
    df: pd.DataFrame,
    target_col: str = "Quantity",
    percentile: float = 99
) -> pd.DataFrame:
    """
    Winsorize (cap) outlier values at specified percentile.
    
    This preserves the signal of high demand while preventing skewing.
    """
    df = df.copy()
    
    logger.info(f"Winsorizing values at {percentile}th percentile")
    
    # Calculate percentile threshold
    threshold = np.percentile(df[target_col], percentile)
    logger.info(f"99th percentile threshold: {threshold:.2f}")
    
    # Count values to be capped
    values_to_cap = (df[target_col] > threshold).sum()
    logger.info(f"Capping {values_to_cap} values above {threshold:.2f}")
    
    # Create capped column (preserve original)
    df[f"{target_col}_original"] = df[target_col].copy()
    df[target_col] = df[target_col].clip(upper=threshold)
    
    # Log statistics
    if values_to_cap > 0:
        logger.info(f"Maximum value before: {df[f'{target_col}_original'].max():.2f}")
        logger.info(f"Maximum value after: {df[target_col].max():.2f}")
    
    return df


def main():
    """Main function to run Step 3."""
    logger.info("=" * 50)
    logger.info("STEP 3: Outlier Detection & Treatment")
    logger.info("=" * 50)
    
    # Load featured data
    logger.info(f"Loading featured data from {config.FEATURED_DATA_FILE}")
    df = pd.read_parquet(config.FEATURED_DATA_FILE)
    logger.info(f"Loaded {len(df)} rows")
    
    # Log initial statistics
    logger.info(f"Initial statistics for {config.TARGET_COL}:")
    logger.info(f"  Mean: {df[config.TARGET_COL].mean():.2f}")
    logger.info(f"  Std: {df[config.TARGET_COL].std():.2f}")
    logger.info(f"  Min: {df[config.TARGET_COL].min():.2f}")
    logger.info(f"  Max: {df[config.TARGET_COL].max():.2f}")
    logger.info(f"  99th percentile: {df[config.TARGET_COL].quantile(0.99):.2f}")
    
    # Detect outliers
    df = detect_outliers(df, config.TARGET_COL)
    
    # Winsorize (cap) outliers
    df = winsorize_values(df, config.TARGET_COL, config.WINSORIZE_PERCENTILE)
    
    # Log final statistics
    logger.info(f"Final statistics for {config.TARGET_COL}:")
    logger.info(f"  Mean: {df[config.TARGET_COL].mean():.2f}")
    logger.info(f"  Std: {df[config.TARGET_COL].std():.2f}")
    logger.info(f"  Min: {df[config.TARGET_COL].min():.2f}")
    logger.info(f"  Max: {df[config.TARGET_COL].max():.2f}")
    
    # Save cleaned data
    logger.info(f"Saving cleaned data to {config.CLEANED_DATA_FILE}")
    df.to_parquet(config.CLEANED_DATA_FILE, index=False)
    
    logger.info("Step 3 completed successfully!")
    return df


if __name__ == "__main__":
    main()

