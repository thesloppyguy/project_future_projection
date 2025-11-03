"""
Step 1: Data Aggregation

Aggregates raw transactional data to weekly frequency and creates a complete panel dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

try:
    from . import config
    from .utils import ensure_quantity_non_negative, create_complete_panel
except ImportError:
    import config
    from utils import ensure_quantity_non_negative, create_complete_panel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Load raw data from CSV file."""
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows")
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data: convert dates, ensure non-negative quantities."""
    df = df.copy()
    
    # Convert Date to datetime
    logger.info("Converting Date to datetime")
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    
    # Ensure quantity is non-negative
    logger.info("Ensuring quantities are non-negative")
    df = ensure_quantity_non_negative(df, config.TARGET_COL)
    df = df[df[config.DATE_COL] >= '2021-09-01']
    # Remove rows with missing critical columns
    initial_rows = len(df)
    df = df.dropna(subset=[config.DATE_COL, config.TARGET_COL] + config.GROUP_BY_COLS[1:])
    if len(df) < initial_rows:
        logger.warning(f"Removed {initial_rows - len(df)} rows with missing values")
    
    return df


def aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data to weekly frequency.
    
    Groups by Date (resampled to weekly), Branch, and Tonnage,
    then sums the Quantity.
    """
    logger.info("Aggregating to weekly frequency")
    
    # First, resample dates to weekly (W-MON = weekly starting Monday)
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    
    # Group by the aggregation columns and sum quantity
    # We'll resample the date first
    df["Date_Week"] = df[config.DATE_COL].dt.to_period(config.AGGREGATION_FREQ).dt.to_timestamp()
    
    # Aggregate by week and grouping columns
    agg_df = df.groupby(
        ["Date_Week"] + config.GROUP_BY_COLS[1:],  # Date_Week, Branch, Tonnage
        dropna=False
    )[config.TARGET_COL].sum().reset_index()
    
    # Rename Date_Week back to Date
    agg_df = agg_df.rename(columns={"Date_Week": config.DATE_COL})
    
    logger.info(f"Aggregated to {len(agg_df)} rows")
    return agg_df


def create_panel_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create complete panel dataset by filling missing combinations with zeros.
    
    This is critical for the model to learn when demand is zero.
    """
    logger.info("Creating complete panel dataset")
    
    panel_df = create_complete_panel(
        df=df,
        date_col=config.DATE_COL,
        group_cols=config.GROUP_BY_COLS,
        value_col=config.TARGET_COL,
        freq=config.AGGREGATION_FREQ
    )
    
    logger.info(f"Complete panel has {len(panel_df)} rows")
    logger.info(f"Date range: {panel_df[config.DATE_COL].min()} to {panel_df[config.DATE_COL].max()}")
    
    # Log statistics
    group_cols = config.GROUP_BY_COLS[1:]  # Branch, Tonnage (or just Branch if EXCLUDE_TONNAGE=True)
    unique_combo_count = panel_df.groupby(group_cols).size().shape[0]
    
    if len(group_cols) == 2:
        logger.info(f"Unique combinations ({group_cols[0]}, {group_cols[1]}): {unique_combo_count}")
    else:
        logger.info(f"Unique {group_cols[0]}s: {unique_combo_count}")
    
    logger.info(f"Total weeks: {panel_df[config.DATE_COL].nunique()}")
    
    zero_count = (panel_df[config.TARGET_COL] == 0).sum()
    zero_pct = zero_count / len(panel_df) * 100
    logger.info(f"Zero quantity rows: {zero_count} ({zero_pct:.2f}%)")
    
    # Breakdown by categories
    logger.info("\nZero quantity breakdown by categories:")
    zero_breakdown = panel_df.groupby(group_cols, group_keys=False).apply(
        lambda x: pd.Series({
            'total_rows': len(x),
            'zero_rows': (x[config.TARGET_COL] == 0).sum(),
            'zero_pct': (x[config.TARGET_COL] == 0).sum() / len(x) * 100,
            'non_zero_rows': (x[config.TARGET_COL] > 0).sum(),
            'avg_quantity': x[config.TARGET_COL].mean(),
            'max_quantity': x[config.TARGET_COL].max()
        })
    ).reset_index()
    
    # Sort by zero percentage
    zero_breakdown = zero_breakdown.sort_values('zero_pct', ascending=False)
    
    # Format header and rows based on number of group columns
    if len(group_cols) == 2:
        logger.info(f"\n{group_cols[0]:<10} {group_cols[1]:<10} {'Total Rows':<12} {'Zero Rows':<12} {'Zero %':<10} {'Avg Qty':<12} {'Max Qty':<12}")
        logger.info("-" * 90)
        for _, row in zero_breakdown.iterrows():
            logger.info(f"{str(row[group_cols[0]]):<10} {str(row[group_cols[1]]):<10} "
                       f"{int(row['total_rows']):<12} {int(row['zero_rows']):<12} "
                       f"{row['zero_pct']:<10.2f} {row['avg_quantity']:<12.2f} {row['max_quantity']:<12.2f}")
        
        # Summary statistics
        logger.info("\nSummary:")
        max_idx = zero_breakdown['zero_pct'].idxmax()
        min_idx = zero_breakdown['zero_pct'].idxmin()
        logger.info(f"  Highest zero %: {zero_breakdown['zero_pct'].max():.2f}% ({zero_breakdown.loc[max_idx, group_cols[0]]}, {zero_breakdown.loc[max_idx, group_cols[1]]})")
        logger.info(f"  Lowest zero %: {zero_breakdown['zero_pct'].min():.2f}% ({zero_breakdown.loc[min_idx, group_cols[0]]}, {zero_breakdown.loc[min_idx, group_cols[1]]})")
    else:
        logger.info(f"\n{group_cols[0]:<15} {'Total Rows':<12} {'Zero Rows':<12} {'Zero %':<10} {'Avg Qty':<12} {'Max Qty':<12}")
        logger.info("-" * 80)
        for _, row in zero_breakdown.iterrows():
            logger.info(f"{str(row[group_cols[0]]):<15} "
                       f"{int(row['total_rows']):<12} {int(row['zero_rows']):<12} "
                       f"{row['zero_pct']:<10.2f} {row['avg_quantity']:<12.2f} {row['max_quantity']:<12.2f}")
        
        # Summary statistics
        logger.info("\nSummary:")
        max_idx = zero_breakdown['zero_pct'].idxmax()
        min_idx = zero_breakdown['zero_pct'].idxmin()
        logger.info(f"  Highest zero %: {zero_breakdown['zero_pct'].max():.2f}% ({zero_breakdown.loc[max_idx, group_cols[0]]})")
        logger.info(f"  Lowest zero %: {zero_breakdown['zero_pct'].min():.2f}% ({zero_breakdown.loc[min_idx, group_cols[0]]})")
    
    logger.info(f"  Mean zero % across combinations: {zero_breakdown['zero_pct'].mean():.2f}%")
    logger.info(f"  Median zero % across combinations: {zero_breakdown['zero_pct'].median():.2f}%")
    
    # Save breakdown to CSV
    breakdown_path = config.OUTPUT_DIR / "zero_quantity_breakdown.csv"
    zero_breakdown.to_csv(breakdown_path, index=False)
    logger.info(f"\nSaved zero quantity breakdown to {breakdown_path}")
    
    return panel_df


def save_aggregated_data(df: pd.DataFrame, output_path: Path):
    """Save aggregated data to parquet format."""
    logger.info(f"Saving aggregated data to {output_path}")
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")


def main():
    """Main function to run Step 1."""
    logger.info("=" * 50)
    logger.info("STEP 1: Data Aggregation")
    logger.info("=" * 50)
    
    # Load raw data
    df = load_raw_data(config.SOURCE_DATA_FILE)
    
    # Prepare data
    df = prepare_data(df)
    
    # Aggregate to weekly
    df_weekly = aggregate_to_weekly(df)
    
    # Create complete panel
    df_panel = create_panel_dataset(df_weekly)
    
    # Save aggregated data
    save_aggregated_data(df_panel, config.AGGREGATED_DATA_FILE)
    
    logger.info("Step 1 completed successfully!")
    return df_panel


if __name__ == "__main__":
    main()

