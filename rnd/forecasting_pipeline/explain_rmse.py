"""
Explain how RMSE is calculated in validation.

Shows that RMSE is aggregated across all combinations and weeks.
"""

import pandas as pd
import numpy as np
from pathlib import Path

try:
    from . import config
except ImportError:
    import config

def explain_rmse_calculation():
    """Explain how RMSE is calculated."""
    
    print("="*70)
    print("HOW RMSE IS CALCULATED IN VALIDATION")
    print("="*70)
    print()
    
    # Load cleaned data to understand structure
    if config.CLEANED_DATA_FILE.exists():
        df = pd.read_parquet(config.CLEANED_DATA_FILE)
        df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
        
        # Show structure
        print("Data Structure:")
        print(f"  Total rows: {len(df)}")
        print(f"  Unique combinations (Branch, Tonnage): {df.groupby(['Branch', 'Tonnage']).ngroups}")
        print(f"  Date range: {df[config.DATE_COL].min()} to {df[config.DATE_COL].max()}")
        print(f"  Total weeks: {df[config.DATE_COL].nunique()}")
        print()
        
        # Show example validation period
        val_start = pd.to_datetime("2023-01-01")
        val_end = pd.to_datetime("2023-12-31")
        val_mask = (df[config.DATE_COL] >= val_start) & (df[config.DATE_COL] <= val_end)
        val_df = df[val_mask]
        
        print(f"Example Validation Period (Fold 1): {val_start.date()} to {val_end.date()}")
        print(f"  Validation rows: {len(val_df)}")
        print(f"  Unique combinations: {val_df.groupby(['Branch', 'Tonnage']).ngroups}")
        print(f"  Weeks in validation: {val_df[config.DATE_COL].nunique()}")
        print(f"  Expected rows (combinations × weeks): {val_df.groupby(['Branch', 'Tonnage']).ngroups} × {val_df[config.DATE_COL].nunique()} = {val_df.groupby(['Branch', 'Tonnage']).ngroups * val_df[config.DATE_COL].nunique()}")
        print()
    
    print("RMSE Calculation:")
    print("-"*70)
    print("The RMSE = 186 (or similar) is calculated as:")
    print()
    print("  RMSE = sqrt(mean((y_actual - y_predicted)²))")
    print()
    print("Where:")
    print("  • y_actual and y_predicted are arrays of length N")
    print("  • N = Total validation rows = (combinations × weeks)")
    print()
    print("For example, if validation period has:")
    print("  • 25 combinations (Branch × Tonnage)")
    print("  • 52 weeks")
    print("  • Then N = 25 × 52 = 1,300 predictions")
    print()
    print("The RMSE aggregates ALL 1,300 predictions into a SINGLE metric.")
    print()
    print("This means:")
    print("  ✓ RMSE = 186 is the average error across ALL combinations AND ALL weeks")
    print("  ✗ It is NOT per combination per week")
    print("  ✗ It is NOT per week aggregated across combinations")
    print("  ✓ It's a single metric representing overall model performance")
    print()
    print("Interpretation:")
    print("  • RMSE = 186 means: On average, the model's predictions are")
    print("    off by about 186 units (RMSE) across all Branch/Tonnage")
    print("    combinations and all weeks in the validation period.")
    print()
    print("If you want per-combination metrics, you would need to:")
    print("  1. Group predictions by (Branch, Tonnage)")
    print("  2. Calculate RMSE separately for each combination")
    print("  3. Then average those RMSEs (or look at distribution)")
    print()


if __name__ == "__main__":
    explain_rmse_calculation()

