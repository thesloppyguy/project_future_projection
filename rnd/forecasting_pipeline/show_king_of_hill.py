"""
Display King of the Hill Results

Shows the model rankings and best performing model.
"""

import pandas as pd
from pathlib import Path
import json

try:
    from . import config
except ImportError:
    import config


def display_king_of_hill():
    """Display the King of the Hill results."""
    rankings_path = config.OUTPUT_DIR / "model_rankings.csv"
    results_path = config.OUTPUT_DIR / "multi_model_validation_results.json"
    
    if not rankings_path.exists():
        print("❌ Rankings file not found. Please run Step 5 validation first.")
        return
    
    # Load rankings
    rankings_df = pd.read_csv(rankings_path)
    
    print("\n" + "="*80)
    print("🏆 KING OF THE HILL - MODEL RANKINGS")
    print("="*80)
    print()
    
    # Display top 5
    print("Top 5 Models:")
    print("-"*80)
    for idx, row in rankings_df.head(5).iterrows():
        rank_icon = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else f"{int(row['Overall_Rank'])}."
        print(f"{rank_icon} {row['Model']:<15} "
              f"RMSE: {row['RMSE (mean)']:>8.2f} (±{row['RMSE (std)']:.2f}) | "
              f"MAE: {row['MAE (mean)']:>8.2f} (±{row['MAE (std)']:.2f}) | "
              f"R²: {row['R² (mean)']:>6.4f} (±{row['R² (std)']:.4f})")
    
    print()
    print("="*80)
    print(f"👑 KING OF THE HILL: {rankings_df.iloc[0]['Model'].upper()}")
    print("="*80)
    print()
    
    # Detailed metrics
    king = rankings_df.iloc[0]
    print("Best Model Metrics:")
    print(f"  • RMSE: {king['RMSE (mean)']:.2f} (±{king['RMSE (std)']:.2f})")
    print(f"  • MAE:  {king['MAE (mean)']:.2f} (±{king['MAE (std)']:.2f})")
    print(f"  • MAPE: {king['MAPE (mean)']:.2f} (±{king['MAPE (std)']:.2f})%")
    print(f"  • R²:   {king['R² (mean)']:.4f} (±{king['R² (std)']:.4f})")
    print(f"  • Validated on {int(king['Folds'])} fold(s)")
    print()
    
    # Show all rankings
    print("\nFull Rankings:")
    print("-"*80)
    for idx, row in rankings_df.iterrows():
        print(f"{int(row['Overall_Rank']):>2}. {row['Model']:<15} "
              f"RMSE: {row['RMSE (mean)']:>8.2f} | "
              f"MAE: {row['MAE (mean)']:>8.2f} | "
              f"R²: {row['R² (mean)']:>6.4f}")
    
    # Load detailed results if available
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        print("\n" + "="*80)
        print("Per-Fold Results:")
        print("="*80)
        for fold_key, fold_data in results.items():
            print(f"\n{fold_key.upper()}:")
            print(f"  Train: {fold_data['train_period'][0]} to {fold_data['train_period'][1]}")
            print(f"  Validation: {fold_data['val_period'][0]} to {fold_data['val_period'][1]}")
            print()
            for model_name, metrics in fold_data['models'].items():
                print(f"    {model_name:<15} RMSE: {metrics['rmse']:>8.2f} | "
                      f"MAE: {metrics['mae']:>8.2f} | R²: {metrics['r2']:>6.4f}")


if __name__ == "__main__":
    display_king_of_hill()

