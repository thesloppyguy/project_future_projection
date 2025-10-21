#!/usr/bin/env python3
"""
Sales Data Monthly Aggregation Script

Processes Final Sales.csv to create monthly aggregated statistics
grouped by State, Star Rating, and Tonnage.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def main():
    # File paths
    input_file = "/Users/sahil/Dev/cdro/rnd/data/Final Sales.csv"
    output_file = "/Users/sahil/Dev/cdro/rnd/data/monthly_sales_summary.csv"
    
    print("Reading sales data...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} records")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Parse the date column
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    # Clean up the data
    # Handle missing values in Star rating and Tonnage columns
    df['Star rating'] = df['Star rating'].replace('#N/A', np.nan)
    df['Tonnage'] = df['Tonnage'].replace('#N/A', np.nan)
    
    # Convert Tonnage to numeric, handling any remaining non-numeric values
    df['Tonnage'] = pd.to_numeric(df['Tonnage'], errors='coerce')
    
    # Convert Sales Qty. to numeric
    df['Sales Qty.'] = pd.to_numeric(df['Sales Qty.'], errors='coerce')
    
    # Remove rows with missing critical data
    df_clean = df.dropna(subset=['Sales Qty.', 'State', 'Star rating', 'Tonnage'])
    
    print(f"After cleaning: {len(df_clean)} records")
    
    # Group by Year, Month, State, Star rating, Tonnage
    grouped = df_clean.groupby(['Year', 'Month', 'State', 'Star rating', 'Tonnage'])
    
    # Calculate daily sales for each group
    daily_sales = df_clean.groupby(['Year', 'Month', 'State', 'Star rating', 'Tonnage', 'Date'])['Sales Qty.'].sum().reset_index()
    
    # Calculate metrics for each group
    results = []
    
    for (year, month, state, rating, tonnage), group in grouped:
        # Monthly total sales
        monthly_total = group['Sales Qty.'].sum()
        
        # Number of transactions
        num_transactions = len(group)
        
        # Get daily sales for this group
        group_daily = daily_sales[
            (daily_sales['Year'] == year) & 
            (daily_sales['Month'] == month) & 
            (daily_sales['State'] == state) & 
            (daily_sales['Star rating'] == rating) & 
            (daily_sales['Tonnage'] == tonnage)
        ]
        
        # Daily statistics
        daily_totals = group_daily['Sales Qty.']
        daily_max = daily_totals.max() if len(daily_totals) > 0 else 0
        daily_min = daily_totals.min() if len(daily_totals) > 0 else 0
        num_days = len(daily_totals)
        
        # Monthly average daily sales
        monthly_avg_daily = monthly_total / num_days if num_days > 0 else 0
        
        # Weekly average sales (approximate: days in month / 7)
        days_in_month = pd.Timestamp(year, month, 1).days_in_month
        weeks_in_month = days_in_month / 7
        weekly_avg = monthly_total / weeks_in_month if weeks_in_month > 0 else 0
        
        results.append({
            'Year': year,
            'Month': month,
            'State': state,
            'Star_Rating': rating,
            'Tonnage': tonnage,
            'Monthly_Total_Sales': monthly_total,
            'Monthly_Avg_Daily_Sales': round(monthly_avg_daily, 2),
            'Daily_Max_Sales': daily_max,
            'Daily_Min_Sales': daily_min,
            'Weekly_Avg_Sales': round(weekly_avg, 2),
            'Number_of_Days': num_days,
            'Number_of_Transactions': num_transactions
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by Year, Month, State, Star_Rating, Tonnage
    results_df = results_df.sort_values(['Year', 'Month', 'State', 'Star_Rating', 'Tonnage'])
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    
    print(f"\nAggregation complete!")
    print(f"Generated {len(results_df)} monthly summary records")
    print(f"Output saved to: {output_file}")
    
    # Display summary statistics
    print(f"\nSummary:")
    print(f"Date range: {results_df['Year'].min()}-{results_df['Month'].min():02d} to {results_df['Year'].max()}-{results_df['Month'].max():02d}")
    print(f"Total states: {results_df['State'].nunique()}")
    print(f"Total star ratings: {results_df['Star_Rating'].nunique()}")
    print(f"Total tonnage categories: {results_df['Tonnage'].nunique()}")
    print(f"Total monthly sales across all groups: {results_df['Monthly_Total_Sales'].sum():,.2f}")
    
    # Show first few rows
    print(f"\nFirst 5 rows of output:")
    print(results_df.head().to_string(index=False))

if __name__ == "__main__":
    main()
