#!/usr/bin/env python3
"""
CLI to run and evaluate multiple forecasting pipelines on monthly sales data
with optional weather covariates.

Usage examples:
  python run_pipelines.py --state "Tamil Nadu" --star "3 Star" --tonnage 1.5 \
    --models prophet ets tbats lstm deepar transformer
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd

from model_pipelines import train_and_forecast, save_series


def load_sales(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as f:
        df = pd.read_csv(f)
    return df


def filter_segment(df: pd.DataFrame, state: str, star: str, tonnage: float) -> pd.DataFrame:
    filt = df[(df['State'] == state) & (df['Star_Rating'] == star) & (df['Tonnage'] == tonnage)].copy()
    if len(filt) == 0:
        raise ValueError(f"No rows for {state}, {star}, {tonnage}")
    filt['Date'] = pd.to_datetime(filt[['Year', 'Month']].assign(day=1))
    filt = filt.sort_values('Date').reset_index(drop=True)
    return filt[['Date', 'Monthly_Total_Sales']]


def load_weather_for_state(base_dir: str, state: str) -> pd.DataFrame:
    mapping = {'Andhra Pradesh': 'AP', 'Karnataka': 'KA', 'Kerala': 'KL', 'Telangana': 'TL', 'Tamil Nadu': 'TN'}
    code = mapping.get(state)
    if not code:
        raise ValueError(f"Unknown state mapping for {state}")
    path = os.path.join(base_dir, f"{code}_weather_timeseries.csv")
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    # basic engineered covariates
    df['temp_avg'] = (df['Max_Temp'] + df['Min_Temp']) / 2
    return df


def make_future_weather(weather_df: pd.DataFrame, start_date: pd.Timestamp, periods: int) -> pd.DataFrame:
    # Simple repeat-last-year pattern for covariates
    months = pd.date_range(start=start_date, periods=periods, freq='MS')
    fut = []
    for d in months:
        src = weather_df[weather_df['Date'].dt.month == d.month]
        row = src.iloc[-1] if len(src) > 0 else weather_df.iloc[-1]
        fut.append({
            'Date': d,
            'Max_Temp': row['Max_Temp'],
            'Min_Temp': row['Min_Temp'],
            'Humidity': row['Humidity'],
            'Wind_Speed': row['Wind_Speed'],
            'temp_avg': row['temp_avg']
        })
    fut_df = pd.DataFrame(fut)
    fut_df['ds'] = fut_df['Date']
    return fut_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sales', default='data/monthly_sales_summary.csv')
    parser.add_argument('--weather_dir', default='outputs/processed_weather_data')
    parser.add_argument('--state', required=True)
    parser.add_argument('--star', required=True)
    parser.add_argument('--tonnage', type=float, required=True)
    parser.add_argument('--split', default='2024-12-31')
    parser.add_argument('--models', nargs='+', default=['prophet', 'ets', 'tbats', 'lstm', 'deepar', 'transformer'])
    parser.add_argument('--horizon', type=int, default=None)
    args = parser.parse_args()

    os.makedirs('outputs/evaluations', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)

    sales = load_sales(args.sales)
    seg = filter_segment(sales, args.state, args.star, args.tonnage)
    weather = load_weather_for_state(args.weather_dir, args.state)

    merged = seg.merge(weather, on='Date', how='inner')

    train = merged[merged['Date'] < args.split].copy()
    test = merged[merged['Date'] >= args.split].copy()
    if len(train) == 0:
        raise ValueError('No training rows after merge')

    n = args.horizon if args.horizon is not None else len(test)
    future_weather = make_future_weather(weather, start_date=test['Date'].min() if len(test) else (train['Date'].max() + pd.offsets.MonthBegin(1)), periods=n)

    cov_train = merged[['Date', 'Max_Temp', 'Min_Temp', 'Humidity', 'Wind_Speed', 'temp_avg']].copy()

    rows = []
    all_preds = {}
    for m in args.models:
        try:
            preds, metrics = train_and_forecast(
                model_name=m,
                train_df=train[['Date', 'Monthly_Total_Sales']].copy(),
                test_df=test[['Date', 'Monthly_Total_Sales']].copy(),
                time_col='Date',
                value_col='Monthly_Total_Sales',
                past_covariates_train=cov_train,
                future_covariates_test=future_weather,
                n_forecast=n,
            )
            rows.append({'Model': m, **metrics})
            all_preds[m] = preds
            # save individual predictions
            save_series(preds, f"outputs/evaluations/{m}_predictions.csv")
            print(f"[OK] {m}: MAE={metrics['MAE']:.2f} RMSE={metrics['RMSE']:.2f} MAPE={metrics['MAPE']:.2f}%")
        except Exception as e:
            rows.append({'Model': m, 'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'error': str(e)})
            print(f"[ERR] {m}: {e}")

    results = pd.DataFrame(rows)
    results.to_csv('outputs/evaluations/model_pipelines_results.csv', index=False)

    # Combine predictions for comparison if test available
    if len(test) > 0 and len(all_preds) > 0:
        comp = pd.DataFrame({'Date': test['Date']}).set_index('Date')
        for k, s in all_preds.items():
            comp[k] = s.reindex(comp.index)
        comp['Actual'] = test.set_index('Date')['Monthly_Total_Sales']
        comp.to_csv('outputs/evaluations/model_pipelines_comparison.csv')

    print('Saved: outputs/evaluations/model_pipelines_results.csv')


if __name__ == '__main__':
    main()


