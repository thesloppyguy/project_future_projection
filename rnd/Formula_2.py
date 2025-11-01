import pandas as pd
import numpy as np

# load your csv into df; ensure Date parsed as datetime
df = pd.read_csv('./data/general_df_monthly.csv', parse_dates=['Date'])

# Example variable: df with columns ['Date','Branch','Qty']
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# 1) company totals by year-month
company = df.groupby(['Year', 'Month'])[
    'Qty'].sum().reset_index(name='CompanyTotal')

# restrict years for seasonality calculation
years = [2021, 2022, 2023]
season = company[company['Year'].isin(years)]

# 2) average total per calendar month across the selected years
avg_by_month = season.groupby(
    'Month')['CompanyTotal'].mean().rename('AvgTotal').reset_index()

# 3) average March total
avg_march = avg_by_month.loc[avg_by_month['Month'] == 3, 'AvgTotal'].iloc[0]

# 4) seasonality index
avg_by_month['SI'] = avg_by_month['AvgTotal'] / avg_march

# 5) anchor (actual total in March 2024)
anchor = company[(company['Year'] == 2024) & (
    company['Month'] == 3)]['CompanyTotal'].iloc[0]  # should be 76810.5

# 6) company forecasts Apr2024-Mar2025
months = list(range(4, 13)) + list(range(1, 4))  # Apr..Dec then Jan..Mar
forecast_company = []
for m in months:
    si = avg_by_month.loc[avg_by_month['Month'] == m, 'SI'].iloc[0]
    forecast_company.append({'Month': m, 'ForecastCompany': anchor * si})

fc_company_df = pd.DataFrame(forecast_company)

# 7) branch shares from March-2024
march2024 = df[(df['Year'] == 2024) & (df['Month'] == 3)
               ].groupby('Branch')['Qty'].sum()
march_total = march2024.sum()
shares = (march2024 / march_total).rename('Share').reset_index()

# 8) expand company forecast to branches
fc_company_df = fc_company_df.assign(key=1)
shares = shares.assign(key=1)
fc_branches = fc_company_df.merge(shares, on='key').drop(columns='key')
fc_branches['ForecastBranch'] = fc_branches['ForecastCompany'] * \
    fc_branches['Share']

# round if needed
fc_branches['ForecastBranch'] = fc_branches['ForecastBranch'].round(0)
fc_company_df['ForecastCompany'] = fc_company_df['ForecastCompany'].round(0)

print(fc_company_df)
# test_start_date = '2024-04-01'
# actual_company = (
#     df[df['Date'] >= test_start_date]
#     .groupby('Date')['Qty']
#     .sum()
#     .reset_index()
#     .rename(columns={'Qty': 'ActualCompany'})
# )
# actual_branch = (
#     df[df['Date'] >= test_start_date]
#     .groupby(['Date', 'Branch'])['Qty']
#     .sum()
#     .reset_index()
#     .rename(columns={'Qty': 'ActualBranch'})
# )


# # Merge company forecasts with actuals
# company_comparison = fc_company_df['ForecastCompany'].merge(
#     actual_company,
#     on='Date',
#     how='left'
# )
# company_comparison['Error'] = company_comparison['ForecastCompany'] - \
#     company_comparison['ActualCompany']
# company_comparison['Error_Pct'] = np.where(
#     company_comparison['ActualCompany'] != 0,
#     (company_comparison['Error'] / company_comparison['ActualCompany']) * 100,
#     np.nan
# )
# company_comparison['AbsError'] = company_comparison['Error'].abs()
# company_comparison['AbsErrorPct'] = company_comparison['Error_Pct'].abs()

# # Merge branch forecasts with actuals
# branch_comparison = fc_branches['ForecastBranch'].merge(
#     actual_branch,
#     on=['Date', 'Branch'],
#     how='left'
# )
# branch_comparison['Error'] = branch_comparison['ForecastBranch'] - \
#     branch_comparison['ActualBranch']
# branch_comparison['Error_Pct'] = np.where(
#     branch_comparison['ActualBranch'] != 0,
#     (branch_comparison['Error'] / branch_comparison['ActualBranch']) * 100,
#     np.nan
# )
# branch_comparison['AbsError'] = branch_comparison['Error'].abs()
# branch_comparison['AbsErrorPct'] = branch_comparison['Error_Pct'].abs()

# # Calculate metrics for company forecast (excluding NaN values)
# company_comparison_valid = company_comparison.dropna(subset=['ActualCompany'])
# company_metrics = {
#     'MAE': company_comparison_valid['AbsError'].mean() if len(company_comparison_valid) > 0 else np.nan,
#     'RMSE': np.sqrt((company_comparison_valid['Error'] ** 2).mean()) if len(company_comparison_valid) > 0 else np.nan,
#     'MAPE': company_comparison_valid['AbsErrorPct'].mean() if len(company_comparison_valid) > 0 else np.nan,
#     'MeanError': company_comparison_valid['Error'].mean() if len(company_comparison_valid) > 0 else np.nan,
# }

# # Calculate metrics for branch forecast (excluding NaN values)
# branch_comparison_valid = branch_comparison.dropna(subset=['ActualBranch'])
# branch_metrics = {
#     'MAE': branch_comparison_valid['AbsError'].mean() if len(branch_comparison_valid) > 0 else np.nan,
#     'RMSE': np.sqrt((branch_comparison_valid['Error'] ** 2).mean()) if len(branch_comparison_valid) > 0 else np.nan,
#     'MAPE': branch_comparison_valid['AbsErrorPct'].mean() if len(branch_comparison_valid) > 0 else np.nan,
#     'MeanError': branch_comparison_valid['Error'].mean() if len(branch_comparison_valid) > 0 else np.nan,
# }

# print("=" * 80)
# print("COMPANY FORECAST COMPARISON")
# print("=" * 80)
# print(f"MAE (Mean Absolute Error): {company_metrics['MAE']:.2f}")
# print(f"RMSE (Root Mean Squared Error): {company_metrics['RMSE']:.2f}")
# print(f"MAPE (Mean Absolute Percentage Error): {company_metrics['MAPE']:.2f}%")
# print(f"Mean Error: {company_metrics['MeanError']:.2f}")
# print("\nCompany Forecast vs Actual:")
# print(company_comparison[['Date', 'Month', 'ForecastCompany',
#       'ActualCompany', 'Error', 'Error_Pct']].to_string(index=False))


# print("\n" + "=" * 80)
# print("BRANCH FORECAST COMPARISON")
# print("=" * 80)
# print(f"MAE (Mean Absolute Error): {branch_metrics['MAE']:.2f}")
# print(f"RMSE (Root Mean Squared Error): {branch_metrics['RMSE']:.2f}")
# print(f"MAPE (Mean Absolute Percentage Error): {branch_metrics['MAPE']:.2f}%")
# print(f"Mean Error: {branch_metrics['MeanError']:.2f}")
# print("\nBranch Forecast vs Actual (first 20 rows):")
# print(branch_comparison[['Date', 'Branch', 'ForecastBranch',
#       'ActualBranch', 'Error', 'Error_Pct']].head(20).to_string(index=False))

# # Display summary by branch
# print("\n" + "=" * 80)
# print("BRANCH-WISE METRICS")
# print("=" * 80)
# branch_summary = branch_comparison_valid.groupby('Branch').agg({
#     'AbsError': 'mean',
#     'AbsErrorPct': 'mean',
#     'Error': 'mean'
# }).round(2)
# branch_summary.columns = ['MAE', 'MAPE', 'MeanError']
# print(branch_summary)
