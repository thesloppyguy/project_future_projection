import pandas as pd
import numpy as np
from scipy import stats
from itertools import product
import random
from typing import Dict, List, Any, Optional
import warnings


def forecast_seasonal_anchor(
    df,
    months_ahead=12,
    season_years=[2021, 2022, 2023],
    season_norm='annual_mean',     # 'base_month' or 'annual_mean'
    base_month=3,
    # --- anchor parameters ---
    use_same_anchor_month=False,   # True = use historical anchor(s)
    anchor_months_list=[3],        # list of anchor months if same anchor
    anchor_years=[2021, 2022, 2023, 2024],
    anchor_months_recent=1,        # number of recent months if not using same anchor
    # --- trend parameters ---
    trend_method='growth',           # 'none', 'linear', 'growth'
    trend_window_months=12,
    # --- branch share parameters ---
    share_method='fixed',          # 'fixed' or 'moving'
    share_moving_window=3,
    # --- confidence interval ---
    conf_level=0.95
):
    """
    Forecast company & branch sales with seasonal index and flexible anchor logic.
    Includes:
      - Fixed or moving branch share
      - Historical or recent anchor
      - Confidence intervals
    """
    df = df.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # --- 1️⃣ Company totals ---
    company = df.groupby(['Year', 'Month'])['Qty'].sum().reset_index()

    # --- 2️⃣ Seasonality pattern ---
    season = company[company['Year'].isin(season_years)]
    avg_by_month = season.groupby(
        'Month')['Qty'].mean().rename('AvgTotal').reset_index()

    if season_norm == 'base_month':
        avg_m0 = avg_by_month.loc[avg_by_month['Month']
                                  == base_month, 'AvgTotal'].iloc[0]
        avg_by_month['SI'] = avg_by_month['AvgTotal'] / avg_m0
    else:
        annual_mean = avg_by_month['AvgTotal'].mean()
        avg_by_month['SI'] = avg_by_month['AvgTotal'] / annual_mean

    # --- 3️⃣ Anchor calculation ---
    if use_same_anchor_month:
        mask = company['Month'].isin(
            anchor_months_list) & company['Year'].isin(anchor_years)
        anchor_vals = company.loc[mask, 'Qty']
        Anchor = anchor_vals.mean()
        anchor_description = f"Average of months {anchor_months_list} across years {anchor_years}"
    else:
        df_company_dates = df.groupby(
            'Date')['Qty'].sum().reset_index().sort_values('Date')
        Anchor = df_company_dates.tail(anchor_months_recent)['Qty'].mean()
        anchor_description = f"Average of last {anchor_months_recent} month(s)"

    # --- 4️⃣ Trend multiplier ---
    df_company_dates = df.groupby(
        'Date')['Qty'].sum().reset_index().sort_values('Date')

    def TrendMultiplier(h): return 1.2
    if trend_method == 'linear':
        df_company_dates['t'] = np.arange(len(df_company_dates)) + 1
        slope, intercept, _, _, _ = stats.linregress(
            df_company_dates['t'], df_company_dates['Qty'])
        t_last = df_company_dates['t'].iloc[-1]

        def TrendMultiplier(h):
            num = intercept + slope * (t_last + h)
            den = intercept + slope * t_last
            return num / den if den != 0 else 1.0
    elif trend_method == 'growth':
        if len(df_company_dates) >= trend_window_months + 1:
            Q_start = df_company_dates['Qty'].iloc[-(trend_window_months + 1)]
            Q_end = df_company_dates['Qty'].iloc[-1]
            g = (Q_end / Q_start) ** (1.0 / trend_window_months) - 1.0
            def TrendMultiplier(h): return (1 + g) ** h

    # --- 5️⃣ Company-level forecast ---
    last_date = df['Date'].max()
    start = (last_date + pd.DateOffset(months=1)).replace(day=1)
    months = [(start + pd.DateOffset(months=i)).to_period('M').to_timestamp()
              for i in range(months_ahead)]

    fc_list = []
    for h, dt in enumerate(months, start=1):
        m = dt.month
        si = float(avg_by_month.loc[avg_by_month['Month'] == m, 'SI'])
        fc_company = Anchor * TrendMultiplier(h) * si
        fc_list.append({'Date': dt, 'Month': m, 'h': h,
                       'ForecastCompany': fc_company})
    fc_df = pd.DataFrame(fc_list)

    # --- 6️⃣ Confidence intervals ---
    sd_by_month = season.groupby(
        'Month')['Qty'].std().rename('SD').reset_index()
    avg_by_month = avg_by_month.merge(sd_by_month, on='Month', how='left')
    avg_by_month['CV'] = avg_by_month['SD'] / avg_by_month['AvgTotal']
    z = stats.norm.ppf((1 + conf_level) / 2)

    fc_df = fc_df.merge(avg_by_month[['Month', 'CV']], on='Month', how='left')
    fc_df['PI_lower'] = fc_df['ForecastCompany'] - \
        z * fc_df['CV'] * fc_df['ForecastCompany']
    fc_df['PI_upper'] = fc_df['ForecastCompany'] + \
        z * fc_df['CV'] * fc_df['ForecastCompany']

    # --- 7️⃣ Branch shares ---
    df_sorted = df.sort_values('Date')
    last_month = df_sorted['Date'].max().to_period('M').to_timestamp()

    if share_method == 'fixed':
        # Use last month
        branch_data = df[df['Date'] == last_month].groupby('Branch')[
            'Qty'].sum()
        share_df = (branch_data / branch_data.sum()
                    ).rename('Share').reset_index()
        share_description = f"Fixed shares from {last_month.strftime('%Y-%m')}"
    elif share_method == 'moving':
        # Use average of last N months
        start_period = (
            last_month - pd.DateOffset(months=share_moving_window - 1)).replace(day=1)
        recent = df[(df['Date'] >= start_period) & (df['Date'] <= last_month)]
        branch_data = recent.groupby('Branch')['Qty'].sum()
        share_df = (branch_data / branch_data.sum()
                    ).rename('Share').reset_index()
        share_description = f"Moving average of last {share_moving_window} months"
    else:
        raise ValueError("share_method must be 'fixed' or 'moving'")

    # --- 8️⃣ Merge company & branch forecasts ---
    fc_df['key'] = 1
    share_df['key'] = 1
    fc_br = fc_df.merge(share_df, on='key').drop(columns='key')
    fc_br['ForecastBranch'] = fc_br['ForecastCompany'] * fc_br['Share']

    return {
        'company_forecast': fc_df,
        'branch_forecast': fc_br[['Date', 'Branch', 'ForecastBranch', 'Share']],
        'anchor_value': Anchor,
        'anchor_description': anchor_description,
        'share_description': share_description,
        'seasonality': avg_by_month[['Month', 'AvgTotal', 'SI', 'CV']],
    }


def evaluate_forecast_params(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: Dict[str, Any],
    metric: str = 'MAPE',
    level: str = 'company'
) -> Dict[str, float]:
    """
    Evaluate forecast performance with given hyperparameters.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data with columns ['Date', 'Branch', 'Qty']
    test_df : pd.DataFrame
        Test data with columns ['Date', 'Branch', 'Qty'] (actuals for validation)
    params : Dict[str, Any]
        Hyperparameters for forecast_seasonal_anchor
    metric : str
        Metric to optimize: 'MAPE', 'MAE', 'RMSE', or 'MeanError'
    level : str
        Level to evaluate: 'company' or 'branch'

    Returns:
    --------
    Dict with evaluation metrics and the score
    """
    try:
        # Generate forecast using training data
        forecast_result = forecast_seasonal_anchor(train_df, **params)

        # Get actual values for the forecast period
        forecast_start_date = train_df['Date'].max() + pd.DateOffset(months=1)
        actual_company = (
            test_df[test_df['Date'] >= forecast_start_date]
            .groupby('Date')['Qty']
            .sum()
            .reset_index()
            .rename(columns={'Qty': 'ActualCompany'})
        )

        if level == 'company':
            # Merge company forecasts with actuals
            comparison = forecast_result['company_forecast'].merge(
                actual_company,
                on='Date',
                how='left'
            )
            comparison = comparison.dropna(subset=['ActualCompany'])

            if len(comparison) == 0:
                return {'score': np.inf, 'MAE': np.inf, 'RMSE': np.inf, 'MAPE': np.inf, 'MeanError': np.inf}

            comparison['Error'] = comparison['ForecastCompany'] - \
                comparison['ActualCompany']
            comparison['Error_Pct'] = np.where(
                comparison['ActualCompany'] != 0,
                (comparison['Error'] / comparison['ActualCompany']) * 100,
                np.nan
            )
            comparison['AbsError'] = comparison['Error'].abs()
            comparison['AbsErrorPct'] = comparison['Error_Pct'].abs()

            metrics = {
                'MAE': comparison['AbsError'].mean(),
                'RMSE': np.sqrt((comparison['Error'] ** 2).mean()),
                'MAPE': comparison['AbsErrorPct'].mean(),
                'MeanError': comparison['Error'].mean(),
            }
        else:  # branch level
            actual_branch = (
                test_df[test_df['Date'] >= forecast_start_date]
                .groupby(['Date', 'Branch'])['Qty']
                .sum()
                .reset_index()
                .rename(columns={'Qty': 'ActualBranch'})
            )

            comparison = forecast_result['branch_forecast'].merge(
                actual_branch,
                on=['Date', 'Branch'],
                how='left'
            )
            comparison = comparison.dropna(subset=['ActualBranch'])

            if len(comparison) == 0:
                return {'score': np.inf, 'MAE': np.inf, 'RMSE': np.inf, 'MAPE': np.inf, 'MeanError': np.inf}

            comparison['Error'] = comparison['ForecastBranch'] - \
                comparison['ActualBranch']
            comparison['Error_Pct'] = np.where(
                comparison['ActualBranch'] != 0,
                (comparison['Error'] / comparison['ActualBranch']) * 100,
                np.nan
            )
            comparison['AbsError'] = comparison['Error'].abs()
            comparison['AbsErrorPct'] = comparison['Error_Pct'].abs()

            metrics = {
                'MAE': comparison['AbsError'].mean(),
                'RMSE': np.sqrt((comparison['Error'] ** 2).mean()),
                'MAPE': comparison['AbsErrorPct'].mean(),
                'MeanError': comparison['Error'].mean(),
            }

        metrics['score'] = metrics[metric]
        return metrics

    except Exception as e:
        warnings.warn(f"Error evaluating parameters {params}: {str(e)}")
        return {'score': np.inf, 'MAE': np.inf, 'RMSE': np.inf, 'MAPE': np.inf, 'MeanError': np.inf}


def get_default_param_grid() -> Dict[str, List[Any]]:
    """
    Get default hyperparameter search space.
    Optimized for peak sales months: March (3), April (4), May (5), June (6).

    Returns:
    --------
    Dict with parameter names as keys and lists of values to try
    """
    # Get available years from data (will be set dynamically)
    # Prioritizing peak sales months (3, 4, 5, 6) based on business knowledge
    return {
        'season_years': [
            [2021, 2022, 2023],
            [2022, 2023],
            [2021, 2022, 2023, 2024, 2025],
            [2022, 2023, 2024, 2025],
            [2023, 2024, 2025],
        ],
        'season_norm': ['annual_mean', 'base_month'],
        # Focus on peak months: March (3), April (4), May (5), June (6)
        'base_month': [3, 4, 5, 6, 1, 12],
        'use_same_anchor_month': [True, False],
        # Prioritize peak months and combinations of peak months
        'anchor_months_list': [
            [3], [4], [5], [6],           # Individual peak months
            [3, 4], [4, 5], [5, 6],       # Adjacent peak month pairs
            # Peak month triplets and combinations
            [3, 4, 5], [4, 5, 6], [3, 5],
            [1], [12],                    # Other months for comparison
        ],
        'anchor_years': [
            [2021, 2022, 2023],
            [2022, 2023],
            [2021, 2022, 2023, 2024, 2025],
            [2022, 2023, 2024, 2025],
            [2023, 2024, 2025],
        ],
        'anchor_months_recent': [1, 2, 3, 6],
        'trend_method': ['none', 'linear', 'growth'],
        'trend_window_months': [6, 12, 18, 24],
        'share_method': ['fixed', 'moving'],
        'share_moving_window': [1, 2, 3, 6],
        'conf_level': [0.90, 0.95, 0.99],
    }


def filter_valid_params(
    params: Dict[str, Any],
    available_years: List[int]
) -> bool:
    """
    Filter out invalid parameter combinations.

    Parameters:
    -----------
    params : Dict[str, Any]
        Parameter combination to validate
    available_years : List[int]
        Available years in the dataset

    Returns:
    --------
    bool: True if valid, False otherwise
    """
    # Filter years that are not available
    if 'season_years' in params:
        params['season_years'] = [
            y for y in params['season_years'] if y in available_years]
        if not params['season_years']:
            return False

    if 'anchor_years' in params:
        params['anchor_years'] = [
            y for y in params['anchor_years'] if y in available_years]
        if not params['anchor_years']:
            return False

    # If using same anchor month, anchor_months_list must be valid
    if params.get('use_same_anchor_month', False):
        if not params.get('anchor_months_list'):
            return False

    return True


def tune_forecast_hyperparameters(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    method: str = 'random',
    n_iter: int = 50,
    metric: str = 'MAPE',
    level: str = 'company',
    random_state: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Tune hyperparameters for forecast_seasonal_anchor.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data with columns ['Date', 'Branch', 'Qty']
    test_df : pd.DataFrame
        Test data with columns ['Date', 'Branch', 'Qty'] (actuals for validation)
    param_grid : Optional[Dict[str, List[Any]]]
        Custom parameter grid. If None, uses default grid.
    method : str
        Search method: 'grid' (exhaustive) or 'random' (random search)
    n_iter : int
        Number of iterations for random search (ignored for grid search)
    metric : str
        Metric to optimize: 'MAPE', 'MAE', 'RMSE', or 'MeanError'
    level : str
        Level to optimize: 'company' or 'branch'
    random_state : Optional[int]
        Random seed for reproducibility
    verbose : bool
        Whether to print progress

    Returns:
    --------
    Dict with:
        - 'best_params': Best hyperparameters found
        - 'best_score': Best score achieved
        - 'best_metrics': All metrics for best parameters
        - 'results': List of all tried parameter combinations and their scores
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    if param_grid is None:
        param_grid = get_default_param_grid()

    # Get available years from training data
    available_years = sorted(train_df['Date'].dt.year.unique())

    # Filter param_grid to remove invalid years
    param_grid = param_grid.copy()
    if 'season_years' in param_grid:
        param_grid['season_years'] = [
            [y for y in years if y in available_years]
            for years in param_grid['season_years']
        ]
        param_grid['season_years'] = [
            y for y in param_grid['season_years'] if y]

    if 'anchor_years' in param_grid:
        param_grid['anchor_years'] = [
            [y for y in years if y in available_years]
            for years in param_grid['anchor_years']
        ]
        param_grid['anchor_years'] = [
            y for y in param_grid['anchor_years'] if y]

    # Generate parameter combinations
    if method == 'grid':
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

        if verbose:
            print(
                f"Grid search: {len(param_combinations)} combinations to evaluate")
    else:  # random search
        param_combinations = []
        for _ in range(n_iter):
            combo = {}
            for key, values in param_grid.items():
                combo[key] = random.choice(values)
            param_combinations.append(combo)

        if verbose:
            print(f"Random search: {n_iter} combinations to evaluate")

    # Evaluate each combination
    results = []
    best_score = np.inf
    best_params = None
    best_metrics = None

    for i, params_dict in enumerate(param_combinations):
        # Create parameter dict
        params = dict(zip(param_grid.keys(), params_dict)
                      ) if method == 'grid' else params_dict

        # Filter invalid parameters
        if not filter_valid_params(params, available_years):
            continue

        # Set default values for optional parameters
        params.setdefault('months_ahead', 12)

        # Evaluate
        metrics = evaluate_forecast_params(
            train_df, test_df, params, metric=metric, level=level
        )

        results.append({
            'params': params.copy(),
            'metrics': metrics.copy(),
            'score': metrics['score']
        })

        if metrics['score'] < best_score:
            best_score = metrics['score']
            best_params = params.copy()
            best_metrics = metrics.copy()

        if verbose and (i + 1) % max(1, len(param_combinations) // 10) == 0:
            print(f"  Progress: {i+1}/{len(param_combinations)} | "
                  f"Best {metric}: {best_score:.4f}")

    if best_params is None:
        raise ValueError("No valid parameter combinations found!")

    if verbose:
        print(f"\n{'='*80}")
        print("HYPERPARAMETER TUNING RESULTS")
        print('='*80)
        print(f"Best {metric}: {best_score:.4f}")
        print(f"\nBest Parameters:")
        for key, value in sorted(best_params.items()):
            print(f"  {key}: {value}")
        print(f"\nBest Metrics:")
        for key, value in best_metrics.items():
            if key != 'score':
                print(f"  {key}: {value:.4f}")

    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_metrics': best_metrics,
        'results': results
    }


df = pd.read_csv('./data/general_df_monthly.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Split data for training and testing
train_df = df[df['Date'] < '2024-04-01']
test_df = df[df['Date'] >= '2024-04-01']
# anchor_months_list=[4],
# anchor_months_recent=2,
# anchor_years=[2022, 2023],
# base_month=3,
# conf_level=0.95,
# months_ahead=12,
# season_norm='base_month',
# season_years=[2023, 2024, 2025],
# share_method='fixed',
# share_moving_window=6,
# trend_method='growth',
# trend_window_months=24,
# use_same_anchor_month=True
response = forecast_seasonal_anchor(train_df,
                                    anchor_months_list=[3, 4, 5],
                                    anchor_months_recent=1,
                                    anchor_years=[2022, 2023, 2024],
                                    base_month=12,
                                    conf_level=0.9,
                                    months_ahead=12,
                                    season_norm='base_month',
                                    season_years=[2021, 2022, 2023, 2024],
                                    share_method='fixed',
                                    share_moving_window=3,
                                    trend_method='none',
                                    trend_window_months=12,
                                    use_same_anchor_month=False,)
#   anchor_months_list: [3, 4, 5]
#   anchor_months_recent: 1
#   anchor_years: [2022, 2023, 2024]
#   base_month: 12
#   conf_level: 0.9
#   months_ahead: 12
#   season_norm: base_month
#   season_years: [2021, 2022, 2023, 2024]
#   share_method: fixed
#   share_moving_window: 3
#   trend_method: none
#   trend_window_months: 12
#   use_same_anchor_month: False
test_start_date = train_df['Date'].max() + pd.DateOffset(months=1)
actual_company = (
    test_df[test_df['Date'] >= test_start_date]
    .groupby('Date')['Qty']
    .sum()
    .reset_index()
    .rename(columns={'Qty': 'ActualCompany'})
)
actual_branch = (
    test_df[test_df['Date'] >= test_start_date]
    .groupby(['Date', 'Branch'])['Qty']
    .sum()
    .reset_index()
    .rename(columns={'Qty': 'ActualBranch'})
)


# Merge company forecasts with actuals
company_comparison = response['company_forecast'].merge(
    actual_company,
    on='Date',
    how='left'
)
company_comparison['Error'] = company_comparison['ForecastCompany'] - \
    company_comparison['ActualCompany']
company_comparison['Error_Pct'] = np.where(
    company_comparison['ActualCompany'] != 0,
    (company_comparison['Error'] / company_comparison['ActualCompany']) * 100,
    np.nan
)
company_comparison['AbsError'] = company_comparison['Error'].abs()
company_comparison['AbsErrorPct'] = company_comparison['Error_Pct'].abs()

# Merge branch forecasts with actuals
branch_comparison = response['branch_forecast'].merge(
    actual_branch,
    on=['Date', 'Branch'],
    how='left'
)
branch_comparison['Error'] = branch_comparison['ForecastBranch'] - \
    branch_comparison['ActualBranch']
branch_comparison['Error_Pct'] = np.where(
    branch_comparison['ActualBranch'] != 0,
    (branch_comparison['Error'] / branch_comparison['ActualBranch']) * 100,
    np.nan
)
branch_comparison['AbsError'] = branch_comparison['Error'].abs()
branch_comparison['AbsErrorPct'] = branch_comparison['Error_Pct'].abs()

# Calculate metrics for company forecast (excluding NaN values)
company_comparison_valid = company_comparison.dropna(subset=['ActualCompany'])
company_metrics = {
    'MAE': company_comparison_valid['AbsError'].mean() if len(company_comparison_valid) > 0 else np.nan,
    'RMSE': np.sqrt((company_comparison_valid['Error'] ** 2).mean()) if len(company_comparison_valid) > 0 else np.nan,
    'MAPE': company_comparison_valid['AbsErrorPct'].mean() if len(company_comparison_valid) > 0 else np.nan,
    'MeanError': company_comparison_valid['Error'].mean() if len(company_comparison_valid) > 0 else np.nan,
}

# Calculate metrics for branch forecast (excluding NaN values)
branch_comparison_valid = branch_comparison.dropna(subset=['ActualBranch'])
branch_metrics = {
    'MAE': branch_comparison_valid['AbsError'].mean() if len(branch_comparison_valid) > 0 else np.nan,
    'RMSE': np.sqrt((branch_comparison_valid['Error'] ** 2).mean()) if len(branch_comparison_valid) > 0 else np.nan,
    'MAPE': branch_comparison_valid['AbsErrorPct'].mean() if len(branch_comparison_valid) > 0 else np.nan,
    'MeanError': branch_comparison_valid['Error'].mean() if len(branch_comparison_valid) > 0 else np.nan,
}

print("=" * 80)
print("COMPANY FORECAST COMPARISON")
print("=" * 80)
print(f"MAE (Mean Absolute Error): {company_metrics['MAE']:.2f}")
print(f"RMSE (Root Mean Squared Error): {company_metrics['RMSE']:.2f}")
print(f"MAPE (Mean Absolute Percentage Error): {company_metrics['MAPE']:.2f}%")
print(f"Mean Error: {company_metrics['MeanError']:.2f}")
print("\nCompany Forecast vs Actual:")
print(company_comparison[['Date', 'Month', 'ForecastCompany',
      'ActualCompany', 'Error', 'Error_Pct']].to_string(index=False))


print("\n" + "=" * 80)
print("BRANCH FORECAST COMPARISON")
print("=" * 80)
print(f"MAE (Mean Absolute Error): {branch_metrics['MAE']:.2f}")
print(f"RMSE (Root Mean Squared Error): {branch_metrics['RMSE']:.2f}")
print(f"MAPE (Mean Absolute Percentage Error): {branch_metrics['MAPE']:.2f}%")
print(f"Mean Error: {branch_metrics['MeanError']:.2f}")
print("\nBranch Forecast vs Actual (first 20 rows):")
print(branch_comparison[['Date', 'Branch', 'ForecastBranch',
      'ActualBranch', 'Error', 'Error_Pct']].head(20).to_string(index=False))

# Display summary by branch
print("\n" + "=" * 80)
print("BRANCH-WISE METRICS")
print("=" * 80)
branch_summary = branch_comparison_valid.groupby('Branch').agg({
    'AbsError': 'mean',
    'AbsErrorPct': 'mean',
    'Error': 'mean'
}).round(2)
branch_summary.columns = ['MAE', 'MAPE', 'MeanError']
print(branch_summary)

# # Example: Run hyperparameter tuning
# print("=" * 80)
# print("HYPERPARAMETER TUNING EXAMPLE")
# print("=" * 80)
# print("\nStarting hyperparameter tuning with random search...")
# print("This may take a few minutes depending on n_iter...\n")

# tuning_results = tune_forecast_hyperparameters(
#     train_df=train_df,
#     test_df=test_df,
#     method='random',  # 'random' or 'grid'
#     n_iter=30,  # Number of random combinations to try
#     metric='RMSE',  # 'MAPE', 'MAE', 'RMSE', or 'MeanError'
#     level='company',  # 'company' or 'branch'
#     random_state=42,
#     verbose=True
# )

# # Use best parameters for final forecast
# print("\n" + "=" * 80)
# print("FORECAST WITH BEST PARAMETERS")
# print("=" * 80)
# response = forecast_seasonal_anchor(train_df, **tuning_results['best_params'])


# # Compare forecasts against actual values
# # Get actual values for the forecast period
# forecast_start_date = train_df['Date'].max() + pd.DateOffset(months=1)
# actual_company = (
#     test_df[test_df['Date'] >= forecast_start_date]
#     .groupby('Date')['Qty']
#     .sum()
#     .reset_index()
#     .rename(columns={'Qty': 'ActualCompany'})
# )
# actual_branch = (
#     test_df[test_df['Date'] >= forecast_start_date]
#     .groupby(['Date', 'Branch'])['Qty']
#     .sum()
#     .reset_index()
#     .rename(columns={'Qty': 'ActualBranch'})
# )

# # Merge company forecasts with actuals
# company_comparison = response['company_forecast'].merge(
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
# branch_comparison = response['branch_forecast'].merge(
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
