from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from itertools import product

warnings.filterwarnings('ignore')


def load_sales_data():
    with open('./data/monthly_sales_summary.csv', 'r', encoding='utf-8') as file:
        df = pd.read_csv(file)
    return df


def load_weather_data():
    with open('./outputs/processed_weather_data/TN_weather_timeseries.csv', 'r', encoding='utf-8') as file:
        df = pd.read_csv(file)
    return df


def format_data(data):
    # filter for rows where state = Tamil Nadu, Star_Rating = 3 Star, Tonnage = 1.5
    filtered_sales_data = data[
        (data['State'] == 'Tamil Nadu') &
        (data['Star_Rating'] == '3 Star') &
        (data['Tonnage'] == 1.5)
    ]

    # Create a new DataFrame with time series data for Year, Month, and Monthly_Total_Sales
    response_data = filtered_sales_data[[
        'Year', 'Month', 'Monthly_Total_Sales']].copy()

    # Convert Year and Month to datetime (first day of each month)
    response_data['Date'] = pd.to_datetime(
        response_data[['Year', 'Month']].assign(day=1))

    # Sort by Date for proper time series order
    response_data = response_data.sort_values('Date')

    # Reset index for clean DataFrame
    response_data = response_data.reset_index(drop=True)

    # Reorder columns to have Date first
    response_data = response_data[['Date', 'Monthly_Total_Sales']]

    return response_data


def check_stationarity(timeseries):
    """Check if the time series is stationary using ADF test"""
    result = adfuller(timeseries.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

    if result[1] <= 0.05:
        print("Series is stationary")
        return True
    else:
        print("Series is not stationary")
        return False


def find_best_sarimax_params(endog, exog, max_p=3, max_d=2, max_q=3, max_P=2, max_D=1, max_Q=2, s=12):
    """Find the best SARIMAX parameters using grid search"""
    best_aic = float('inf')
    best_params = None

    # Generate parameter combinations
    p_values = range(0, max_p + 1)
    d_values = range(0, max_d + 1)
    q_values = range(0, max_q + 1)
    P_values = range(0, max_P + 1)
    D_values = range(0, max_D + 1)
    Q_values = range(0, max_Q + 1)

    print("Searching for best SARIMAX parameters...")
    total_combinations = len(p_values) * len(d_values) * \
        len(q_values) * len(P_values) * len(D_values) * len(Q_values)
    current_combination = 0

    for p, d, q, P, D, Q in product(p_values, d_values, q_values, P_values, D_values, Q_values):
        current_combination += 1
        if current_combination % 50 == 0:
            print(f"Progress: {current_combination}/{total_combinations}")

        try:
            model = SARIMAX(
                endog,
                exog=exog,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False, maxiter=50)

            if fitted_model.aic < best_aic:
                best_aic = fitted_model.aic
                best_params = (p, d, q, P, D, Q)
                print(
                    f"New best AIC: {best_aic:.2f} with params: {best_params}")

        except Exception as e:
            continue

    print(f"Best parameters: {best_params}")
    print(f"Best AIC: {best_aic:.2f}")
    return best_params


def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}


def create_lag_features(df, weather_df, lags=[1, 2, 3, 12]):
    """Create lagged features for both sales and weather data"""
    # Create lagged sales features
    for lag in lags:
        df[f'sales_lag_{lag}'] = df['Monthly_Total_Sales'].shift(lag)

    # Create lagged weather features
    weather_cols = ['Max_Temp', 'Min_Temp', 'Humidity', 'Wind_Speed']
    for col in weather_cols:
        for lag in lags:
            weather_df[f'{col}_lag_{lag}'] = weather_df[col].shift(lag)

    # Create rolling statistics
    for window in [3, 6, 12]:
        df[f'sales_rolling_mean_{window}'] = df['Monthly_Total_Sales'].rolling(
            window=window).mean()
        df[f'sales_rolling_std_{window}'] = df['Monthly_Total_Sales'].rolling(
            window=window).std()

    return df, weather_df


# Load and prepare data
print("Loading data...")
sales_data = load_sales_data()
time_series_data = format_data(sales_data)
weather_data = load_weather_data()

# Convert weather data date column
weather_data['Date'] = pd.to_datetime(weather_data['Date'])

# Merge sales and weather data on date
merged_data = pd.merge(time_series_data, weather_data, on='Date', how='inner')

# Create lag features
print("Creating lag features...")
merged_data, _ = create_lag_features(merged_data, merged_data)

# Split data
train_data = merged_data.query('Date < "2024-12-31"').dropna()
test_data = merged_data.query('Date >= "2024-12-31"')

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Check stationarity
print("\nChecking stationarity of sales data...")
check_stationarity(train_data['Monthly_Total_Sales'])

# Prepare features for modeling
weather_features = ['Max_Temp', 'Min_Temp', 'Humidity', 'Wind_Speed']
lag_features = [col for col in train_data.columns if 'lag_' in col]
rolling_features = [col for col in train_data.columns if 'rolling_' in col]

# Use a subset of features to avoid overfitting
selected_features = weather_features + lag_features[:8] + rolling_features[:4]

# Remove any columns that don't exist in the data
selected_features = [
    col for col in selected_features if col in train_data.columns]

print(f"Selected features: {selected_features}")

# Find best parameters (commented out for speed - you can uncomment to run)
# best_params = find_best_sarimax_params(
#     train_data['Monthly_Total_Sales'],
#     train_data[selected_features]
# )

# Use reasonable default parameters for now
best_params = (1, 1, 1, 1, 1, 1)

print(f"\nUsing parameters: {best_params}")

# Fit the model
print("Fitting SARIMAX model...")
model = SARIMAX(
    train_data['Monthly_Total_Sales'],
    exog=train_data[selected_features],
    order=best_params[:3],
    seasonal_order=best_params[3:] + (12,),
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)
print("\nModel Summary:")
print(results.summary())

# Make predictions
print("\nMaking predictions...")
if len(test_data) > 0:
    # Ensure test data has the same features
    test_features = test_data[selected_features].fillna(
        method='ffill').fillna(method='bfill')

    pred = results.get_forecast(
        steps=len(test_data),
        exog=test_features
    )

    pred_mean = pred.predicted_mean
    pred_ci = pred.conf_int()

    print("Predicted Mean Sales:")
    print(pred_mean)
    print("\nPredicted Confidence Interval:")
    print(pred_ci)

    print("\nActual Sales:")
    print(test_data['Monthly_Total_Sales'].values)

    # Evaluate the model
    print("\nModel Evaluation:")
    actual_values = test_data['Monthly_Total_Sales'].values
    predicted_values = pred_mean.values

    # Remove any NaN values for evaluation
    mask = ~(np.isnan(actual_values) | np.isnan(predicted_values))
    if np.any(mask):
        evaluate_model(actual_values[mask], predicted_values[mask])

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(train_data['Date'], train_data['Monthly_Total_Sales'],
             label='Training Data', color='blue')
    plt.plot(test_data['Date'], test_data['Monthly_Total_Sales'],
             label='Actual', color='green', marker='o')
    plt.plot(test_data['Date'], pred_mean,
             label='Predicted', color='red', marker='s')
    plt.fill_between(test_data['Date'], pred_ci.iloc[:, 0],
                     pred_ci.iloc[:, 1], alpha=0.3, color='red')
    plt.title('Sales Forecasting with SARIMAX')
    plt.xlabel('Date')
    plt.ylabel('Monthly Total Sales')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./outputs/charts/improved_sarimax_forecast.png',
                dpi=300, bbox_inches='tight')
    plt.show()

else:
    print("No test data available for evaluation")
