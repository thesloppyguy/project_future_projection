from statsmodels.tsa.statespace.sarimax import SARIMAX
from nixtla import NixtlaClient
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

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
    response_data = response_data[[
        'Date', 'Monthly_Total_Sales']]

    return response_data


sales_data = load_sales_data()
time_series_data = format_data(sales_data)
df_train = time_series_data.query('Date < "2024-12-31"')
df_test = time_series_data.query('Date >= "2024-12-31"')
weather_data = load_weather_data()
weather_data_train = weather_data.query(
    'Date < "2024-12-31" & Date >= "2022-01-01"')
weather_data_test = weather_data.query('Date >= "2024-12-31"')

# Reset indices to ensure alignment between sales and weather data
df_train_aligned = df_train.reset_index(drop=True)
weather_data_train_aligned = weather_data_train.reset_index(drop=True)

# Add lag features to improve model performance
df_train_aligned['sales_lag_1'] = df_train_aligned['Monthly_Total_Sales'].shift(
    1)
df_train_aligned['sales_lag_12'] = df_train_aligned['Monthly_Total_Sales'].shift(
    12)
weather_data_train_aligned['temp_avg'] = (
    weather_data_train_aligned['Max_Temp'] + weather_data_train_aligned['Min_Temp']) / 2

# Remove rows with NaN values from lag features
df_train_clean = df_train_aligned.dropna().reset_index(drop=True)
weather_data_clean = weather_data_train_aligned.iloc[1:].reset_index(
    drop=True)  # Skip first row to align with lag features

# Ensure both datasets have the same length and reset indices
min_length = min(len(df_train_clean), len(weather_data_clean))
df_train_final = df_train_clean.iloc[:min_length].reset_index(drop=True)
weather_data_final = weather_data_clean.iloc[:min_length].reset_index(
    drop=True)

print(f"Final training data shape: {df_train_final.shape}")
print(f"Final weather data shape: {weather_data_final.shape}")

# Try different model configurations
model_configs = [
    ((1, 1, 1), (1, 1, 1, 12)),  # Original
    ((0, 1, 1), (0, 1, 1, 12)),  # Simpler
    ((1, 1, 0), (1, 1, 0, 12)),  # No MA terms
    ((2, 1, 2), (1, 1, 1, 12)),  # More complex
]

best_aic = float('inf')
best_model = None
best_config = None

for order, seasonal_order in model_configs:
    try:
        model = SARIMAX(
            df_train_final['Monthly_Total_Sales'],
            exog=weather_data_final[[
                'Max_Temp', 'Min_Temp', 'Humidity', 'Wind_Speed', 'temp_avg']],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False, maxiter=100)

        if results.aic < best_aic:
            best_aic = results.aic
            best_model = results
            best_config = (order, seasonal_order)
            print(f"New best AIC: {best_aic:.2f} with config: {best_config}")

    except Exception as e:
        print(f"Failed with config {order}, {seasonal_order}: {e}")
        continue

print(f"\nBest configuration: {best_config}")
print(f"Best AIC: {best_aic:.2f}")

# Use the best model for predictions
if best_model is not None:
    results = best_model
    print("\nBest Model Summary:")
    print(results.summary())
else:
    print("No valid model found. Using fallback approach...")
    # Fallback to simple model without lag features
    df_train_simple = df_train_aligned.dropna().reset_index(drop=True)
    weather_data_simple = weather_data_train_aligned.reset_index(drop=True)

    # Ensure same length
    min_len = min(len(df_train_simple), len(weather_data_simple))
    df_train_simple = df_train_simple.iloc[:min_len].reset_index(drop=True)
    weather_data_simple = weather_data_simple.iloc[:min_len].reset_index(
        drop=True)

    model = SARIMAX(
        df_train_simple['Monthly_Total_Sales'],
        exog=weather_data_simple[['Max_Temp',
                                  'Min_Temp', 'Humidity', 'Wind_Speed']],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    print("\nFallback Model Summary:")
    print(results.summary())

# Prepare test data with same features
weather_data_test_aligned = weather_data_test.reset_index(drop=True)
weather_data_test_aligned['temp_avg'] = (
    weather_data_test_aligned['Max_Temp'] + weather_data_test_aligned['Min_Temp']) / 2

# Make predictions
if best_model is not None:
    pred = results.get_forecast(
        steps=len(df_test['Monthly_Total_Sales']),
        exog=weather_data_test_aligned[[
            'Max_Temp', 'Min_Temp', 'Humidity', 'Wind_Speed', 'temp_avg']][:6]
    )
else:
    pred = results.get_forecast(
        steps=len(df_test['Monthly_Total_Sales']),
        exog=weather_data_test_aligned[[
            'Max_Temp', 'Min_Temp', 'Humidity', 'Wind_Speed']][:6]
    )
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int()

print("\nPredicted Mean Sales:")
print(pred_mean)
print("\nPredicted Confidence Interval:")
print(pred_ci)

print("\nActual Sales:")
actual_sales = df_test['Monthly_Total_Sales'].values
print(actual_sales)

# Calculate evaluation metrics
print("\nModel Evaluation:")
mae = mean_absolute_error(actual_sales, pred_mean.values)
mape = mean_absolute_percentage_error(actual_sales, pred_mean.values) * 100

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Calculate percentage improvement
original_predictions = np.array(
    [14294.69, 15389.50, 19755.58, 49061.95, 24860.06, 12253.96])
original_mae = mean_absolute_error(actual_sales, original_predictions)
original_mape = mean_absolute_percentage_error(
    actual_sales, original_predictions) * 100

print(f"\nComparison with original model:")
print(f"Original MAE: {original_mae:.2f}")
print(f"Improved MAE: {mae:.2f}")
print(f"MAE Improvement: {((original_mae - mae) / original_mae * 100):.1f}%")

print(f"Original MAPE: {original_mape:.2f}%")
print(f"Improved MAPE: {mape:.2f}%")
print(
    f"MAPE Improvement: {((original_mape - mape) / original_mape * 100):.1f}%")
