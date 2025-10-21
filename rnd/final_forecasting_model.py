"""
Final Sales Forecasting Model
============================

This script implements an improved SARIMAX model for sales forecasting with:
- Automatic parameter optimization
- Feature engineering with lag variables
- Model validation and evaluation
- Robust error handling
- Visualization of results

Author: AI Assistant
Date: 2025
"""

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from itertools import product
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')


class SalesForecastingModel:
    """A comprehensive sales forecasting model using SARIMAX"""

    def __init__(self, state='Tamil Nadu', star_rating='1 Star', tonnage=1.8):
        self.state = state
        self.star_rating = star_rating
        self.tonnage = tonnage
        self.model = None
        self.results = None
        self.best_params = None
        self.feature_names = None

    def load_sales_data(self):
        """Load sales data from CSV file"""
        try:
            with open('./data/monthly_sales_summary.csv', 'r', encoding='utf-8') as file:
                df = pd.read_csv(file)
            print(f"Loaded sales data: {df.shape[0]} rows")
            return df
        except FileNotFoundError:
            print("Sales data file not found!")
            return None

    def load_weather_data(self):
        """Load weather data from CSV file"""
        try:
            with open('./outputs/processed_weather_data/TN_weather_timeseries.csv', 'r', encoding='utf-8') as file:
                df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'])
            print(f"Loaded weather data: {df.shape[0]} rows")
            return df
        except FileNotFoundError:
            print("Weather data file not found!")
            return None

    def format_sales_data(self, data):
        """Filter and format sales data for specific criteria"""
        filtered_data = data[
            (data['State'] == self.state) &
            (data['Star_Rating'] == self.star_rating) &
            (data['Tonnage'] == self.tonnage)
        ]

        if len(filtered_data) == 0:
            print(
                f"No data found for {self.state}, {self.star_rating}, {self.tonnage}")
            return None

        # Create time series data
        response_data = filtered_data[[
            'Year', 'Month', 'Monthly_Total_Sales']].copy()
        response_data['Date'] = pd.to_datetime(
            response_data[['Year', 'Month']].assign(day=1))
        response_data = response_data.sort_values(
            'Date').reset_index(drop=True)
        response_data = response_data[['Date', 'Monthly_Total_Sales']]

        print(f"Formatted sales data: {len(response_data)} months")
        return response_data

    def create_features(self, sales_df, weather_df):
        """Create engineered features for the model"""
        # Merge sales and weather data
        merged_data = pd.merge(sales_df, weather_df, on='Date', how='inner')

        # Create lag features for sales
        for lag in [1, 2, 3, 12]:
            merged_data[f'sales_lag_{lag}'] = merged_data['Monthly_Total_Sales'].shift(
                lag)

        # Create weather aggregations
        merged_data['temp_avg'] = (
            merged_data['Max_Temp'] + merged_data['Min_Temp']) / 2
        merged_data['temp_range'] = merged_data['Max_Temp'] - \
            merged_data['Min_Temp']

        # Create lag features for weather
        weather_cols = ['Max_Temp', 'Min_Temp',
                        'Humidity', 'Wind_Speed', 'temp_avg']
        for col in weather_cols:
            for lag in [1, 12]:
                merged_data[f'{col}_lag_{lag}'] = merged_data[col].shift(lag)

        # Create rolling statistics
        for window in [3, 6]:
            merged_data[f'sales_rolling_mean_{window}'] = merged_data['Monthly_Total_Sales'].rolling(
                window=window).mean()
            merged_data[f'sales_rolling_std_{window}'] = merged_data['Monthly_Total_Sales'].rolling(
                window=window).std()

        print(f"Created features: {merged_data.shape[1]} columns")
        return merged_data

    def check_stationarity(self, timeseries):
        """Check if the time series is stationary using ADF test"""
        result = adfuller(timeseries.dropna())
        is_stationary = result[1] <= 0.05

        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        print(
            f"Series is {'stationary' if is_stationary else 'not stationary'}")

        return is_stationary

    def find_best_parameters(self, endog, exog, max_p=2, max_d=1, max_q=2, max_P=1, max_D=1, max_Q=1):
        """Find the best SARIMAX parameters using grid search"""
        print("Searching for best SARIMAX parameters...")

        best_aic = float('inf')
        best_params = None
        best_model = None

        # Generate parameter combinations
        p_values = range(0, max_p + 1)
        d_values = range(0, max_d + 1)
        q_values = range(0, max_q + 1)
        P_values = range(0, max_P + 1)
        D_values = range(0, max_D + 1)
        Q_values = range(0, max_Q + 1)

        total_combinations = len(p_values) * len(d_values) * \
            len(q_values) * len(P_values) * len(D_values) * len(Q_values)
        current_combination = 0

        for p, d, q, P, D, Q in product(p_values, d_values, q_values, P_values, D_values, Q_values):
            current_combination += 1

            try:
                model = SARIMAX(
                    endog,
                    exog=exog,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fitted_model = model.fit(disp=False, maxiter=100)

                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_params = (p, d, q, P, D, Q)
                    best_model = fitted_model
                    print(
                        f"  New best AIC: {best_aic:.2f} with params: {best_params}")

            except Exception:
                continue

        if best_params is None:
            print("No valid parameters found, using defaults")
            best_params = (1, 1, 1, 1, 1, 1)
            best_model = None

        print(f"Best parameters: {best_params}")
        print(f"Best AIC: {best_aic:.2f}")

        return best_params, best_model

    def prepare_data(self, merged_data, test_split_date='2024-12-31'):
        """Prepare training and test datasets"""
        # Split data
        train_data = merged_data.query(f'Date < "{test_split_date}"').dropna()
        test_data = merged_data.query(f'Date >= "{test_split_date}"')

        if len(train_data) == 0:
            print("No training data available!")
            return None, None, None, None

        # Select features
        weather_features = ['Max_Temp', 'Min_Temp',
                            'Humidity', 'Wind_Speed', 'temp_avg', 'temp_range']
        lag_features = [col for col in train_data.columns if 'lag_' in col]
        rolling_features = [
            col for col in train_data.columns if 'rolling_' in col]

        # Use a subset of features to avoid overfitting
        selected_features = weather_features + \
            lag_features[:6] + rolling_features[:4]
        selected_features = [
            col for col in selected_features if col in train_data.columns]

        self.feature_names = selected_features

        print(f"Training data: {train_data.shape[0]} rows")
        print(f"Test data: {test_data.shape[0]} rows")
        print(f"Selected features: {len(selected_features)}")

        return train_data, test_data, selected_features

    def fit_model(self, train_data, selected_features):
        """Fit the SARIMAX model"""
        print("Fitting SARIMAX model...")

        # Find best parameters
        best_params, best_model = self.find_best_parameters(
            train_data['Monthly_Total_Sales'],
            train_data[selected_features]
        )

        self.best_params = best_params

        # If no best model found, create a new one
        if best_model is None:
            model = SARIMAX(
                train_data['Monthly_Total_Sales'],
                exog=train_data[selected_features],
                order=best_params[:3],
                seasonal_order=best_params[3:] + (12,),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.results = model.fit(disp=False, maxiter=200)
        else:
            self.results = best_model

        print("Model fitted successfully")
        return self.results

    def make_predictions(self, test_data, selected_features, steps=None):
        """Make predictions on test data"""
        if steps is None:
            steps = len(test_data)

        print(f"Making predictions for {steps} periods...")

        # Prepare test features
        test_features = test_data[selected_features].fillna(
            method='ffill').fillna(method='bfill')

        # Make predictions
        pred = self.results.get_forecast(
            steps=steps, exog=test_features.iloc[:steps])
        pred_mean = pred.predicted_mean
        pred_ci = pred.conf_int()

        return pred_mean, pred_ci

    def evaluate_model(self, y_true, y_pred):
        """Calculate and display evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        print("\nModel Evaluation:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  MSE:  {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")

        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}

    def plot_results(self, train_data, test_data, pred_mean, pred_ci, save_path=None):
        """Create and save visualization of results"""
        plt.figure(figsize=(14, 8))

        # Plot training data
        plt.plot(train_data['Date'], train_data['Monthly_Total_Sales'],
                 label='Training Data', color='blue', linewidth=2)

        # Plot actual test data
        if len(test_data) > 0:
            plt.plot(test_data['Date'], test_data['Monthly_Total_Sales'],
                     label='Actual', color='green', marker='o', linewidth=2, markersize=6)

        # Plot predictions
        plt.plot(test_data['Date'], pred_mean,
                 label='Predicted', color='red', marker='s', linewidth=2, markersize=6)

        # Plot confidence interval
        plt.fill_between(test_data['Date'], pred_ci.iloc[:, 0], pred_ci.iloc[:, 1],
                         alpha=0.3, color='red', label='95% Confidence Interval')

        plt.title(f'Sales Forecasting - {self.state} ({self.star_rating}, {self.tonnage} Ton)',
                  fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Monthly Total Sales', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()

    def save_model(self, filepath):
        """Save the trained model to a file"""
        model_data = {
            'results': self.results,
            'best_params': self.best_params,
            'feature_names': self.feature_names,
            'state': self.state,
            'star_rating': self.star_rating,
            'tonnage': self.tonnage,
            'timestamp': datetime.now()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {filepath}")

    def run_full_pipeline(self):
        """Run the complete forecasting pipeline"""
        print("Starting Sales Forecasting Pipeline")
        print("=" * 50)

        # Load data
        sales_data = self.load_sales_data()
        weather_data = self.load_weather_data()

        if sales_data is None or weather_data is None:
            return None

        # Format sales data
        sales_formatted = self.format_sales_data(sales_data)
        if sales_formatted is None:
            return None

        # Create features
        merged_data = self.create_features(sales_formatted, weather_data)

        # Check stationarity
        print("\nChecking stationarity...")
        self.check_stationarity(merged_data['Monthly_Total_Sales'])

        # Prepare data
        train_data, test_data, selected_features = self.prepare_data(
            merged_data)
        print("test_data:")
        print(test_data.head())
        if train_data is None:
            return None

        # Fit model
        self.fit_model(train_data, selected_features)

        # Make predictions
        if len(test_data) > 0:
            pred_mean, pred_ci = self.make_predictions(
                test_data, selected_features)

            # Evaluate model
            actual_values = test_data['Monthly_Total_Sales'].values
            predicted_values = pred_mean.values

            # Remove NaN values for evaluation
            mask = ~(np.isnan(actual_values) | np.isnan(predicted_values))
            if np.any(mask):
                metrics = self.evaluate_model(
                    actual_values[mask], predicted_values[mask])

            # Display results
            print("\nPrediction Results:")
            print("Predicted Sales:")
            for i, (date, pred_val) in enumerate(zip(test_data['Date'], pred_mean)):
                actual_val = test_data['Monthly_Total_Sales'].iloc[i] if i < len(
                    test_data) else 'N/A'
                print(
                    f"  {date.strftime('%Y-%m')}: {pred_val:.0f} (Actual: {actual_val})")

            # Create visualization
            self.plot_results(train_data, test_data, pred_mean, pred_ci,
                              './outputs/charts/final_forecast_plot.png')

            # Save model
            self.save_model(
                './outputs/models/final_sales_forecasting_model.pkl')

            return {
                'predictions': pred_mean,
                'confidence_intervals': pred_ci,
                'metrics': metrics if 'metrics' in locals() else None,
                'model': self.results
            }
        else:
            print("No test data available for evaluation")
            return None


def main():
    """Main function to run the forecasting model"""
    # Initialize the model
    forecaster = SalesForecastingModel(
        state='Tamil Nadu',
        star_rating='1 Star',
        tonnage=1.8
    )

    # Run the complete pipeline
    results = forecaster.run_full_pipeline()

    if results:
        print("\nForecasting pipeline completed successfully!")
    else:
        print("\nForecasting pipeline failed!")


if __name__ == "__main__":
    main()
