#!/usr/bin/env python3
"""
Phase 6: Segment Models Analysis

This script implements segment-level forecasting models for the top-performing 
State-Star_Rating-Tonnage combinations.

Models: ARIMA, SARIMAX, Prophet
Expected: 10-15 segments × 3 models = 30-45 models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Time series and forecasting
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# Handle pmdarima import gracefully
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    print("Warning: pmdarima not available, will use alternative ARIMA parameter selection")
    PMDARIMA_AVAILABLE = False

# Utilities
import joblib
import os
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_identify_top_segments():
    """
    Load sales data and identify top segments by sales volume
    """
    print("Loading sales data and identifying top segments...")
    
    # Load sales data
    sales_df = pd.read_csv('data/monthly_sales_summary.csv')
    sales_df['Date'] = pd.to_datetime(
        sales_df['Year'].astype(str) + '-' + 
        sales_df['Month'].astype(str) + '-01'
    )
    
    print(f"Total sales records: {len(sales_df)}")
    print(f"Date range: {sales_df['Date'].min()} to {sales_df['Date'].max()}")
    print(f"States: {sales_df['State'].unique()}")
    print(f"Star Ratings: {sales_df['Star_Rating'].unique()}")
    print(f"Tonnage: {sales_df['Tonnage'].unique()}")
    
    # Calculate total sales by segment
    segment_sales = sales_df.groupby(['State', 'Star_Rating', 'Tonnage']).agg({
        'Monthly_Total_Sales': 'sum',
        'Number_of_Transactions': 'sum'
    }).reset_index()
    
    # Sort by total sales
    segment_sales = segment_sales.sort_values('Monthly_Total_Sales', ascending=False)
    
    # Calculate cumulative percentage
    total_sales = segment_sales['Monthly_Total_Sales'].sum()
    segment_sales['Cumulative_Sales'] = segment_sales['Monthly_Total_Sales'].cumsum()
    segment_sales['Cumulative_Percentage'] = (segment_sales['Cumulative_Sales'] / total_sales) * 100
    
    # Identify top segments (70-80% of sales)
    top_segments = segment_sales[segment_sales['Cumulative_Percentage'] <= 80].copy()
    
    print(f"\nTop {len(top_segments)} segments represent {top_segments['Cumulative_Percentage'].iloc[-1]:.1f}% of total sales")
    print(f"Total sales in top segments: {top_segments['Monthly_Total_Sales'].sum():,.0f}")
    print(f"Total sales overall: {total_sales:,.0f}")
    
    print("\nTop 15 Segments by Sales Volume:")
    print("=" * 80)
    print(top_segments.head(15)[['State', 'Star_Rating', 'Tonnage', 'Monthly_Total_Sales', 'Cumulative_Percentage']])
    
    return sales_df, segment_sales, top_segments

def load_weather_data():
    """
    Load weather data for all states
    """
    print("\nLoading weather data...")
    
    weather_data = []
    states = ['AP', 'KA', 'KL', 'TL', 'TN']
    state_mapping = {
        'AP': 'Andhra Pradesh',
        'KA': 'Karnataka', 
        'KL': 'Kerala',
        'TL': 'Telangana',
        'TN': 'Tamil Nadu'
    }
    
    for state_code in states:
        file_path = f'outputs/processed_weather_data/{state_code}_weather_timeseries.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df['State'] = state_mapping[state_code]
            weather_data.append(df)
            print(f"Loaded weather data for {state_mapping[state_code]}: {len(df)} records")
        else:
            print(f"Warning: Weather file not found for {state_code}")
    
    # Combine weather data
    weather_df = pd.concat(weather_data, ignore_index=True)
    weather_df = weather_df.sort_values(['Date', 'State']).reset_index(drop=True)
    
    return weather_df

def prepare_segment_data(sales_df, weather_df, top_segments):
    """
    Prepare time series data for each top segment
    """
    print("\nPreparing segment data...")
    
    segment_data = {}
    
    # Define date ranges
    train_end = pd.to_datetime('2024-12-01')
    val_end = pd.to_datetime('2025-06-01')
    
    for idx, segment in top_segments.iterrows():
        state = segment['State']
        star_rating = segment['Star_Rating']
        tonnage = segment['Tonnage']
        
        # Filter sales data for this segment
        segment_sales = sales_df[
            (sales_df['State'] == state) & 
            (sales_df['Star_Rating'] == star_rating) & 
            (sales_df['Tonnage'] == tonnage)
        ].copy()
        
        if len(segment_sales) == 0:
            print(f"No data found for segment: {state} - {star_rating} - {tonnage}")
            continue
        
        # Sort by date
        segment_sales = segment_sales.sort_values('Date')
        
        # Split into train/validation
        train_data = segment_sales[segment_sales['Date'] <= train_end].copy()
        val_data = segment_sales[(segment_sales['Date'] > train_end) & (segment_sales['Date'] <= val_end)].copy()
        
        if len(train_data) < 12 or len(val_data) == 0:
            print(f"Insufficient data for segment: {state} - {star_rating} - {tonnage}")
            continue
        
        # Prepare time series
        train_ts = train_data.set_index('Date')['Monthly_Total_Sales']
        val_ts = val_data.set_index('Date')['Monthly_Total_Sales']
        
        # Get weather data for the state
        state_weather = weather_df[weather_df['State'] == state].copy()
        state_weather = state_weather.sort_values('Date')
        
        # Merge weather with sales data
        train_merged = train_data.merge(state_weather, on='Date', how='inner')
        val_merged = val_data.merge(state_weather, on='Date', how='inner')
        
        # Prepare weather features
        weather_cols = ['Max_Temp', 'Min_Temp', 'Humidity', 'Wind_Speed']
        train_weather = train_merged.set_index('Date')[weather_cols]
        val_weather = val_merged.set_index('Date')[weather_cols]
        
        segment_key = f"{state}_{star_rating}_{tonnage}".replace(' ', '_')
        
        segment_data[segment_key] = {
            'state': state,
            'star_rating': star_rating,
            'tonnage': tonnage,
            'train_ts': train_ts,
            'val_ts': val_ts,
            'train_weather': train_weather,
            'val_weather': val_weather,
            'train_data': train_merged,
            'val_data': val_merged,
            'weather_cols': weather_cols
        }
        
        print(f"Prepared data for {segment_key}: Train {len(train_ts)} months, Val {len(val_ts)} months")
    
    print(f"\nSuccessfully prepared data for {len(segment_data)} segments")
    return segment_data

def fit_segment_arima(train_ts, val_ts, segment_key):
    """Fit ARIMA model for segment"""
    try:
        if PMDARIMA_AVAILABLE:
            model = auto_arima(
                train_ts, seasonal=True, m=12,
                max_p=3, max_q=3, max_P=2, max_Q=2,
                suppress_warnings=True, stepwise=True, error_action='ignore'
            )
            order = model.order
            seasonal_order = model.seasonal_order
        else:
            # Use default parameters when pmdarima is not available
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 12)
        
        fitted_model = ARIMA(train_ts, order=order, seasonal_order=seasonal_order).fit()
        predictions = fitted_model.forecast(steps=len(val_ts))
        
        # Save model
        joblib.dump(fitted_model, f'outputs/models/segment_arima_{segment_key}.pkl')
        
        return predictions, fitted_model
    except Exception as e:
        print(f"Error fitting ARIMA for {segment_key}: {e}")
        return None, None

def fit_segment_sarimax(train_ts, train_weather, val_ts, val_weather, segment_key):
    """Fit SARIMAX model for segment with weather regressors"""
    try:
        if PMDARIMA_AVAILABLE:
            model = auto_arima(
                train_ts, exogenous=train_weather, seasonal=True, m=12,
                max_p=3, max_q=3, max_P=2, max_Q=2,
                suppress_warnings=True, stepwise=True, error_action='ignore'
            )
            order = model.order
            seasonal_order = model.seasonal_order
        else:
            # Use default parameters when pmdarima is not available
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 12)
        
        fitted_model = SARIMAX(
            train_ts, exog=train_weather, order=order, seasonal_order=seasonal_order
        ).fit(disp=False)
        
        predictions = fitted_model.forecast(steps=len(val_ts), exog=val_weather)
        
        # Save model
        joblib.dump(fitted_model, f'outputs/models/segment_sarimax_{segment_key}.pkl')
        
        return predictions, fitted_model
    except Exception as e:
        print(f"Error fitting SARIMAX for {segment_key}: {e}")
        return None, None

def fit_segment_prophet(train_data, val_data, segment_key):
    """Fit Prophet model for segment with weather regressors"""
    try:
        weather_cols = ['Max_Temp', 'Min_Temp', 'Humidity', 'Wind_Speed']
        prophet_train = train_data[['Date', 'Monthly_Total_Sales'] + weather_cols].copy()
        prophet_train.columns = ['ds', 'y'] + weather_cols
        
        model = Prophet(
            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
            seasonality_mode='additive'
        )
        
        for col in weather_cols:
            model.add_regressor(col)
        
        model.fit(prophet_train)
        
        prophet_val = val_data[['Date'] + weather_cols].copy()
        prophet_val.columns = ['ds'] + weather_cols
        
        forecast = model.predict(prophet_val)
        prophet_pred = forecast['yhat'].values
        
        # Save model
        joblib.dump(model, f'outputs/models/segment_prophet_{segment_key}.pkl')
        
        return prophet_pred
    except Exception as e:
        print(f"Error fitting Prophet for {segment_key}: {e}")
        return None

def train_all_segment_models(segment_data):
    """Train all models for each segment"""
    print("\nTraining segment models...")
    print("=" * 60)
    
    segment_results = {}
    
    for segment_key, data in segment_data.items():
        print(f"\nTraining models for {segment_key}...")
        
        train_ts = data['train_ts']
        val_ts = data['val_ts']
        train_weather = data['train_weather']
        val_weather = data['val_weather']
        train_data = data['train_data']
        val_data = data['val_data']
        
        segment_predictions = {}
        
        # 1. ARIMA
        print(f"  - ARIMA...")
        arima_pred, _ = fit_segment_arima(train_ts, val_ts, segment_key)
        segment_predictions['ARIMA'] = arima_pred
        
        # 2. SARIMAX
        print(f"  - SARIMAX...")
        sarimax_pred, _ = fit_segment_sarimax(train_ts, train_weather, val_ts, val_weather, segment_key)
        segment_predictions['SARIMAX'] = sarimax_pred
        
        # 3. Prophet
        print(f"  - Prophet...")
        prophet_pred = fit_segment_prophet(train_data, val_data, segment_key)
        segment_predictions['Prophet'] = prophet_pred
        
        segment_results[segment_key] = {
            'predictions': segment_predictions,
            'actual': val_ts.values,
            'dates': val_ts.index,
            'state': data['state'],
            'star_rating': data['star_rating'],
            'tonnage': data['tonnage']
        }
        
        print(f"  ✅ Completed {segment_key}")
    
    print("\n" + "=" * 60)
    print("Segment model training completed!")
    return segment_results

def evaluate_segment_models(segment_results):
    """Evaluate all segment models and return performance metrics"""
    print("\nEvaluating segment models...")
    
    all_results = []
    
    for segment_key, data in segment_results.items():
        predictions = data['predictions']
        actual = data['actual']
        state = data['state']
        star_rating = data['star_rating']
        tonnage = data['tonnage']
        
        for model_name, pred in predictions.items():
            if pred is not None and len(pred) == len(actual):
                # Calculate metrics
                mae = np.mean(np.abs(pred - actual))
                rmse = np.sqrt(np.mean((pred - actual) ** 2))
                mape = np.mean(np.abs((actual - pred) / actual)) * 100
                
                all_results.append({
                    'Segment': segment_key,
                    'State': state,
                    'Star_Rating': star_rating,
                    'Tonnage': tonnage,
                    'Model': model_name,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape
                })
    
    return pd.DataFrame(all_results)

def create_segment_comparison_visualizations(segment_results_df):
    """Create segment model comparison visualizations"""
    print("\nCreating segment comparison visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Segment Models Performance Comparison', fontsize=16, y=0.98)
    
    # 1. Model performance comparison
    model_performance = segment_results_df.groupby('Model')[['MAE', 'RMSE', 'MAPE']].mean()
    model_performance.plot(kind='bar', ax=axes[0, 0], width=0.8)
    axes[0, 0].set_title('Average Performance by Model')
    axes[0, 0].set_ylabel('Metric Value')
    axes[0, 0].legend(['MAE', 'RMSE', 'MAPE'])
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Performance by state
    state_performance = segment_results_df.groupby(['State', 'Model'])['MAE'].mean().unstack()
    state_performance.plot(kind='bar', ax=axes[0, 1], width=0.8)
    axes[0, 1].set_title('Performance by State and Model')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend(['ARIMA', 'Prophet', 'SARIMAX'])
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Performance by star rating
    star_performance = segment_results_df.groupby(['Star_Rating', 'Model'])['MAE'].mean().unstack()
    star_performance.plot(kind='bar', ax=axes[1, 0], width=0.8)
    axes[1, 0].set_title('Performance by Star Rating and Model')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].legend(['ARIMA', 'Prophet', 'SARIMAX'])
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Performance by tonnage
    tonnage_performance = segment_results_df.groupby(['Tonnage', 'Model'])['MAE'].mean().unstack()
    tonnage_performance.plot(kind='bar', ax=axes[1, 1], width=0.8)
    axes[1, 1].set_title('Performance by Tonnage and Model')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].legend(['ARIMA', 'Prophet', 'SARIMAX'])
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/charts/segment_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Segment comparison chart saved to: outputs/charts/segment_models_comparison.png")

def validate_segment_aggregation(segment_results, sales_df, segment_results_df):
    """
    Validate that segment forecasts aggregate correctly to regional totals
    """
    print("\nValidating segment aggregation...")
    
    # Calculate actual regional totals for validation period
    val_start = pd.to_datetime('2025-01-01')
    val_end = pd.to_datetime('2025-06-01')
    
    actual_regional = sales_df[
        (sales_df['Date'] >= val_start) & (sales_df['Date'] <= val_end)
    ].groupby(['Date', 'State'])['Monthly_Total_Sales'].sum().reset_index()
    
    # Aggregate segment forecasts by state
    forecast_regional = {}
    
    for segment_key, data in segment_results.items():
        state = data['state']
        predictions = data['predictions']
        dates = data['dates']
        
        # Use best model (lowest MAE) for each segment
        best_model = segment_results_df[segment_results_df['Segment'] == segment_key].nsmallest(1, 'MAE')['Model'].iloc[0]
        best_pred = predictions[best_model]
        
        if best_pred is not None:
            for i, date in enumerate(dates):
                if date not in forecast_regional:
                    forecast_regional[date] = {}
                if state not in forecast_regional[date]:
                    forecast_regional[date][state] = 0
                forecast_regional[date][state] += best_pred[i]
    
    # Convert to DataFrame
    forecast_data = []
    for date, states in forecast_regional.items():
        for state, total in states.items():
            forecast_data.append({'Date': date, 'State': state, 'Forecast_Total': total})
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Merge with actual
    comparison = actual_regional.merge(forecast_df, on=['Date', 'State'], how='inner')
    comparison['Error'] = comparison['Forecast_Total'] - comparison['Monthly_Total_Sales']
    comparison['Error_Pct'] = (comparison['Error'] / comparison['Monthly_Total_Sales']) * 100
    
    return comparison

def main():
    """Main execution function"""
    print("=" * 80)
    print("PHASE 6: SEGMENT MODELS ANALYSIS")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    
    # Ensure output directories exist
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/charts', exist_ok=True)
    
    try:
        # 1. Load data and identify top segments
        sales_df, segment_sales, top_segments = load_and_identify_top_segments()
        
        # 2. Load weather data
        weather_df = load_weather_data()
        
        # 3. Prepare segment data
        segment_data = prepare_segment_data(sales_df, weather_df, top_segments)
        
        # 4. Train all segment models
        segment_results = train_all_segment_models(segment_data)
        
        # 5. Evaluate all segment models
        segment_results_df = evaluate_segment_models(segment_results)
        
        print("\nSegment Model Performance Summary:")
        print("=" * 80)
        print(segment_results_df.groupby(['Model'])[['MAE', 'RMSE', 'MAPE']].mean().round(2))
        
        print("\nBest Model per Segment (by MAE):")
        print("=" * 80)
        best_segment_models = segment_results_df.loc[segment_results_df.groupby('Segment')['MAE'].idxmin()]
        print(best_segment_models[['Segment', 'State', 'Star_Rating', 'Tonnage', 'Model', 'MAE', 'RMSE', 'MAPE']])
        
        # 6. Create visualizations
        create_segment_comparison_visualizations(segment_results_df)
        
        # 7. Validate segment aggregation
        aggregation_validation = validate_segment_aggregation(segment_results, sales_df, segment_results_df)
        
        print("\nSegment Aggregation Validation:")
        print("=" * 80)
        print(f"Average absolute error: {np.mean(np.abs(aggregation_validation['Error'])):.2f}")
        print(f"Average error percentage: {np.mean(np.abs(aggregation_validation['Error_Pct'])):.2f}%")
        print(f"RMSE: {np.sqrt(np.mean(aggregation_validation['Error']**2)):.2f}")
        
        print("\nValidation by State:")
        print(aggregation_validation.groupby('State')[['Error', 'Error_Pct']].mean().round(2))
        
        # 8. Save detailed results
        segment_results_df.to_csv('outputs/segment_models_results.csv', index=False)
        aggregation_validation.to_csv('outputs/segment_aggregation_validation.csv', index=False)
        
        print("\nDetailed results saved to:")
        print("- outputs/segment_models_results.csv")
        print("- outputs/segment_aggregation_validation.csv")
        
        print("\n" + "=" * 80)
        print("PHASE 6 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Total segments analyzed: {len(segment_data)}")
        print(f"Total models trained: {len(segment_results_df)}")
        print(f"Models saved to: outputs/models/segment_*")
        print(f"Results saved to: outputs/segment_models_results.csv")
        print(f"Charts saved to: outputs/charts/segment_models_comparison.png")
        print(f"Completed at: {datetime.now()}")
        
    except Exception as e:
        print(f"\nError in segment models analysis: {e}")
        raise

if __name__ == "__main__":
    main()
