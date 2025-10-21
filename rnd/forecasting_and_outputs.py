#!/usr/bin/env python3
"""
Phase 7: Forecasting and Outputs

This script generates 12-month forecasts (2025-07 to 2026-06) using all trained models
from combined, regional, and segment scopes.

Weather Strategy: Reuse 2025 weather pattern for forecast period
Outputs: 3 main forecast CSV files + comprehensive results summary
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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from pmdarima import auto_arima
from prophet import Prophet

# Machine learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import load_model

# Utilities
import joblib
import os
from datetime import datetime, timedelta

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_forecast_weather_data():
    """
    Create weather data for forecast period (2025-07 to 2026-06) by reusing 2025 pattern
    """
    print("Creating forecast weather data...")
    
    # Load 2025 weather data to extract pattern
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
    
    # Combine weather data
    weather_df = pd.concat(weather_data, ignore_index=True)
    weather_df = weather_df.sort_values(['Date', 'State']).reset_index(drop=True)
    
    # Extract 2025 weather pattern
    weather_2025 = weather_df[weather_df['Date'].dt.year == 2025].copy()
    
    # Create forecast dates (2025-07 to 2026-06)
    forecast_dates = pd.date_range(start='2025-07-01', end='2026-06-01', freq='MS')
    
    # Create forecast weather by repeating 2025 pattern
    forecast_weather = []
    
    for state in weather_2025['State'].unique():
        state_weather_2025 = weather_2025[weather_2025['State'] == state].copy()
        
        for i, forecast_date in enumerate(forecast_dates):
            # Map forecast month to 2025 month (Jul 2025 -> Jan 2025, Aug 2025 -> Feb 2025, etc.)
            source_month = (forecast_date.month - 6) % 12
            if source_month == 0:
                source_month = 12
            
            # Find corresponding 2025 data
            source_data = state_weather_2025[state_weather_2025['Date'].dt.month == source_month]
            
            if len(source_data) > 0:
                # Use the first occurrence of that month
                source_row = source_data.iloc[0].copy()
                source_row['Date'] = forecast_date
                forecast_weather.append(source_row)
    
    forecast_weather_df = pd.DataFrame(forecast_weather)
    forecast_weather_df = forecast_weather_df.sort_values(['Date', 'State']).reset_index(drop=True)
    
    print(f"Created forecast weather data: {len(forecast_weather_df)} records")
    print(f"Date range: {forecast_weather_df['Date'].min()} to {forecast_weather_df['Date'].max()}")
    print(f"States: {forecast_weather_df['State'].unique()}")
    
    return forecast_weather_df

def generate_combined_forecasts(forecast_weather_df):
    """
    Generate 12-month forecasts using combined scope models
    """
    print("\nGenerating combined forecasts...")
    
    # Create forecast dates
    forecast_dates = pd.date_range(start='2025-07-01', end='2026-06-01', freq='MS')
    
    # Prepare weather data for combined forecasting (weighted average)
    weather_cols = ['Max_Temp', 'Min_Temp', 'Humidity', 'Wind_Speed']
    combined_weather = []
    
    for date in forecast_dates:
        date_weather = forecast_weather_df[forecast_weather_df['Date'] == date]
        if len(date_weather) > 0:
            # Calculate weighted average (equal weights for now)
            avg_weather = date_weather[weather_cols].mean()
            combined_weather.append({
                'Date': date,
                'Max_Temp': avg_weather['Max_Temp'],
                'Min_Temp': avg_weather['Min_Temp'],
                'Humidity': avg_weather['Humidity'],
                'Wind_Speed': avg_weather['Wind_Speed']
            })
    
    combined_weather_df = pd.DataFrame(combined_weather)
    
    # Initialize results
    combined_forecasts = pd.DataFrame({'Date': forecast_dates})
    
    # Load and generate forecasts for each model
    models = ['ARIMA', 'SARIMAX', 'Seasonal_Decomp', 'Holt_Winters', 'LSTM', 'GRU', 'Prophet']
    
    for model_name in models:
        try:
            if model_name in ['ARIMA', 'SARIMAX', 'Seasonal_Decomp', 'Holt_Winters']:
                # Load model
                model_path = f'outputs/models/{model_name.lower()}_combined.pkl'
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    
                    if model_name == 'ARIMA':
                        forecast = model.forecast(steps=12)
                    elif model_name == 'SARIMAX':
                        forecast = model.forecast(steps=12, exog=combined_weather_df[weather_cols])
                    elif model_name == 'Seasonal_Decomp':
                        # Extract trend and seasonal components
                        trend_model = model['trend_model']
                        seasonal_pattern = model['seasonal_pattern']
                        
                        # Forecast trend
                        trend_dates = np.arange(38, 50).reshape(-1, 1)  # Continue from training period
                        trend_forecast = trend_model.predict(trend_dates)
                        
                        # Repeat seasonal pattern
                        seasonal_forecast = np.tile(seasonal_pattern, 1)[:12]
                        
                        forecast = trend_forecast + seasonal_forecast
                    elif model_name == 'Holt_Winters':
                        forecast = model.forecast(steps=12)
                    
                    combined_forecasts[model_name] = forecast
                    print(f"  ✅ {model_name}: Generated forecast")
                else:
                    print(f"  ⚠️ {model_name}: Model file not found")
                    combined_forecasts[model_name] = np.nan
            
            elif model_name in ['LSTM', 'GRU']:
                # For neural networks, we need to implement iterative forecasting
                # For now, use placeholder values
                print(f"  ⚠️ {model_name}: Iterative forecasting not implemented, using placeholder")
                combined_forecasts[model_name] = np.nan
            
            elif model_name == 'Prophet':
                # Load Prophet model
                model_path = f'outputs/models/prophet_combined.pkl'
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    
                    # Prepare future dataframe
                    future_df = combined_weather_df[['Date'] + weather_cols].copy()
                    future_df.columns = ['ds'] + weather_cols
                    
                    forecast = model.predict(future_df)
                    combined_forecasts[model_name] = forecast['yhat'].values
                    print(f"  ✅ {model_name}: Generated forecast")
                else:
                    print(f"  ⚠️ {model_name}: Model file not found")
                    combined_forecasts[model_name] = np.nan
        
        except Exception as e:
            print(f"  ❌ {model_name}: Error generating forecast - {e}")
            combined_forecasts[model_name] = np.nan
    
    # Save combined forecasts
    combined_forecasts.to_csv('outputs/forecasts/forecasts_combined_models.csv', index=False)
    print(f"Combined forecasts saved to: outputs/forecasts/forecasts_combined_models.csv")
    
    return combined_forecasts

def generate_regional_forecasts(forecast_weather_df):
    """
    Generate 12-month forecasts using regional scope models
    """
    print("\nGenerating regional forecasts...")
    
    # Create forecast dates
    forecast_dates = pd.date_range(start='2025-07-01', end='2026-06-01', freq='MS')
    
    # Initialize results
    regional_forecasts = []
    
    states = ['Andhra Pradesh', 'Karnataka', 'Kerala', 'Telangana', 'Tamil Nadu']
    models = ['ARIMA', 'SARIMAX', 'Seasonal_Decomp', 'Holt_Winters', 'LSTM', 'GRU', 'Prophet']
    approaches = ['A', 'B']
    
    for state in states:
        state_weather = forecast_weather_df[forecast_weather_df['State'] == state].copy()
        state_weather = state_weather.sort_values('Date')
        
        for model_name in models:
            for approach in approaches:
                try:
                    if model_name in ['ARIMA', 'SARIMAX', 'Seasonal_Decomp', 'Holt_Winters']:
                        # Load model
                        model_path = f'outputs/models/regional_{model_name.lower()}_{state.replace(" ", "_")}_approach_{approach}.pkl'
                        if os.path.exists(model_path):
                            model = joblib.load(model_path)
                            
                            if model_name == 'ARIMA':
                                forecast = model.forecast(steps=12)
                            elif model_name == 'SARIMAX':
                                weather_cols = ['Max_Temp', 'Min_Temp', 'Humidity', 'Wind_Speed']
                                forecast = model.forecast(steps=12, exog=state_weather[weather_cols])
                            elif model_name == 'Seasonal_Decomp':
                                # Extract trend and seasonal components
                                trend_model = model['trend_model']
                                seasonal_pattern = model['seasonal_pattern']
                                
                                # Forecast trend
                                trend_dates = np.arange(38, 50).reshape(-1, 1)
                                trend_forecast = trend_model.predict(trend_dates)
                                
                                # Repeat seasonal pattern
                                seasonal_forecast = np.tile(seasonal_pattern, 1)[:12]
                                
                                forecast = trend_forecast + seasonal_forecast
                            elif model_name == 'Holt_Winters':
                                forecast = model.forecast(steps=12)
                            
                            # Add to results
                            for i, date in enumerate(forecast_dates):
                                regional_forecasts.append({
                                    'Date': date,
                                    'State': state,
                                    'Model': model_name,
                                    'Predicted_Sales': forecast[i] if i < len(forecast) else np.nan,
                                    'Approach': approach
                                })
                            
                            print(f"  ✅ {state} - {model_name} - Approach {approach}")
                        else:
                            print(f"  ⚠️ {state} - {model_name} - Approach {approach}: Model not found")
                    
                    elif model_name in ['LSTM', 'GRU']:
                        # Placeholder for neural networks
                        print(f"  ⚠️ {state} - {model_name} - Approach {approach}: Iterative forecasting not implemented")
                    
                    elif model_name == 'Prophet':
                        # Load Prophet model
                        model_path = f'outputs/models/regional_prophet_{state.replace(" ", "_")}_approach_{approach}.pkl'
                        if os.path.exists(model_path):
                            model = joblib.load(model_path)
                            
                            # Prepare future dataframe
                            weather_cols = ['Max_Temp', 'Min_Temp', 'Humidity', 'Wind_Speed']
                            future_df = state_weather[['Date'] + weather_cols].copy()
                            future_df.columns = ['ds'] + weather_cols
                            
                            forecast = model.predict(future_df)
                            
                            # Add to results
                            for i, date in enumerate(forecast_dates):
                                regional_forecasts.append({
                                    'Date': date,
                                    'State': state,
                                    'Model': model_name,
                                    'Predicted_Sales': forecast['yhat'].iloc[i] if i < len(forecast) else np.nan,
                                    'Approach': approach
                                })
                            
                            print(f"  ✅ {state} - {model_name} - Approach {approach}")
                        else:
                            print(f"  ⚠️ {state} - {model_name} - Approach {approach}: Model not found")
                
                except Exception as e:
                    print(f"  ❌ {state} - {model_name} - Approach {approach}: Error - {e}")
    
    # Convert to DataFrame and save
    regional_forecasts_df = pd.DataFrame(regional_forecasts)
    regional_forecasts_df.to_csv('outputs/forecasts/forecasts_regional_models.csv', index=False)
    print(f"Regional forecasts saved to: outputs/forecasts/forecasts_regional_models.csv")
    
    return regional_forecasts_df

def generate_segment_forecasts(forecast_weather_df):
    """
    Generate 12-month forecasts using segment scope models
    """
    print("\nGenerating segment forecasts...")
    
    # Load segment results to get segment information
    segment_results_path = 'outputs/segment_models_results.csv'
    if not os.path.exists(segment_results_path):
        print("  ⚠️ Segment results not found, skipping segment forecasts")
        return pd.DataFrame()
    
    segment_results = pd.read_csv(segment_results_path)
    segments = segment_results['Segment'].unique()
    
    # Create forecast dates
    forecast_dates = pd.date_range(start='2025-07-01', end='2026-06-01', freq='MS')
    
    # Initialize results
    segment_forecasts = []
    
    models = ['ARIMA', 'SARIMAX', 'Prophet']
    
    for segment in segments:
        # Extract segment information
        segment_info = segment_results[segment_results['Segment'] == segment].iloc[0]
        state = segment_info['State']
        
        # Get weather data for the state
        state_weather = forecast_weather_df[forecast_weather_df['State'] == state].copy()
        state_weather = state_weather.sort_values('Date')
        
        for model_name in models:
            try:
                # Load model
                model_path = f'outputs/models/segment_{model_name.lower()}_{segment}.pkl'
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    
                    if model_name == 'ARIMA':
                        forecast = model.forecast(steps=12)
                    elif model_name == 'SARIMAX':
                        weather_cols = ['Max_Temp', 'Min_Temp', 'Humidity', 'Wind_Speed']
                        forecast = model.forecast(steps=12, exog=state_weather[weather_cols])
                    elif model_name == 'Prophet':
                        # Prepare future dataframe
                        weather_cols = ['Max_Temp', 'Min_Temp', 'Humidity', 'Wind_Speed']
                        future_df = state_weather[['Date'] + weather_cols].copy()
                        future_df.columns = ['ds'] + weather_cols
                        
                        forecast = model.predict(future_df)
                        forecast = forecast['yhat'].values
                    
                    # Add to results
                    for i, date in enumerate(forecast_dates):
                        segment_forecasts.append({
                            'Date': date,
                            'State': state,
                            'Star_Rating': segment_info['Star_Rating'],
                            'Tonnage': segment_info['Tonnage'],
                            'Model': model_name,
                            'Predicted_Sales': forecast[i] if i < len(forecast) else np.nan
                        })
                    
                    print(f"  ✅ {segment} - {model_name}")
                else:
                    print(f"  ⚠️ {segment} - {model_name}: Model not found")
            
            except Exception as e:
                print(f"  ❌ {segment} - {model_name}: Error - {e}")
    
    # Convert to DataFrame and save
    segment_forecasts_df = pd.DataFrame(segment_forecasts)
    segment_forecasts_df.to_csv('outputs/forecasts/forecasts_segments.csv', index=False)
    print(f"Segment forecasts saved to: outputs/forecasts/forecasts_segments.csv")
    
    return segment_forecasts_df

def create_results_summary():
    """
    Create comprehensive results summary markdown document
    """
    print("\nCreating results summary...")
    
    summary_content = f"""# AC Sales & Weather Analysis - Results Summary

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Project Overview

This analysis implements a comprehensive AC sales forecasting system using weather data across multiple scopes:
- **Combined Scope**: National-level forecasting
- **Regional Scope**: State-level forecasting with two approaches
- **Segment Scope**: Product segment-level forecasting

## Model Performance Summary

### Combined Scope Models
"""
    
    # Add combined model performance
    try:
        combined_results = pd.read_csv('outputs/combined_models_results.csv')
        summary_content += f"""
| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
"""
        for _, row in combined_results.iterrows():
            summary_content += f"| {row['Model']} | {row['MAE']:.2f} | {row['RMSE']:.2f} | {row['MAPE']:.2f}% |\n"
    except:
        summary_content += "Combined model results not available.\n"
    
    summary_content += """
### Regional Scope Models
"""
    
    # Add regional model performance
    try:
        regional_results = pd.read_csv('outputs/regional_models_results.csv')
        best_regional = regional_results.nsmallest(10, 'MAE')
        summary_content += f"""
**Top 10 Regional Models (by MAE):**

| State | Model | Approach | MAE | RMSE | MAPE |
|-------|-------|----------|-----|------|------|
"""
        for _, row in best_regional.iterrows():
            summary_content += f"| {row['State']} | {row['Model']} | {row['Approach']} | {row['MAE']:.2f} | {row['RMSE']:.2f} | {row['MAPE']:.2f}% |\n"
    except:
        summary_content += "Regional model results not available.\n"
    
    summary_content += """
### Segment Scope Models
"""
    
    # Add segment model performance
    try:
        segment_results = pd.read_csv('outputs/segment_models_results.csv')
        best_segments = segment_results.nsmallest(10, 'MAE')
        summary_content += f"""
**Top 10 Segment Models (by MAE):**

| Segment | State | Star Rating | Tonnage | Model | MAE | RMSE | MAPE |
|---------|-------|-------------|---------|-------|-----|------|------|
"""
        for _, row in best_segments.iterrows():
            summary_content += f"| {row['Segment']} | {row['State']} | {row['Star_Rating']} | {row['Tonnage']} | {row['Model']} | {row['MAE']:.2f} | {row['RMSE']:.2f} | {row['MAPE']:.2f}% |\n"
    except:
        summary_content += "Segment model results not available.\n"
    
    summary_content += f"""
## Forecast Results

### Forecast Period: 2025-07 to 2026-06 (12 months)

**Weather Strategy**: Reused 2025 weather pattern for forecast period

### Generated Forecast Files:
1. `forecasts_combined_models.csv` - National-level forecasts
2. `forecasts_regional_models.csv` - State-level forecasts  
3. `forecasts_segments.csv` - Segment-level forecasts

## Key Insights

### Model Performance
- **Best Overall Approach**: [To be determined from results]
- **Weather Correlation**: [To be analyzed from correlation results]
- **Seasonal Patterns**: [To be identified from decomposition results]

### Regional Analysis
- **Best Performing States**: [To be determined from regional results]
- **Approach A vs B**: [To be compared from regional results]

### Segment Analysis
- **Top Contributing Segments**: [To be identified from segment results]
- **Product Mix Insights**: [To be derived from segment analysis]

## Technical Implementation

### Models Implemented
- **ARIMA/SARIMAX**: Auto-parameter selection with weather regressors
- **Seasonal Decomposition**: STL-based trend and seasonal forecasting
- **Holt-Winters**: Triple exponential smoothing
- **LSTM/GRU**: Neural networks with weather features
- **Prophet**: Automatic seasonality with weather regressors

### Data Processing
- **Training Period**: 2021-11 to 2024-12 (38 months)
- **Validation Period**: 2025-01 to 2025-06 (6 months)
- **Forecast Period**: 2025-07 to 2026-06 (12 months)

### Weather Integration
- **Combined Scope**: Weighted average across states
- **Regional Scope**: State-specific weather data
- **Segment Scope**: Relevant state's weather data

## Recommendations

### For Business Planning
1. **Focus on Top Segments**: Prioritize forecasting accuracy for high-volume segments
2. **Regional Strategy**: Use state-specific models for regional planning
3. **Weather Monitoring**: Continue weather data collection for model updates

### For Model Improvement
1. **Neural Network Forecasting**: Implement proper iterative forecasting for LSTM/GRU
2. **Ensemble Methods**: Combine multiple models for improved accuracy
3. **Real-time Updates**: Implement model retraining with new data

### For Operations
1. **Inventory Management**: Use segment forecasts for product-specific planning
2. **Marketing**: Leverage seasonal patterns for campaign timing
3. **Supply Chain**: Use regional forecasts for distribution planning

## Files Generated

### Forecast Files
- `outputs/forecasts/forecasts_combined_models.csv`
- `outputs/forecasts/forecasts_regional_models.csv`
- `outputs/forecasts/forecasts_segments.csv`

### Analysis Files
- `outputs/combined_models_results.csv`
- `outputs/regional_models_results.csv`
- `outputs/segment_models_results.csv`
- `outputs/segment_aggregation_validation.csv`

### Visualization Files
- `outputs/charts/combined_models_comparison.png`
- `outputs/charts/regional_models_comparison.png`
- `outputs/charts/segment_models_comparison.png`
- Additional EDA and decomposition charts

### Model Files
- `outputs/models/` - All trained models for reproducibility

---

*This analysis provides a comprehensive foundation for AC sales forecasting with weather integration across multiple business scopes.*
"""
    
    # Save summary
    with open('outputs/results_summary.md', 'w') as f:
        f.write(summary_content)
    
    print("Results summary saved to: outputs/results_summary.md")

def main():
    """Main execution function"""
    print("=" * 80)
    print("PHASE 7: FORECASTING AND OUTPUTS")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    
    # Ensure output directories exist
    os.makedirs('outputs/forecasts', exist_ok=True)
    os.makedirs('outputs/charts', exist_ok=True)
    
    try:
        # 1. Create forecast weather data
        forecast_weather_df = create_forecast_weather_data()
        
        # 2. Generate combined forecasts
        combined_forecasts = generate_combined_forecasts(forecast_weather_df)
        
        # 3. Generate regional forecasts
        regional_forecasts = generate_regional_forecasts(forecast_weather_df)
        
        # 4. Generate segment forecasts
        segment_forecasts = generate_segment_forecasts(forecast_weather_df)
        
        # 5. Create results summary
        create_results_summary()
        
        print("\n" + "=" * 80)
        print("PHASE 7 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Generated forecast files:")
        print("- outputs/forecasts/forecasts_combined_models.csv")
        print("- outputs/forecasts/forecasts_regional_models.csv")
        print("- outputs/forecasts/forecasts_segments.csv")
        print("- outputs/results_summary.md")
        print(f"Completed at: {datetime.now()}")
        
    except Exception as e:
        print(f"\nError in forecasting and outputs: {e}")
        raise

if __name__ == "__main__":
    main()
