"""
Comprehensive Dashboard Generator for Branch and Regional Managers

This module generates two distinct interactive dashboards:
1. Branch Manager Dashboard: Tactical/operational focus
2. Regional Manager Dashboard: Strategic/comparative focus
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import math
from pathlib import Path


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_data():
    """Load and preprocess sales and weather data"""
    print("Loading data...")
    
    # Load sales data
    sales_df = pd.read_csv('./data/merged_filter_ingestion.csv')
    sales_df['Date'] = pd.to_datetime(sales_df['Date'], errors='coerce')
    sales_df['Quantity'] = pd.to_numeric(sales_df['Quantity'], errors='coerce').fillna(0)
    sales_df['Tonnage'] = pd.to_numeric(sales_df['Tonnage'], errors='coerce').fillna(0)
    
    # Add time-based columns
    sales_df['Year'] = sales_df['Date'].dt.year
    sales_df['Month'] = sales_df['Date'].dt.month
    sales_df['Day'] = sales_df['Date'].dt.day
    sales_df['Week'] = sales_df['Date'].dt.isocalendar().week
    sales_df['Weekday'] = sales_df['Date'].dt.day_name()
    sales_df['WeekdayNum'] = sales_df['Date'].dt.dayofweek
    
    # Load weather data
    weather_df = pd.read_csv('./data/weather_data.csv')
    weather_df['Date'] = pd.to_datetime(weather_df['Date'], errors='coerce')
    
    # Merge sales and weather data
    # First, aggregate weather to daily level (if needed)
    weather_daily = weather_df.groupby(['Date', 'Branch']).agg({
        'Min Temp': 'mean',
        'Max Temp': 'mean',
        'Avg Temp': 'mean',
        'Min Humidity': 'mean',
        'Max Humidity': 'mean',
        'Avg Humidity': 'mean',
        'Min Wind Speed': 'mean',
        'Max Wind Speed': 'mean',
        'Avg Wind Speed': 'mean'
    }).reset_index()
    
    # Merge on Date and Branch
    merged_df = sales_df.merge(
        weather_daily,
        on=['Date', 'Branch'],
        how='left'
    )
    
    print(f"Sales data: {len(sales_df)} records")
    print(f"Weather data: {len(weather_df)} records")
    print(f"Merged data: {len(merged_df)} records")
    print(f"Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
    print(f"Branches: {sorted(merged_df['Branch'].unique())}")
    
    return merged_df, sales_df, weather_df


# =============================================================================
# FORECASTING FUNCTIONS (from FormulaWeight/1.ipynb)
# =============================================================================

def get_financial_year(dt):
    """Get financial year for a date"""
    if pd.isnull(dt):
        return None
    return dt.year + 1 if dt.month >= 4 else dt.year


def calculate_growth_factor(df: pd.DataFrame):
    """
    Calculate growth factor as average of last 3 financial years YoY growth.
    Expects columns 'Date' and 'Quantity' in df.
    """
    df = df.copy()
    df['FinancialYear'] = df['Date'].apply(get_financial_year)
    fy_branch_summary = df.groupby(['Branch', 'FinancialYear'], as_index=False)['Quantity'].sum()
    fy_branch_summary = fy_branch_summary.sort_values(['Branch', 'FinancialYear'])
    fy_branch_summary['YoY_Growth'] = fy_branch_summary.groupby('Branch')['Quantity'].pct_change()

    last_3_yoy = fy_branch_summary['YoY_Growth'].dropna()[-3:] if len(fy_branch_summary['YoY_Growth'].dropna()) >= 3 else fy_branch_summary['YoY_Growth'].dropna()

    if last_3_yoy.empty:
        return 1.25  # Fallback
    avg_growth_percent = last_3_yoy.mean()
    return 1 + avg_growth_percent


def generate_forecasts(df: pd.DataFrame, train_from_date: str = '2021-09-01', forecast_months: int = 12):
    """
    Generate forecasts for all branches using seasonal decomposition and growth factors.
    Returns a DataFrame with forecast dates and quantities per branch.
    """
    df = df.copy()
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
    df = df[df['Date'] >= pd.Timestamp(train_from_date)]

    # Group by date + branch and sum quantities
    grouped = (
        df.groupby(['Date', 'Branch'])['Quantity']
        .sum()
        .reset_index()
        .sort_values('Date')
    )

    branches = grouped['Branch'].unique()
    forecast_rows = []

    for branch in branches:
        branch_data = grouped[grouped['Branch'] == branch].copy()
        if branch_data.empty:
            continue

        # Aggregate by month
        branch_data['year'] = branch_data['Date'].dt.year
        branch_data['month'] = branch_data['Date'].dt.month
        monthly = (
            branch_data.groupby(['year', 'month'])['Quantity']
            .sum()
            .reset_index()
        )
        monthly['Date'] = pd.to_datetime(
            monthly[['year', 'month']].assign(day=1)
        )
        monthly = monthly.sort_values('Date')

        if monthly.empty:
            continue

        last_date = monthly['Date'].max()

        # --- Compute Seasonal Indices ---
        monthly_by_month = (
            monthly.groupby('month')['Quantity']
            .apply(list)
            .to_dict()
        )

        seasonal_indices = {
            month: sum(vals) / len(vals) if len(vals) > 0 else 1
            for month, vals in monthly_by_month.items()
        }
        overall_avg = sum(seasonal_indices.values()) / len(seasonal_indices) if seasonal_indices else 1
        seasonal_indices = {
            m: (v / overall_avg if overall_avg > 0 else 1)
            for m, v in seasonal_indices.items()
        }

        # --- Trend averages ---
        recent_monthly = monthly.tail(12)
        avg_recent_qty = recent_monthly['Quantity'].mean() if not recent_monthly.empty else 0

        # Calculate growth rate
        growth_rate = calculate_growth_factor(branch_data)

        forecast_start = last_date + pd.DateOffset(months=1)
        
        for i in range(forecast_months):
            forecast_date = forecast_start + pd.DateOffset(months=i)
            month = forecast_date.month
            months_from_last = (forecast_date - last_date).days // 30 if forecast_date > last_date else 0
            growth_factor = math.pow(growth_rate, months_from_last / 12)
            seasonal = seasonal_indices.get(month, 1)
            forecast_qty = avg_recent_qty * seasonal * growth_factor

            forecast_rows.append({
                "Branch": branch,
                "Date": forecast_date.strftime("%Y-%m-%d"),
                "Forecast Quantity": round(forecast_qty, 1),
            })
    
    return pd.DataFrame(forecast_rows)


# =============================================================================
# HELPER FUNCTIONS FOR CALCULATIONS
# =============================================================================

def calculate_mom_growth(df, branch=None):
    """Calculate Month-over-Month growth"""
    if branch:
        df = df[df['Branch'] == branch]
    
    monthly = df.groupby([df['Date'].dt.to_period('M').dt.to_timestamp(), 'Branch'])['Quantity'].sum().reset_index()
    monthly = monthly.sort_values(['Branch', 'Date'])
    monthly['MoM_Growth'] = monthly.groupby('Branch')['Quantity'].pct_change() * 100
    return monthly


def calculate_yoy_growth(df):
    """Calculate Year-over-Year growth by branch"""
    df = df.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    yearly = df.groupby(['Year', 'Month', 'Branch'])['Quantity'].sum().reset_index()
    yearly = yearly.sort_values(['Branch', 'Year', 'Month'])
    
    # Calculate YoY for same month
    yearly['YoY_Growth'] = None
    for branch in yearly['Branch'].unique():
        branch_data = yearly[yearly['Branch'] == branch].copy()
        for month in branch_data['Month'].unique():
            month_data = branch_data[branch_data['Month'] == month].sort_values('Year')
            if len(month_data) > 1:
                prev_qty = month_data.iloc[-2]['Quantity']
                curr_qty = month_data.iloc[-1]['Quantity']
                if prev_qty > 0:
                    growth = ((curr_qty - prev_qty) / prev_qty) * 100
                    yearly.loc[(yearly['Branch'] == branch) & 
                              (yearly['Month'] == month) & 
                              (yearly['Year'] == month_data.iloc[-1]['Year']), 'YoY_Growth'] = growth
    
    return yearly


def detect_anomalies(df, branch=None, window=30):
    """Detect sales anomalies using rolling statistics"""
    if branch:
        df = df[df['Branch'] == branch]
    
    daily = df.groupby(['Date', 'Branch'])['Quantity'].sum().reset_index()
    daily = daily.sort_values(['Branch', 'Date'])
    
    daily['Rolling_Mean'] = daily.groupby('Branch')['Quantity'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    daily['Rolling_Std'] = daily.groupby('Branch')['Quantity'].transform(lambda x: x.rolling(window=window, min_periods=1).std())
    daily['Z_Score'] = (daily['Quantity'] - daily['Rolling_Mean']) / (daily['Rolling_Std'] + 1e-6)
    daily['Is_Anomaly'] = abs(daily['Z_Score']) > 2
    
    return daily


def calculate_weather_correlation(df, branch=None):
    """Calculate correlation between sales and weather metrics"""
    if branch:
        df = df[df['Branch'] == branch]
    
    # Aggregate to daily level
    daily = df.groupby(['Date', 'Branch']).agg({
        'Quantity': 'sum',
        'Avg Temp': 'mean',
        'Avg Humidity': 'mean',
        'Max Temp': 'mean',
        'Min Temp': 'mean'
    }).reset_index()
    
    correlations = []
    for branch_code in daily['Branch'].unique():
        branch_data = daily[daily['Branch'] == branch_code]
        if len(branch_data) > 1:
            for col in ['Avg Temp', 'Avg Humidity', 'Max Temp', 'Min Temp']:
                if col in branch_data.columns:
                    corr = branch_data[['Quantity', col]].corr().iloc[0, 1]
                    if not pd.isna(corr):
                        correlations.append({
                            'Branch': branch_code,
                            'Weather_Metric': col,
                            'Correlation': corr
                        })
    
    return pd.DataFrame(correlations)


def find_best_weather_window(df, branch=None):
    """Find optimal temperature and humidity range for sales"""
    if branch:
        df = df[df['Branch'] == branch]
    
    daily = df.groupby(['Date', 'Branch']).agg({
        'Quantity': 'sum',
        'Avg Temp': 'mean',
        'Avg Humidity': 'mean'
    }).reset_index()
    
    # Bin temperature and humidity
    daily['Temp_Bin'] = pd.cut(daily['Avg Temp'], bins=10)
    daily['Humidity_Bin'] = pd.cut(daily['Avg Humidity'], bins=10)
    
    # Find bins with highest average sales
    temp_performance = daily.groupby('Temp_Bin')['Quantity'].mean().sort_values(ascending=False)
    humidity_performance = daily.groupby('Humidity_Bin')['Quantity'].mean().sort_values(ascending=False)
    
    best_temp_range = str(temp_performance.index[0])
    best_humidity_range = str(humidity_performance.index[0])
    
    return {
        'best_temp_range': best_temp_range,
        'best_humidity_range': best_humidity_range,
        'temp_details': temp_performance.to_dict(),
        'humidity_details': humidity_performance.to_dict()
    }


# =============================================================================
# BRANCH MANAGER DASHBOARD - CHART GENERATION FUNCTIONS
# =============================================================================

def chart_daily_weekly_sales_trend(df, branch):
    """Daily/Weekly Sales Trend Line Chart for Branch Manager"""
    branch_df = df[df['Branch'] == branch].copy()
    
    # Daily aggregation
    daily = branch_df.groupby('Date')['Quantity'].sum().reset_index()
    daily = daily.sort_values('Date')
    
    # Weekly aggregation
    weekly = branch_df.groupby([branch_df['Date'].dt.to_period('W').dt.start_time, 'Branch'])['Quantity'].sum().reset_index()
    weekly.columns = ['Date', 'Branch', 'Quantity']
    weekly = weekly.sort_values('Date')
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Sales Trend', 'Weekly Sales Trend'),
        vertical_spacing=0.12
    )
    
    # Daily trend
    fig.add_trace(
        go.Scatter(
            x=daily['Date'],
            y=daily['Quantity'],
            mode='lines+markers',
            name='Daily Sales',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Weekly trend
    fig.add_trace(
        go.Scatter(
            x=weekly['Date'],
            y=weekly['Quantity'],
            mode='lines+markers',
            name='Weekly Sales',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sales Quantity", row=1, col=1)
    fig.update_yaxes(title_text="Sales Quantity", row=2, col=1)
    
    fig.update_layout(
        title=f"Sales Trend - {branch} Branch",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def chart_mom_growth(df, branch):
    """Month-over-Month Growth Bar Chart"""
    monthly = calculate_mom_growth(df, branch)
    monthly = monthly[monthly['Branch'] == branch].tail(12)  # Last 12 months
    
    colors = ['green' if x > 0 else 'red' for x in monthly['MoM_Growth'].fillna(0)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=monthly['Date'],
            y=monthly['MoM_Growth'],
            marker_color=colors,
            text=[f"{x:.1f}%" for x in monthly['MoM_Growth']],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f"Month-over-Month Growth - {branch}",
        xaxis_title="Month",
        yaxis_title="Growth %",
        height=400
    )
    
    return fig


def chart_sales_target_vs_actual(df, branch, target_multiplier=1.1):
    """Sales Target vs Actual Gauge/Bullet Chart"""
    # Calculate current month sales
    current_month = datetime.now().replace(day=1)
    branch_df = df[df['Branch'] == branch].copy()
    current_sales = branch_df[branch_df['Date'] >= current_month]['Quantity'].sum()
    
    # Calculate previous month for target estimation
    prev_month_start = (current_month - pd.DateOffset(months=1))
    prev_month_sales = branch_df[
        (branch_df['Date'] >= prev_month_start) & 
        (branch_df['Date'] < current_month)
    ]['Quantity'].sum()
    
    target = prev_month_sales * target_multiplier
    progress = min(100, (current_sales / target * 100) if target > 0 else 0)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=progress,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Target Progress - {branch}"},
        delta={'reference': 100},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig, {'current': current_sales, 'target': target, 'progress': progress}


def chart_sales_by_star_rating(df, branch):
    """Sales by Star Rating - Stacked Bar or Donut Chart"""
    branch_df = df[df['Branch'] == branch].copy()
    star_sales = branch_df.groupby('Star Rating')['Quantity'].sum().reset_index()
    star_sales = star_sales.sort_values('Quantity', ascending=False)
    
    fig = go.Figure(data=[
        go.Pie(
            labels=star_sales['Star Rating'],
            values=star_sales['Quantity'],
            hole=0.4,
            textinfo='label+percent',
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f"Sales by Star Rating - {branch}",
        height=400
    )
    
    return fig


def chart_sales_by_segment(df, branch):
    """Sales by Segment - Horizontal Bar Chart"""
    branch_df = df[df['Branch'] == branch].copy()
    segment_sales = branch_df.groupby('Segment')['Quantity'].sum().reset_index()
    segment_sales = segment_sales.sort_values('Quantity', ascending=True)
    
    fig = go.Figure(data=[
        go.Bar(
            y=segment_sales['Segment'],
            x=segment_sales['Quantity'],
            orientation='h',
            marker_color='steelblue',
            text=[f"{x:,.0f}" for x in segment_sales['Quantity']],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f"Sales by Segment - {branch}",
        xaxis_title="Sales Quantity",
        yaxis_title="Segment",
        height=300
    )
    
    return fig


def chart_tonnage_vs_quantity(df, branch):
    """Tonnage vs Quantity Correlation Scatter Plot"""
    branch_df = df[df['Branch'] == branch].copy()
    
    # Aggregate by Item Code and Tonnage
    item_data = branch_df.groupby(['Item Code', 'Tonnage']).agg({
        'Quantity': 'sum'
    }).reset_index()
    
    fig = go.Figure(data=[
        go.Scatter(
            x=item_data['Tonnage'],
            y=item_data['Quantity'],
            mode='markers',
            marker=dict(
                size=10,
                color=item_data['Quantity'],
                colorscale='Viridis',
                showscale=True
            ),
            text=item_data['Item Code'],
            hovertemplate='<b>%{text}</b><br>Tonnage: %{x}<br>Quantity: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f"Tonnage vs Quantity Correlation - {branch}",
        xaxis_title="Tonnage",
        yaxis_title="Total Quantity",
        height=400
    )
    
    return fig


def chart_top_item_codes(df, branch, top_n=10):
    """Top N Item Codes Bar Chart"""
    branch_df = df[df['Branch'] == branch].copy()
    item_sales = branch_df.groupby('Item Code')['Quantity'].sum().reset_index()
    item_sales = item_sales.sort_values('Quantity', ascending=False).head(top_n)
    
    fig = go.Figure(data=[
        go.Bar(
            x=item_sales['Item Code'],
            y=item_sales['Quantity'],
            marker_color='coral',
            text=[f"{x:,.0f}" for x in item_sales['Quantity']],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f"Top {top_n} Item Codes - {branch}",
        xaxis_title="Item Code",
        yaxis_title="Sales Quantity",
        height=400,
        xaxis={'tickangle': -45}
    )
    
    return fig


def chart_sales_vs_temperature(df, branch):
    """Sales vs Average Temperature - Scatter/Line Chart"""
    branch_df = df[df['Branch'] == branch].copy()
    daily = branch_df.groupby('Date').agg({
        'Quantity': 'sum',
        'Avg Temp': 'mean'
    }).reset_index().sort_values('Date')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=daily['Date'],
            y=daily['Quantity'],
            mode='lines+markers',
            name='Sales',
            line=dict(color='royalblue', width=2)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily['Date'],
            y=daily['Avg Temp'],
            mode='lines+markers',
            name='Avg Temperature',
            line=dict(color='orange', width=2)
        ),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Sales Quantity", secondary_y=False)
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
    
    fig.update_layout(
        title=f"Sales vs Temperature - {branch}",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def chart_sales_vs_humidity(df, branch):
    """Sales vs Humidity Scatter Plot"""
    branch_df = df[df['Branch'] == branch].copy()
    daily = branch_df.groupby('Date').agg({
        'Quantity': 'sum',
        'Avg Humidity': 'mean'
    }).reset_index()
    
    fig = go.Figure(data=[
        go.Scatter(
            x=daily['Avg Humidity'],
            y=daily['Quantity'],
            mode='markers',
            marker=dict(
                size=8,
                color=daily['Quantity'],
                colorscale='Blues',
                showscale=True
            ),
            text=daily['Date'].dt.strftime('%Y-%m-%d'),
            hovertemplate='<b>%{text}</b><br>Humidity: %{x:.1f}%<br>Sales: %{y:,.0f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f"Sales vs Humidity - {branch}",
        xaxis_title="Average Humidity (%)",
        yaxis_title="Sales Quantity",
        height=400
    )
    
    return fig


def chart_weather_adjusted_sales(df, branch):
    """Weather-Adjusted Sales Trend - Dual Axis Chart"""
    branch_df = df[df['Branch'] == branch].copy()
    daily = branch_df.groupby('Date').agg({
        'Quantity': 'sum',
        'Avg Temp': 'mean',
        'Avg Humidity': 'mean'
    }).reset_index().sort_values('Date')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=daily['Date'],
            y=daily['Quantity'],
            name='Sales',
            marker_color='steelblue'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily['Date'],
            y=daily['Avg Temp'],
            mode='lines',
            name='Temperature',
            line=dict(color='red', width=2)
        ),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Sales Quantity", secondary_y=False)
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
    
    fig.update_layout(
        title=f"Weather-Adjusted Sales Trend - {branch}",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def chart_sales_per_tonnage_efficiency(df, branch):
    """Sales per Tonnage Efficiency Bar Chart"""
    branch_df = df[df['Branch'] == branch].copy()
    
    # Group by Tonnage and calculate efficiency
    tonnage_efficiency = branch_df.groupby('Tonnage').agg({
        'Quantity': 'sum',
        'Tonnage': 'first'
    }).reset_index()
    tonnage_efficiency['Efficiency'] = tonnage_efficiency['Quantity'] / tonnage_efficiency['Tonnage']
    tonnage_efficiency = tonnage_efficiency.sort_values('Efficiency', ascending=False)
    
    fig = go.Figure(data=[
        go.Bar(
            x=tonnage_efficiency['Tonnage'].astype(str),
            y=tonnage_efficiency['Efficiency'],
            marker_color='teal',
            text=[f"{x:,.0f}" for x in tonnage_efficiency['Efficiency']],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f"Sales per Tonnage Efficiency - {branch}",
        xaxis_title="Tonnage",
        yaxis_title="Sales per Tonnage",
        height=350
    )
    
    return fig


def chart_weekday_sales_heatmap(df, branch):
    """Weekday Sales Distribution Heatmap"""
    branch_df = df[df['Branch'] == branch].copy()
    
    # Create weekday x month heatmap
    branch_df['Month'] = branch_df['Date'].dt.month
    branch_df['MonthName'] = branch_df['Date'].dt.strftime('%b')
    branch_df['WeekdayName'] = branch_df['Date'].dt.day_name()
    
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = branch_df.groupby(['WeekdayName', 'MonthName'])['Quantity'].sum().reset_index()
    
    # Pivot for heatmap
    pivot = heatmap_data.pivot(index='WeekdayName', columns='MonthName', values='Quantity')
    pivot = pivot.reindex(weekday_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='YlOrRd',
        text=pivot.values,
        texttemplate='%{text:.0f}',
        textfont={"size": 10},
        hovertemplate='Weekday: %{y}<br>Month: %{x}<br>Sales: %{z:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Weekday Sales Distribution - {branch}",
        xaxis_title="Month",
        yaxis_title="Weekday",
        height=400
    )
    
    return fig


def chart_forecast_with_confidence(df, branch, forecast_days=30):
    """7/30-Day Rolling Forecast with Confidence Bands"""
    branch_df = df[df['Branch'] == branch].copy()
    daily = branch_df.groupby('Date')['Quantity'].sum().reset_index().sort_values('Date')
    
    # Calculate rolling statistics for forecast
    daily['Rolling_Mean'] = daily['Quantity'].rolling(window=30, min_periods=1).mean()
    daily['Rolling_Std'] = daily['Quantity'].rolling(window=30, min_periods=1).std()
    
    # Generate forecast
    last_date = daily['Date'].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
    
    last_mean = daily['Rolling_Mean'].iloc[-1]
    last_std = daily['Rolling_Std'].iloc[-1]
    
    forecast_values = [last_mean] * forecast_days
    upper_bound = [last_mean + 1.96 * last_std] * forecast_days
    lower_bound = [max(0, last_mean - 1.96 * last_std)] * forecast_days
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=daily['Date'],
        y=daily['Quantity'],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='blue', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines',
        name='Forecast',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    # Confidence bands
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(reversed(forecast_dates)),
        y=list(upper_bound) + list(reversed(lower_bound)),
        fill='toself',
        fillcolor='rgba(0,255,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval'
    ))
    
    fig.update_layout(
        title=f"{forecast_days}-Day Sales Forecast - {branch}",
        xaxis_title="Date",
        yaxis_title="Sales Quantity",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def chart_anomaly_detection(df, branch):
    """Anomaly Detection with Highlighted Points"""
    daily = detect_anomalies(df, branch)
    
    fig = go.Figure()
    
    # Normal points
    normal = daily[~daily['Is_Anomaly']]
    fig.add_trace(go.Scatter(
        x=normal['Date'],
        y=normal['Quantity'],
        mode='markers',
        name='Normal',
        marker=dict(color='blue', size=6)
    ))
    
    # Anomaly points
    anomalies = daily[daily['Is_Anomaly']]
    fig.add_trace(go.Scatter(
        x=anomalies['Date'],
        y=anomalies['Quantity'],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=12, symbol='x')
    ))
    
    fig.update_layout(
        title=f"Anomaly Detection - {branch}",
        xaxis_title="Date",
        yaxis_title="Sales Quantity",
        height=400
    )
    
    return fig



# =============================================================================
# REGIONAL MANAGER DASHBOARD - CHART GENERATION FUNCTIONS
# =============================================================================

def chart_sales_by_branch(df):
    """Sales by Branch - Stacked Bar/Column Chart"""
    monthly = df.groupby([df['Date'].dt.to_period('M').dt.to_timestamp(), 'Branch'])['Quantity'].sum().reset_index()
    monthly.columns = ['Date', 'Branch', 'Quantity']
    monthly = monthly.sort_values(['Date', 'Branch'])
    
    # Pivot for stacked bar
    pivot = monthly.pivot(index='Date', columns='Branch', values='Quantity').fillna(0)
    
    fig = go.Figure()
    for branch in pivot.columns:
        fig.add_trace(go.Bar(
            x=pivot.index,
            y=pivot[branch],
            name=branch,
            hovertemplate=f'<b>{branch}</b><br>Date: %{{x}}<br>Sales: %{{y:,.0f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Sales by Branch - Regional Overview",
        xaxis_title="Month",
        yaxis_title="Sales Quantity",
        barmode='stack',
        height=500
    )
    
    return fig


def chart_yoy_growth_by_branch(df):
    """Year-over-Year Growth by Branch - Line or Bar Chart"""
    yearly = calculate_yoy_growth(df)
    
    # Get latest year's growth for each branch
    if len(yearly) == 0:
        return go.Figure()
    
    latest_year = yearly['Year'].max()
    latest_growth = yearly[yearly['Year'] == latest_year].groupby('Branch')['YoY_Growth'].mean().reset_index()
    latest_growth = latest_growth.sort_values('YoY_Growth', ascending=False)
    
    colors = ['green' if x > 0 else 'red' for x in latest_growth['YoY_Growth'].fillna(0)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=latest_growth['Branch'],
            y=latest_growth['YoY_Growth'],
            marker_color=colors,
            text=[f"{x:.1f}%" if not pd.isna(x) else "N/A" for x in latest_growth['YoY_Growth']],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f"Year-over-Year Growth by Branch ({latest_year})",
        xaxis_title="Branch",
        yaxis_title="YoY Growth %",
        height=400
    )
    
    return fig


def chart_cumulative_sales_trend(df):
    """Cumulative Sales Trend - Area Chart"""
    monthly = df.groupby([df['Date'].dt.to_period('M').dt.to_timestamp(), 'Branch'])['Quantity'].sum().reset_index()
    monthly.columns = ['Date', 'Branch', 'Quantity']
    monthly = monthly.sort_values(['Date', 'Branch'])
    
    # Calculate cumulative sales per branch
    monthly['Cumulative'] = monthly.groupby('Branch')['Quantity'].cumsum()
    
    # Pivot for area chart
    pivot = monthly.pivot(index='Date', columns='Branch', values='Cumulative').ffill().fillna(0)
    
    fig = go.Figure()
    for branch in pivot.columns:
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot[branch],
            fill='tonexty' if len([c for c in pivot.columns]) > 1 else 'tozeroy',
            mode='lines',
            name=branch,
            stackgroup='one'
        ))
    
    fig.update_layout(
        title="Cumulative Sales Trend - Regional Overview",
        xaxis_title="Date",
        yaxis_title="Cumulative Sales Quantity",
        height=500
    )
    
    return fig


def chart_branch_contribution_by_segment(df):
    """Branch Contribution to Each Segment - 100% Stacked Bar"""
    segment_branch = df.groupby(['Segment', 'Branch'])['Quantity'].sum().reset_index()
    
    # Calculate percentages
    segment_totals = segment_branch.groupby('Segment')['Quantity'].sum()
    segment_branch['Percentage'] = segment_branch.apply(
        lambda row: (row['Quantity'] / segment_totals[row['Segment']] * 100) if segment_totals[row['Segment']] > 0 else 0,
        axis=1
    )
    
    # Pivot for stacked bar
    pivot = segment_branch.pivot(index='Segment', columns='Branch', values='Percentage').fillna(0)
    
    fig = go.Figure()
    for branch in pivot.columns:
        fig.add_trace(go.Bar(
            x=pivot.index,
            y=pivot[branch],
            name=branch,
            text=[f"{x:.1f}%" for x in pivot[branch]],
            textposition='inside'
        ))
    
    fig.update_layout(
        title="Branch Contribution to Each Segment",
        xaxis_title="Segment",
        yaxis_title="Percentage Contribution (%)",
        barmode='stack',
        height=400
    )
    
    return fig


def chart_sales_mix_evolution(df):
    """Sales Mix Evolution - Stacked Area Chart"""
    monthly = df.groupby([df['Date'].dt.to_period('M').dt.to_timestamp(), 'Star Rating'])['Quantity'].sum().reset_index()
    monthly.columns = ['Date', 'Star Rating', 'Quantity']
    monthly = monthly.sort_values(['Date', 'Star Rating'])
    
    # Pivot for stacked area
    pivot = monthly.pivot(index='Date', columns='Star Rating', values='Quantity').fillna(0)
    
    fig = go.Figure()
    for rating in pivot.columns:
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot[rating],
            fill='tonexty',
            mode='lines',
            name=rating,
            stackgroup='one'
        ))
    
    fig.update_layout(
        title="Sales Mix Evolution by Star Rating",
        xaxis_title="Date",
        yaxis_title="Sales Quantity",
        height=450
    )
    
    return fig


def chart_top_branch_item_pairs(df, top_n=20):
    """Top Branch-Item Pairs - Tree Map"""
    branch_item = df.groupby(['Branch', 'Item Code'])['Quantity'].sum().reset_index()
    branch_item = branch_item.sort_values('Quantity', ascending=False).head(top_n)
    
    # Create hierarchical structure for treemap
    branch_item['Label'] = branch_item['Branch'] + ' - ' + branch_item['Item Code']
    
    fig = go.Figure(go.Treemap(
        labels=branch_item['Label'],
        values=branch_item['Quantity'],
        parents=[''] * len(branch_item),
        marker=dict(
            colorscale='Viridis',
            showscale=True
        ),
        texttemplate='%{label}<br>%{value:,.0f}',
        textposition='middle center',
        hovertemplate='<b>%{label}</b><br>Sales: %{value:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Branch-Item Pairs",
        height=500
    )
    
    return fig


def chart_weather_correlation_heatmap(df):
    """Weather-Sales Correlation Heatmap per Branch"""
    correlations = calculate_weather_correlation(df)
    
    if len(correlations) == 0:
        return go.Figure()
    
    # Pivot for heatmap
    pivot = correlations.pivot(index='Branch', columns='Weather_Metric', values='Correlation')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlBu',
        zmid=0,
        text=pivot.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        hovertemplate='Branch: %{y}<br>Metric: %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Weather-Sales Correlation Heatmap by Branch",
        xaxis_title="Weather Metric",
        yaxis_title="Branch",
        height=400
    )
    
    return fig


def chart_regional_weather_vs_sales(df):
    """Regional Weather vs Sales Scatter Plot"""
    daily = df.groupby(['Date', 'Branch']).agg({
        'Quantity': 'sum',
        'Avg Temp': 'mean',
        'Avg Humidity': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    branches = daily['Branch'].unique()
    colors = px.colors.qualitative.Set3
    
    for i, branch in enumerate(branches):
        branch_data = daily[daily['Branch'] == branch]
        if 'Avg Temp' in branch_data.columns and len(branch_data) > 0:
            fig.add_trace(go.Scatter(
                x=branch_data['Avg Temp'],
                y=branch_data['Quantity'],
                mode='markers',
                name=branch,
                marker=dict(
                    size=8,
                    color=colors[i % len(colors)],
                    opacity=0.7
                ),
                hovertemplate=f'<b>{branch}</b><br>Temp: %{{x:.1f}}°C<br>Sales: %{{y:,.0f}}<extra></extra>'
            ))
    
    fig.update_layout(
        title="Regional Weather vs Sales Scatter Plot",
        xaxis_title="Average Temperature (°C)",
        yaxis_title="Sales Quantity",
        height=500
    )
    
    return fig


def chart_branch_efficiency_index(df):
    """Branch Efficiency Index - Sales per Tonnage"""
    branch_efficiency = df.groupby('Branch').agg({
        'Quantity': 'sum',
        'Tonnage': 'sum'
    }).reset_index()
    branch_efficiency['Efficiency'] = branch_efficiency['Quantity'] / (branch_efficiency['Tonnage'] + 1e-6)
    branch_efficiency = branch_efficiency.sort_values('Efficiency', ascending=False)
    
    fig = go.Figure(data=[
        go.Bar(
            x=branch_efficiency['Branch'],
            y=branch_efficiency['Efficiency'],
            marker_color='teal',
            text=[f"{x:,.0f}" for x in branch_efficiency['Efficiency']],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Branch Efficiency Index (Sales per Tonnage)",
        xaxis_title="Branch",
        yaxis_title="Efficiency Index",
        height=400
    )
    
    return fig


def chart_regional_forecast(df, forecast_months=12):
    """Regional Forecast - Next 12 Months Line Chart"""
    forecasts = generate_forecasts(df, forecast_months=forecast_months)
    
    if len(forecasts) == 0:
        return go.Figure()
    
    forecasts['Date'] = pd.to_datetime(forecasts['Date'])
    
    # Aggregate by date
    forecast_agg = forecasts.groupby('Date')['Forecast Quantity'].sum().reset_index()
    
    fig = go.Figure(data=[
        go.Scatter(
            x=forecast_agg['Date'],
            y=forecast_agg['Forecast Quantity'],
            mode='lines+markers',
            name='Regional Forecast',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        )
    ])
    
    fig.update_layout(
        title=f"Regional Forecast - Next {forecast_months} Months",
        xaxis_title="Date",
        yaxis_title="Forecast Sales Quantity",
        height=450
    )
    
    return fig


def chart_forecast_deviation_by_branch(df):
    """Forecast Deviation by Branch - Compare Predicted vs Actual"""
    # Generate forecasts
    forecasts = generate_forecasts(df, forecast_months=1)  # Next month only
    
    if len(forecasts) == 0:
        return go.Figure()
    
    forecasts['Date'] = pd.to_datetime(forecasts['Date'])
    
    # Get actual sales for comparison month (if available)
    # For now, we'll compare forecast to last month's actual
    last_month = df['Date'].max().replace(day=1)
    actual = df[df['Date'] >= last_month].groupby('Branch')['Quantity'].sum().reset_index()
    actual.columns = ['Branch', 'Actual Quantity']
    
    # Merge
    forecast_agg = forecasts.groupby('Branch')['Forecast Quantity'].sum().reset_index()
    comparison = forecast_agg.merge(actual, on='Branch', how='left')
    comparison['Actual Quantity'] = comparison['Actual Quantity'].fillna(0)
    comparison['Deviation'] = ((comparison['Forecast Quantity'] - comparison['Actual Quantity']) / 
                              (comparison['Actual Quantity'] + 1e-6) * 100)
    
    colors = ['green' if abs(x) < 10 else 'orange' if abs(x) < 20 else 'red' for x in comparison['Deviation']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=comparison['Branch'],
            y=comparison['Deviation'],
            marker_color=colors,
            text=[f"{x:.1f}%" for x in comparison['Deviation']],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Forecast Deviation by Branch",
        xaxis_title="Branch",
        yaxis_title="Deviation %",
        height=400
    )
    
    return fig


# =============================================================================
# KPI CALCULATION FUNCTIONS
# =============================================================================

def calculate_branch_kpis(df, branch):
    """Calculate all KPIs for Branch Manager Dashboard"""
    branch_df = df[df['Branch'] == branch].copy()
    
    # Current month sales
    current_month = datetime.now().replace(day=1)
    current_sales = branch_df[branch_df['Date'] >= current_month]['Quantity'].sum()
    
    # Last year same month for growth
    last_year_month = current_month.replace(year=current_month.year - 1)
    ly_sales = branch_df[
        (branch_df['Date'] >= last_year_month) & 
        (branch_df['Date'] < last_year_month + pd.DateOffset(months=1))
    ]['Quantity'].sum()
    
    growth_ly = ((current_sales - ly_sales) / (ly_sales + 1e-6)) * 100 if ly_sales > 0 else 0
    
    # Average quantity per day (current month)
    days_in_month = (datetime.now() - current_month).days + 1
    avg_qty_per_day = current_sales / days_in_month if days_in_month > 0 else 0
    
    # Average temperature and humidity
    weather_data = branch_df[branch_df['Date'] >= current_month]
    avg_temp = weather_data['Avg Temp'].mean() if 'Avg Temp' in weather_data.columns and len(weather_data) > 0 else 0
    avg_humidity = weather_data['Avg Humidity'].mean() if 'Avg Humidity' in weather_data.columns and len(weather_data) > 0 else 0
    
    # Top performing SKU
    top_sku = branch_df.groupby('Item Code')['Quantity'].sum().sort_values(ascending=False).head(1)
    top_sku_name = top_sku.index[0] if len(top_sku) > 0 else "N/A"
    top_sku_qty = top_sku.values[0] if len(top_sku) > 0 else 0
    
    # Weather impact score (correlation with temp and humidity)
    daily = branch_df.groupby('Date').agg({
        'Quantity': 'sum',
        'Avg Temp': 'mean',
        'Avg Humidity': 'mean'
    }).reset_index()
    
    temp_corr = daily[['Quantity', 'Avg Temp']].corr().iloc[0, 1] if len(daily) > 1 and 'Avg Temp' in daily.columns else 0
    humidity_corr = daily[['Quantity', 'Avg Humidity']].corr().iloc[0, 1] if len(daily) > 1 and 'Avg Humidity' in daily.columns else 0
    weather_impact_score = abs(temp_corr) * 50 + abs(humidity_corr) * 50 if not pd.isna(temp_corr) and not pd.isna(humidity_corr) else 0
    
    return {
        'total_sales_month': current_sales,
        'growth_vs_ly': growth_ly,
        'avg_qty_per_day': avg_qty_per_day,
        'avg_temp': avg_temp,
        'avg_humidity': avg_humidity,
        'top_sku': top_sku_name,
        'top_sku_qty': top_sku_qty,
        'weather_impact_score': weather_impact_score
    }


def calculate_regional_kpis(df):
    """Calculate all KPIs for Regional Manager Dashboard"""
    # Total regional sales
    current_month = datetime.now().replace(day=1)
    total_regional_sales = df[df['Date'] >= current_month]['Quantity'].sum()
    
    # YoY Growth %
    last_year_month = current_month.replace(year=current_month.year - 1)
    ly_regional_sales = df[
        (df['Date'] >= last_year_month) & 
        (df['Date'] < last_year_month + pd.DateOffset(months=1))
    ]['Quantity'].sum()
    
    yoy_growth = ((total_regional_sales - ly_regional_sales) / (ly_regional_sales + 1e-6)) * 100 if ly_regional_sales > 0 else 0
    
    # Best branch by growth
    yearly = calculate_yoy_growth(df)
    latest_year = yearly['Year'].max() if len(yearly) > 0 else datetime.now().year
    branch_growth = yearly[yearly['Year'] == latest_year].groupby('Branch')['YoY_Growth'].mean().reset_index()
    best_branch = branch_growth.loc[branch_growth['YoY_Growth'].idxmax(), 'Branch'] if len(branch_growth) > 0 else "N/A"
    best_branch_growth = branch_growth['YoY_Growth'].max() if len(branch_growth) > 0 else 0
    
    # Most weather-resilient branch (lowest correlation with weather)
    correlations = calculate_weather_correlation(df)
    if len(correlations) > 0:
        branch_resilience = correlations.groupby('Branch')['Correlation'].mean().reset_index()
        branch_resilience['AbsCorrelation'] = branch_resilience['Correlation'].abs()
        branch_resilience = branch_resilience.sort_values('AbsCorrelation', ascending=True)
        most_resilient = branch_resilience.iloc[0]['Branch'] if len(branch_resilience) > 0 else "N/A"
    else:
        most_resilient = "N/A"
    
    # Top segment contributor
    segment_sales = df[df['Date'] >= current_month].groupby('Segment')['Quantity'].sum().reset_index()
    top_segment = segment_sales.loc[segment_sales['Quantity'].idxmax(), 'Segment'] if len(segment_sales) > 0 else "N/A"
    
    # Regional forecast next month
    forecasts = generate_forecasts(df, forecast_months=1)
    forecast_next_month = forecasts['Forecast Quantity'].sum() if len(forecasts) > 0 else 0
    
    return {
        'total_regional_sales': total_regional_sales,
        'yoy_growth': yoy_growth,
        'best_branch': best_branch,
        'best_branch_growth': best_branch_growth,
        'most_resilient_branch': most_resilient,
        'top_segment': top_segment,
        'forecast_next_month': forecast_next_month
    }


# =============================================================================
# DASHBOARD ASSEMBLY FUNCTIONS
# =============================================================================

def create_branch_manager_dashboard(df, branch):
    """Create complete Branch Manager Dashboard"""
    print(f"\nCreating Branch Manager Dashboard for {branch}...")
    
    # Calculate KPIs
    kpis = calculate_branch_kpis(df, branch)
    
    # Create all charts
    charts = {}
    
    print("  Generating Sales & Performance charts...")
    charts['daily_weekly_trend'] = chart_daily_weekly_sales_trend(df, branch)
    charts['mom_growth'] = chart_mom_growth(df, branch)
    charts['target_vs_actual'], target_info = chart_sales_target_vs_actual(df, branch)
    
    print("  Generating Product & Mix charts...")
    charts['sales_by_star'] = chart_sales_by_star_rating(df, branch)
    charts['sales_by_segment'] = chart_sales_by_segment(df, branch)
    charts['tonnage_vs_quantity'] = chart_tonnage_vs_quantity(df, branch)
    charts['top_items'] = chart_top_item_codes(df, branch)
    
    print("  Generating Weather charts...")
    charts['sales_vs_temp'] = chart_sales_vs_temperature(df, branch)
    charts['sales_vs_humidity'] = chart_sales_vs_humidity(df, branch)
    charts['weather_adjusted'] = chart_weather_adjusted_sales(df, branch)
    
    print("  Generating Efficiency charts...")
    charts['tonnage_efficiency'] = chart_sales_per_tonnage_efficiency(df, branch)
    charts['weekday_heatmap'] = chart_weekday_sales_heatmap(df, branch)
    
    print("  Generating Forecast & Alerts charts...")
    charts['forecast_30d'] = chart_forecast_with_confidence(df, branch, forecast_days=30)
    charts['anomaly_detection'] = chart_anomaly_detection(df, branch)
    
    # Get best weather window
    weather_window = find_best_weather_window(df, branch)
    
    return {
        'kpis': kpis,
        'charts': charts,
        'weather_window': weather_window,
        'target_info': target_info
    }


def create_regional_manager_dashboard(df):
    """Create complete Regional Manager Dashboard"""
    print("\nCreating Regional Manager Dashboard...")
    
    # Calculate KPIs
    kpis = calculate_regional_kpis(df)
    
    # Create all charts
    charts = {}
    
    print("  Generating Regional Sales Overview charts...")
    charts['sales_by_branch'] = chart_sales_by_branch(df)
    charts['yoy_growth'] = chart_yoy_growth_by_branch(df)
    charts['cumulative_trend'] = chart_cumulative_sales_trend(df)
    
    print("  Generating Segment & Product Strategy charts...")
    charts['branch_contribution'] = chart_branch_contribution_by_segment(df)
    charts['sales_mix'] = chart_sales_mix_evolution(df)
    charts['top_branch_items'] = chart_top_branch_item_pairs(df)
    
    print("  Generating Weather Impact charts...")
    charts['weather_correlation'] = chart_weather_correlation_heatmap(df)
    charts['weather_vs_sales'] = chart_regional_weather_vs_sales(df)
    
    print("  Generating Efficiency & Forecast charts...")
    charts['efficiency_index'] = chart_branch_efficiency_index(df)
    charts['regional_forecast'] = chart_regional_forecast(df)
    charts['forecast_deviation'] = chart_forecast_deviation_by_branch(df)
    
    return {
        'kpis': kpis,
        'charts': charts
    }


def generate_html_dashboard(dashboard_data, dashboard_type, branch=None):
    """Generate HTML file for dashboard"""
    if dashboard_type == 'branch':
        title = f"Branch Manager Dashboard - {branch}"
        filename = f"branch_manager_dashboard_{branch}.html"
    else:
        title = "Regional Manager Dashboard"
        filename = "regional_manager_dashboard.html"
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .dashboard-header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .kpi-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .kpi-value {{
            font-size: 32px;
            font-weight: bold;
            color: #3498db;
            margin: 10px 0;
        }}
        .kpi-label {{
            font-size: 14px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .chart-container {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c3e50;
        }}
        .section-header {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin: 30px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>{title}</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section-header">📊 Key Performance Indicators</div>
    <div class="kpi-container">
"""
    
    # Add KPI cards
    kpis = dashboard_data['kpis']
    if dashboard_type == 'branch':
        kpi_items = [
            ('Total Sales (Month)', f"{kpis['total_sales_month']:,.0f}"),
            ('Growth vs LY', f"{kpis['growth_vs_ly']:.1f}%"),
            ('Avg Qty/Day', f"{kpis['avg_qty_per_day']:.1f}"),
            ('Avg Temp', f"{kpis['avg_temp']:.1f}°C"),
            ('Avg Humidity', f"{kpis['avg_humidity']:.1f}%"),
            ('Top SKU', f"{kpis['top_sku']}<br>({kpis['top_sku_qty']:,.0f})"),
            ('Weather Impact', f"{kpis['weather_impact_score']:.1f}/100")
        ]
    else:
        kpi_items = [
            ('Total Regional Sales', f"{kpis['total_regional_sales']:,.0f}"),
            ('YoY Growth %', f"{kpis['yoy_growth']:.1f}%"),
            ('Best Branch', f"{kpis['best_branch']}<br>({kpis['best_branch_growth']:.1f}%)"),
            ('Most Resilient', kpis['most_resilient_branch']),
            ('Top Segment', kpis['top_segment']),
            ('Forecast Next Month', f"{kpis['forecast_next_month']:,.0f}")
        ]
    
    for label, value in kpi_items:
        html_content += f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
"""
    
    html_content += """
    </div>
"""
    
    # Add weather window widget for branch dashboard
    if dashboard_type == 'branch' and 'weather_window' in dashboard_data:
        html_content += f"""
    <div class="chart-container">
        <div class="chart-title">🌤️ Best Weather Window</div>
        <p><strong>Optimal Temperature Range:</strong> {dashboard_data['weather_window']['best_temp_range']}</p>
        <p><strong>Optimal Humidity Range:</strong> {dashboard_data['weather_window']['best_humidity_range']}</p>
    </div>
"""
    
    # Add charts
    charts = dashboard_data['charts']
    chart_titles = {
        'daily_weekly_trend': '📈 Daily/Weekly Sales Trend',
        'mom_growth': '📊 Month-over-Month Growth',
        'target_vs_actual': '🎯 Sales Target vs Actual',
        'sales_by_star': '⭐ Sales by Star Rating',
        'sales_by_segment': '📦 Sales by Segment',
        'tonnage_vs_quantity': '⚖️ Tonnage vs Quantity Correlation',
        'top_items': '🏆 Top 10 Item Codes',
        'sales_vs_temp': '🌡️ Sales vs Temperature',
        'sales_vs_humidity': '💧 Sales vs Humidity',
        'weather_adjusted': '🌦️ Weather-Adjusted Sales Trend',
        'tonnage_efficiency': '⚡ Sales per Tonnage Efficiency',
        'weekday_heatmap': '📅 Weekday Sales Distribution',
        'forecast_30d': '🔮 30-Day Forecast',
        'anomaly_detection': '⚠️ Anomaly Detection',
        'sales_by_branch': '🏢 Sales by Branch',
        'yoy_growth': '📈 Year-over-Year Growth by Branch',
        'cumulative_trend': '📊 Cumulative Sales Trend',
        'branch_contribution': '📊 Branch Contribution by Segment',
        'sales_mix': '📦 Sales Mix Evolution',
        'top_branch_items': '🏆 Top Branch-Item Pairs',
        'weather_correlation': '🌡️ Weather-Sales Correlation',
        'weather_vs_sales': '🌤️ Regional Weather vs Sales',
        'efficiency_index': '⚡ Branch Efficiency Index',
        'regional_forecast': '🔮 Regional Forecast',
        'forecast_deviation': '📉 Forecast Deviation by Branch'
    }
    
    for chart_key, chart_fig in charts.items():
        chart_json = chart_fig.to_json()
        title = chart_titles.get(chart_key, chart_key.replace('_', ' ').title())
        
        html_content += f"""
    <div class="chart-container">
        <div class="chart-title">{title}</div>
        <div id="chart_{chart_key}"></div>
    </div>
    
    <script>
        var chart_{chart_key} = {chart_json};
        Plotly.newPlot('chart_{chart_key}', chart_{chart_key}.data, chart_{chart_key}.layout);
    </script>
"""
    
    html_content += """
</body>
</html>
"""
    
    # Write to file
    output_path = Path('outputs') / 'dashboards'
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"  ✓ Dashboard saved to: {filepath}")
    return filepath


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to generate both dashboards"""
    print("=" * 80)
    print("DASHBOARD GENERATOR - Branch & Regional Manager Dashboards")
    print("=" * 80)
    
    # Load data
    merged_df, sales_df, weather_df = load_data()
    
    # Get all branches
    branches = sorted(merged_df['Branch'].unique())
    print(f"\nAvailable branches: {branches}")
    
    # Generate Branch Manager Dashboards for each branch
    print("\n" + "=" * 80)
    print("GENERATING BRANCH MANAGER DASHBOARDS")
    print("=" * 80)
    
    for branch in branches:
        try:
            dashboard_data = create_branch_manager_dashboard(merged_df, branch)
            generate_html_dashboard(dashboard_data, 'branch', branch)
        except Exception as e:
            print(f"  ✗ Error creating dashboard for {branch}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Generate Regional Manager Dashboard
    print("\n" + "=" * 80)
    print("GENERATING REGIONAL MANAGER DASHBOARD")
    print("=" * 80)
    
    try:
        dashboard_data = create_regional_manager_dashboard(merged_df)
        generate_html_dashboard(dashboard_data, 'regional')
    except Exception as e:
        print(f"  ✗ Error creating regional dashboard: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("DASHBOARD GENERATION COMPLETE!")
    print("=" * 80)
    print("\nDashboards saved in: outputs/dashboards/")
    print("\nOpen the HTML files in your browser to view the dashboards.")


if __name__ == "__main__":
    main()
