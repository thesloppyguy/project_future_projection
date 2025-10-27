# EDA Summary for Feature Engineering and Model Preparation

## Executive Summary

This comprehensive EDA analysis of air conditioner sales data (147,595 records from 2019-2024) reveals critical insights for feature engineering and model preparation. The analysis covers seasonality patterns, lag structures, product trends, geographical performance, weather correlations, and customer trends.

## Key Dataset Characteristics

- **Dataset Size**: 147,595 records
- **Date Range**: April 2019 to March 2024 (5 years)
- **Total Sales Volume**: 1,285,794 units
- **Geographic Coverage**: 5 branches (MAA, SBD, SBD1, BLR, COK)
- **Product Segments**: Inverter (73%) vs Non-Inverter (27%)
- **Tonnage Range**: 0.8T to 2.0T (most popular: 1.5T at 65.1%)

## 1. Seasonality Analysis

### Key Findings:
- **Peak Months**: March, December, February (highest demand)
- **Low Months**: November, August, July (lowest demand)
- **Seasonality Strength**: 0.49 (moderate seasonality)
- **Peak-to-Trough Ratio**: 3.04x difference

**⚠️ Important Note**: The seasonal pattern differs from typical summer AC demand expectations. March and December peaks suggest pre-summer preparation and year-end purchases, while November (Diwali period) shows low demand contrary to expectations.

### Feature Engineering Implications:
1. **Cyclical Features**: Create sine/cosine transformations for monthly seasonality
2. **Seasonal Dummies**: Binary indicators for peak/low seasons
3. **Month Encoding**: Ordinal encoding preserving seasonal order
4. **Quarterly Aggregation**: Q1 (Jan-Mar), Q2 (Apr-Jun), Q3 (Jul-Sep), Q4 (Oct-Dec)

## 2. Lag Structure and Autocorrelation

### Key Findings:
- **Daily Autocorrelation**: Strong at lag-1 (0.6034), moderate at lag-7 (0.2394)
- **Monthly Autocorrelation**: Strong at lag-12 (0.7173) - yearly seasonality
- **Weekly Seasonality**: Strength of 0.976 (very strong)
- **Significant Lags**: First 10 daily lags show significance

**✅ Confirmed**: Strong yearly seasonality (lag-12) and significant lag-1, lag-3 patterns as expected.

### Feature Engineering Implications:
1. **Lag Features**: Include lags 1, 7, 14, 30, 90, 365 for daily models
2. **Rolling Averages**: 7-day, 14-day, 30-day, 90-day moving averages
3. **Rolling Statistics**: Standard deviation, min, max over different windows
4. **Year-over-Year Features**: Same month previous year comparisons

## 3. Product Trends Analysis

### Key Findings:
- **Inverter Growth**: 54.8% growth (2019-2024), market share increased from 53.6% to 85.8%
- **Non-Inverter Decline**: -70.4% growth, market share decreased from 46.4% to 14.2%
- **Star Rating Trends**: 5-star products show 153% growth, 2-star products -99.9% decline
- **Tonnage Preferences**: 1.5T dominates (65.1%), 0.8T shows 115.6% growth

**✅ Confirmed**: Strong inverter adoption trend and higher star ratings gaining significant market share as expected.

### Feature Engineering Implications:
1. **Segment Evolution**: Time-based segment market share features
2. **Product Lifecycle**: Age-based features for product categories
3. **Technology Adoption**: Inverter penetration rate by region/time
4. **Capacity Mix**: Tonnage distribution features

## 4. Branch-wise Performance Analysis

### Key Findings:
- **Market Leaders**: MAA (37.0%), SBD (27.7%), SBD1 (15.8%)
- **Growth Patterns**: COK (+62.8%), SBD (+33.1%), MAA (-25.1%)
- **Seasonality Variation**: SBD most seasonal (0.688), BLR least (0.427)
- **Segment Preferences**: SBD/SBD1 favor inverters (86.1%/74.6%), MAA balanced (52.4%)

**✅ Confirmed**: Significant branch heterogeneity with unique characteristics. COK and SBD show strong growth potential, while MAA and SBD1 show declining trends. Branch-specific seasonality patterns vary significantly, making them good candidates for separate modeling or clustering approaches.

### Feature Engineering Implications:
1. **Geographic Features**: Branch-specific seasonality patterns
2. **Market Share Evolution**: Branch performance relative to total market
3. **Regional Preferences**: Branch-specific product mix features
4. **Competitive Dynamics**: Branch performance ratios

## 5. Weather Correlation Analysis

### Key Findings:
- **Temperature Impact**: 83.4% higher sales in high-temperature periods
- **Humidity Impact**: -68.1% lower sales in high-humidity periods
- **Seasonal Correlations**: Strongest in summer months (June: 0.892, July: 0.856)
- **Branch Variations**: COK shows strongest temperature correlation (0.556)

### Feature Engineering Implications:
1. **Weather Features**: Temperature, humidity, wind speed (min, max, avg)
2. **Weather Lags**: Temperature leading sales by 3 months (optimal lag)
3. **Weather Extremes**: Hot/cold day indicators, humidity thresholds
4. **Seasonal Weather**: Month-specific weather patterns

## 6. Customer Trends Analysis

### Key Findings:
- **Lead-Lag Relationship**: Customer interest leads sales by 3 months
- **Correlation Strength**: 0.477 at 3-month lag (moderate predictive power)
- **Impact Magnitude**: 1.25x difference between high/low interest periods
- **Sensitivity**: 191 units change per interest point

### Feature Engineering Implications:
1. **Trend Features**: Customer interest with 3-month lead
2. **Interest Categories**: High/medium/low interest level indicators
3. **Trend Momentum**: Interest change rates and acceleration
4. **Seasonal Trends**: Month-specific interest patterns

## 7. Statistical Distribution Analysis

### Key Findings:
- **Quantity Distribution**: Highly right-skewed (skewness: 4.525)
- **Outlier Percentage**: 9.26% of records are outliers
- **Coefficient of Variation**: 1.811 (high variability)
- **Non-Normal Distribution**: Confirmed by statistical tests

### Feature Engineering Implications:
1. **Log Transformations**: Apply to quantity for normality
2. **Outlier Handling**: Robust scaling methods, outlier flags
3. **Distribution Features**: Quantile-based features (25th, 50th, 75th percentiles)
4. **Variability Measures**: Coefficient of variation by segment/branch

## Recommended Feature Engineering Strategy

### 1. Temporal Features
```python
# Cyclical encoding for seasonality
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

# Lag features
for lag in [1, 7, 14, 30, 90, 365]:
    df[f'qty_lag_{lag}'] = df['qty'].shift(lag)

# Rolling statistics
for window in [7, 14, 30, 90]:
    df[f'qty_ma_{window}'] = df['qty'].rolling(window).mean()
    df[f'qty_std_{window}'] = df['qty'].rolling(window).std()
```

### 2. Weather Features
```python
# Weather with optimal lag (3 months lead)
df['temp_lead_3m'] = df['avg_temp'].shift(-3)
df['humidity_lead_3m'] = df['avg_humidity'].shift(-3)

# Weather extremes
df['hot_day'] = (df['max_temp'] > df['max_temp'].quantile(0.8)).astype(int)
df['humid_day'] = (df['avg_humidity'] > df['avg_humidity'].quantile(0.8)).astype(int)
```

### 3. Product Features
```python
# Segment evolution
df['inverter_penetration'] = df.groupby(['branch', 'date'])['segment'].apply(
    lambda x: (x == 'Inverter').mean()
)

# Tonnage mix
df['tonnage_1_5_share'] = df.groupby(['branch', 'date'])['tonnage'].apply(
    lambda x: (x == 1.5).mean()
)
```

### 4. Customer Trends Features
```python
# Customer interest with lead
df['interest_lead_3m'] = df['customer_interest'].shift(-3)

# Interest categories
df['interest_level'] = pd.cut(df['customer_interest'], 
                             bins=3, labels=['Low', 'Medium', 'High'])
```

### 5. Geographic Features
```python
# Branch-specific seasonality
df['branch_seasonality'] = df.groupby(['branch', 'month'])['qty'].transform('mean')

# Market share evolution
df['branch_market_share'] = df.groupby('date')['branch'].transform(
    lambda x: x.value_counts(normalize=True)
)
```

## Model Preparation Recommendations

### 1. Data Preprocessing
- **Log Transformation**: Apply to quantity target variable
- **Robust Scaling**: Use median and IQR for scaling
- **Outlier Handling**: Flag outliers but don't remove (they represent real demand spikes)
- **Missing Data**: Forward-fill for weather data, interpolate for trends

### 2. Feature Selection
- **High Priority**: Weather features with 3-month lead, seasonal features, lag features
- **Medium Priority**: Product mix features, branch-specific patterns
- **Low Priority**: Customer trends (weak correlation, not statistically significant)

### 3. Validation Strategy
- **Time Series Split**: Use expanding window validation
- **Seasonal Validation**: Ensure each fold contains complete seasonal cycles
- **Branch-wise Validation**: Stratify by branch for geographic robustness

### 4. Target Variable Engineering
- **Primary Target**: Log-transformed quantity
- **Secondary Targets**: Segment-specific quantities, tonnage-weighted quantities
- **Classification Targets**: Peak/low season indicators, high-demand periods

## Key Insights for Model Development

1. **Strong Seasonality**: Models must capture March-December peaks and summer lows
2. **Weather Dependency**: Temperature and humidity are critical predictors with 3-month lead
3. **Product Evolution**: Inverter technology adoption trend must be modeled
4. **Geographic Variation**: Branch-specific patterns require location-aware features
5. **Lag Structure**: Multiple lag features needed for autoregressive components
6. **Non-Linear Relationships**: Weather-sales relationships may require non-linear modeling

## Conclusion

## Summary: Key Insights Validation

### ✅ **Fully Addressed Insights:**

1. **Lag Structure**: 
   - ✅ Strong lag-12 (yearly seasonality): 0.7173 correlation
   - ✅ Lag-1 significant: 0.6034 (daily), 0.3475 (monthly)
   - ✅ Lag-3 present: 0.2081 (monthly)

2. **Product Trends**:
   - ✅ Inverter AC share growing: 54.8% growth, 53.6% → 85.8% market share
   - ✅ Higher star ratings gaining share: 5-star shows 153% growth

3. **Branch Heterogeneity**:
   - ✅ Branches have unique characteristics: Growth rates vary from -25.1% to +62.8%
   - ✅ Seasonality patterns differ significantly: 0.427 to 0.688 strength
   - ✅ Segment preferences vary: Inverter share ranges from 52.4% to 86.1%
   - ✅ Good clustering candidates: COK/SBD (growth) vs MAA/SBD1 (decline)

### ⚠️ **Partially Addressed Insights:**

1. **Seasonality Pattern**:
   - ❌ Expected: May-July peaks, November (Diwali) peak
   - ✅ Actual: March, December, February peaks; November is low month
   - **Implication**: Seasonal pattern differs from typical summer AC demand expectations

### **Recommendations for Model Development:**

1. **Use Actual Seasonal Patterns**: Model March/December peaks and November lows rather than expected summer patterns
2. **Branch-Specific Models**: Consider separate models for high-growth (COK, SBD) vs declining (MAA, SBD1) branches
3. **Inverter Transition Modeling**: Account for rapid inverter adoption trend in feature engineering
4. **Weather Lag**: Use 3-month temperature lead for optimal correlation

The EDA comprehensively addresses most of your expected insights, with the notable exception of seasonal patterns which differ from typical expectations and require model adaptation.
