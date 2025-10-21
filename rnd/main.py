import pandas as pd
from io import StringIO

# Your data
data = """
"Month","Average","2025","2024","2023","2022","2021"
"January","33.2","33.2","33.7","33.1","32.5","32.7"
"February","34.4","34.2","35.3","34.6","33.2","34.1"
"March","34.8","34.5","35.7","35.5","33.1","34.7"
"April","34.2","33.6","36.2","35.1","32.3","33.3"
"May","32.4","31.6","33.2","34.0","30.1","31.8"
"June","30.9","30.1","30.8","31.5","30.8","31.1"
"July","29.7","29.0","29.3","30.1","29.3","29.8"
"August","30.2","29.3","30.3","32.2","29.3","29.6"
"September","30.4","30.6","31.1","30.3","30.6","29.8"
"October","31.2","31.6","31.1","32.2","31.3","30.1"
"November","32.2","-","32.1","33.2","32.2","30.1"
"December","32.5","-","31.6","33.6","32.7","32.2"
"**Average**","**32.1**","**31.7**","**32.6**","**32.9**","**31.4**","**31.6**"
"""

# Read CSV
df = pd.read_csv(StringIO(data))

# Drop the 'Average' row
df = df[df['Month'] != 'Average']

# Replace '-' with NaN
df.replace('-', pd.NA, inplace=True)

# Melt to long format
df_long = df.melt(id_vars=['Month'], var_name='Year', value_name='Value')

# Convert Value to numeric (NaN will stay)
df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')

# Map month names to numbers
month_map = { 
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
df_long['MonthNum'] = df_long['Month'].map(month_map)

# Create proper datetime
df_long['Date'] = pd.to_datetime(
    df_long['Year'].astype(str) + '-' + df_long['MonthNum'].astype(str).str.zfill(2) + '-01',
    format='%Y-%m-%d',
    errors='coerce'
)

# Drop rows where Date is NaT (if any)
df_long = df_long.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

print(df_long[['Date', 'Value']])
