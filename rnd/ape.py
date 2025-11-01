
import pandas as pd
import numpy as np

data = pd.read_csv('./data/merged_filter_ingestion.csv')

print("Missing values before filling:")
print(data.isna().sum())

# Fill Star Rating: 50% 3 Star, 40% 5 Star, 10% random of 1 Star, 2 Star, and 4 Star
star_rating_mask = data['Star Rating'].isna()
num_missing_star = star_rating_mask.sum()

# Calculate counts for each category
num_3_star = int(num_missing_star * 0.5)
num_5_star = int(num_missing_star * 0.4)
num_other_star = num_missing_star - num_3_star - num_5_star

# Create array of values to fill
star_values = (
    ['3 Star'] * num_3_star +
    ['5 Star'] * num_5_star +
    np.random.choice(['1 Star', '2 Star', '4 Star'],
                     size=num_other_star).tolist()
)

# Shuffle to randomize order
np.random.shuffle(star_values)
data.loc[star_rating_mask, 'Star Rating'] = star_values

# Fill Segment: 90% Inv, 10% Non Inv
segment_mask = data['Segment'].isna()
num_missing_segment = segment_mask.sum()

num_inv = int(num_missing_segment * 0.9)
num_non_inv = num_missing_segment - num_inv

segment_values = ['Inv'] * num_inv + ['Non Inv'] * num_non_inv
np.random.shuffle(segment_values)
data.loc[segment_mask, 'Segment'] = segment_values

# Fill Tonnage: 50% 1.5, 40% 1.0, 10% rest (0.8, 1.8, 2.2)
tonnage_mask = data['Tonnage'].isna()
num_missing_tonnage = tonnage_mask.sum()

num_1_5 = int(num_missing_tonnage * 0.5)
num_1_0 = int(num_missing_tonnage * 0.4)
num_other_tonnage = num_missing_tonnage - num_1_5 - num_1_0

tonnage_values = (
    [1.5] * num_1_5 +
    [1.0] * num_1_0 +
    np.random.choice([0.8, 1.8, 2.2], size=num_other_tonnage).tolist()
)

np.random.shuffle(tonnage_values)
data.loc[tonnage_mask, 'Tonnage'] = tonnage_values

print("\nMissing values after filling:")
print(data.isna().sum())

print("\nValue distributions after filling:")
print("\nStar Rating distribution:")
print(data['Star Rating'].value_counts(normalize=True) * 100)
print("\nSegment distribution:")
print(data['Segment'].value_counts(normalize=True) * 100)
print("\nTonnage distribution:")
print(data['Tonnage'].value_counts(normalize=True) * 100)


data['Quantity'] = data['Quantity'].apply(lambda x: -x if x < 0 else x)

# Save the filled data back to CSV
data.to_csv('./data/merged_filter_ingestion.csv', index=False)
print("\nData saved to './data/merged_filter_ingestion.csv'")
