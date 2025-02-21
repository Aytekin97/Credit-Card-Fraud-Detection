import pandas as pd
# Split labels into individual conditions
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np

# Load metadata
metadata = pd.read_csv("PS_20174392719_1491204439457_log.csv")

# Drop useless columns
data_cleaned = metadata.drop(columns=["nameOrig", "nameDest"])

# Check basic information
print(data_cleaned.info())
print(data_cleaned.head(10))
all_types = data_cleaned['type'].str.split('|').explode()
type_counts = Counter(all_types)
print(type_counts)

""" filtered_data = metadata[
    metadata['Finding Labels'].isin(['Effusion', 'Pneumothorax', 'Atelectasis'])
]

# Calculate the mean count of the three target labels
mean_sample_size = int(filtered_data['Finding Labels'].value_counts().mean())
print(f"Mean sample size for target conditions: {mean_sample_size}")

no_finding_data = metadata[metadata['Finding Labels'] == 'No Finding']
no_finding_sample = no_finding_data.sample(n=4300, random_state=42)

final_data = pd.concat([filtered_data, no_finding_sample], ignore_index=True)

print(final_data['Finding Labels'].value_counts())


# Stratified split
train_data, temp_data = train_test_split(
    final_data, test_size=0.30, stratify=final_data['Finding Labels'], random_state=42
)
val_data, test_data = train_test_split(
    temp_data, test_size=0.50, stratify=temp_data['Finding Labels'], random_state=42
)

# Check splits
print(f"\nTrain data shape: {train_data.shape}")
print("\nTraining set class distribution:")
print(train_data['Finding Labels'].value_counts())

print("\nValidation set class distribution:")
print(val_data['Finding Labels'].value_counts())

print("\nTest set class distribution:")
print(test_data['Finding Labels'].value_counts()) """
