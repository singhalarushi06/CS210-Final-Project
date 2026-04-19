import pandas as pd
import numpy as np

# Load the full dataset
print("Loading full dataset...")
df_full = pd.read_csv('data\\Collisions_sample_500k.csv', low_memory=False)

# Get total rows
total_rows = len(df_full)
print(f"Total rows in dataset: {total_rows:,}")

# Sample 100,000 rows randomly
sample_size = 100000
df_sample = df_full.sample(n=sample_size, random_state=42)

# Save to new CSV
df_sample.to_csv('data\\Collisions_sample_100k.csv', index=False)
print(f"Saved {sample_size:,} random rows to 'data\\Collisions_sample_100k.csv'")

# NOTE: the original dataset has more than 1M rows, so we are deleting the full dataset from memory to free up resources
del df_full