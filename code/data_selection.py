# Part 1: Data Selection + Sampling
# CS210 - NYC Motor Vehicle Collisions

import pandas as pd
import requests
import io
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the NYC Open Data API endpoint for motor vehicle collisions
url = "https://data.cityofnewyork.us/resource/h9gi-nx95.csv"

# BATCH 1: fetch first 50k rows
params1 = {
    "$limit": 50000,
    "$offset": 0,
    "$order": "crash_date DESC"
}
response1 = requests.get(url, params=params1, timeout=60)
print("Batch 1 status code:", response1.status_code)

# parse the response into a dataframe
batch1 = pd.read_csv(io.StringIO(response1.text), low_memory=False)
print("Batch 1 rows:", len(batch1))

# BATCH 2: fetch next 50k rows using offset
params2 = {
    "$limit": 50000,
    "$offset": 50000,
    "$order": "crash_date DESC"
}
response2 = requests.get(url, params=params2, timeout=60)
print("Batch 2 status code:", response2.status_code)

# parse the second batch
batch2 = pd.read_csv(io.StringIO(response2.text), low_memory=False)
print("Batch 2 rows:", len(batch2))

# combine both batches into one dataframe
df = pd.concat([batch1, batch2], ignore_index=True)
print("Combined total:", len(df))

# quick look at the data
print(df.head())

# check columns
print(df.columns.tolist())

# shuffle so we get a random mix from both batches
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# document where this data came from (data provenance)
provenance = {
    "source": url,
    "collected_on": pd.Timestamp.now(),
    "collection_method": "2x GET requests using requests library (50k rows each)",
    "total_rows": len(df),
    "transformations": "Combined 2 batches, shuffled rows"
}
print(provenance)

# save to csv for the next step
os.makedirs(BASE + "\\data", exist_ok=True)
df.to_csv(BASE + "\\data\\Collisions_sample_100k.csv", index=False)
print(f"\nsaved {len(df):,} rows -> data\\Collisions_sample_100k.csv")
print("done! you can now run data_cleaning.py")
