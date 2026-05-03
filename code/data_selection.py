# Part 1: Data Selection + Sampling
# CS210 - NYC Motor Vehicle Collisions
# pulls data directly from the NYC Open Data API instead of a manual download
# API docs: https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95

import pandas as pd
import requests
import io
import os

# BASE points to the CS210-Final-Project folder regardless of where you run the script from
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the base API endpoint for this dataset
API_URL = "https://data.cityofnewyork.us/resource/h9gi-nx95.csv"

# we need 100k rows total but the API caps at 50k per request without a token
# so we make 2 requests of 50k each with different offsets and combine them
BATCH_SIZE = 50_000

def fetch_batch(limit, offset):
    # build the request with limit and offset params - this is how the Socrata API works
    params = {
        "$limit":  limit,
        "$offset": offset,
        "$order":  "crash_date DESC",  # get the most recent crashes first
    }
    print(f"  fetching rows {offset:,} to {offset + limit:,}...")
    response = requests.get(API_URL, params=params, timeout=60)

    # if something goes wrong with the request, tell us what happened
    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code} - {response.text[:200]}")

    # parse the CSV response directly into a dataframe
    batch_df = pd.read_csv(io.StringIO(response.text), low_memory=False)
    print(f"  got {len(batch_df):,} rows")
    return batch_df

print("Fetching NYC collision data from API...")
print(f"Making 2 requests of {BATCH_SIZE:,} rows each\n")

# batch 1: rows 0 to 50k
batch1 = fetch_batch(limit=BATCH_SIZE, offset=0)

# batch 2: rows 50k to 100k
batch2 = fetch_batch(limit=BATCH_SIZE, offset=BATCH_SIZE)

# combine both batches into one dataframe
df = pd.concat([batch1, batch2], ignore_index=True)
print(f"\ncombined total: {len(df):,} rows")

# shuffle the rows so we get a good random mix from both batches
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"shuffled rows for randomness")

# make sure the data folder exists before saving
os.makedirs(BASE + "\\data", exist_ok=True)

# save to the same filename that data_cleaning.py expects - nothing downstream changes
df.to_csv(BASE + "\\data\\Collisions_sample_100k.csv", index=False)
print(f"\nsaved {len(df):,} rows -> data\\Collisions_sample_100k.csv")
print(f"columns: {list(df.columns)}")
print("\ndone! you can now run data_cleaning.py")