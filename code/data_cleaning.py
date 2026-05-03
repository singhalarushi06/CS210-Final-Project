# Part 2: Data Cleaning + Feature Engineering
# CS210 - NYC Motor Vehicle Collisions

import pandas as pd
import numpy as np
import re

# load the smaller dataset from the data selection step
# this is already a random sample of the full data, so we can skip straight to cleaning
print("loading dataset, give it a sec...")
df = pd.read_csv("data\\Collisions_sample_100k.csv", low_memory=False)
print(f"size: {df.shape}")

# rename columns IMMEDIATELY - csv comes with caps + spaces like "CRASH TIME"
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print(f"columns found: {list(df.columns)}")

# keep only the columns we actually care about
cols = [
    "crash_date", "crash_time", "borough",
    "latitude", "longitude",
    "number_of_persons_injured", "number_of_persons_killed",
    "number_of_pedestrians_injured", "number_of_pedestrians_killed",
    "number_of_cyclist_injured", "number_of_cyclist_killed",
    "number_of_motorist_injured", "number_of_motorist_killed",
    "contributing_factor_vehicle_1", "contributing_factor_vehicle_2",
    "vehicle_type_code1", "vehicle_type_code2",
    "vehicle_type_code_3", "vehicle_type_code_4", "vehicle_type_code_5",
]
# only keep cols that exist - handles any naming differences in your csv
cols = [c for c in cols if c in df.columns]
df   = df[cols]

# REGEX CLEANING SECTION
# using patterns to catch and fix messy/invalid data

# crash time: only keep valid HH:MM format
time_pattern = re.compile(r"^\d{1,2}:\d{2}$")
def clean_time(t):
    t = str(t).strip()
    return t if time_pattern.match(t) else None

df["crash_time"] = df["crash_time"].apply(clean_time)
print(f"crash_time cleaned")

# borough: remove digits and symbols, only keep valid names - catches typos like "Br00klyn" or "  QUEENS  "
valid_boroughs = {"BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"}
def clean_borough(val):
    val = str(val).strip().upper()
    val = re.sub(r"[^A-Z\s]", "", val)   # kill anything not a letter or space
    val = re.sub(r"\s+", " ", val)        # collapse double spaces
    return val if val in valid_boroughs else "Unknown"

df["borough"] = df["borough"].apply(clean_borough)
print(f"borough cleaned")

# vehicle type: unify all the messy variations
def normalize_vehicle(val):
    val = str(val).strip().upper()
    val = re.sub(r"[^A-Z0-9\s]", " ", val)  # remove punctuation
    val = re.sub(r"\s+", " ", val).strip()
    patterns = [
        (r"\bSEDAN\b|\b4 DOOR\b|\b2 DOOR\b",          "SEDAN"),
        (r"\bTAXI\b|\bCAB\b|\bYELLOW CAB\b",           "TAXI"),
        (r"\bSUV\b|\bSPORT UTILITY\b",                  "SUV"),
        (r"\bBUS\b|\bSCHOOL BUS\b|\bMTA\b",             "BUS"),
        (r"\bTRUCK\b|\bPICKUP\b|\bPICK UP\b",           "TRUCK"),
        (r"\bMOTORCYCLE\b|\bSCOOTER\b|\bMOTORBIKE\b",  "MOTORCYCLE"),
        (r"\bBICYCLE\b|\bBIKE\b|\bEBIKE\b|\bE BIKE\b", "BICYCLE"),
        (r"\bVAN\b|\bMINIVAN\b|\bMINI VAN\b",           "VAN"),
    ]
    for pattern, label in patterns:
        if re.search(pattern, val):
            return label
    return val if val not in ("", "NAN", "UNKNOWN") else "OTHER"

if "vehicle_type_code1" in df.columns:
    df["vehicle_type_code1"] = df["vehicle_type_code1"].apply(normalize_vehicle)
    print(f"vehicle types cleaned")

# contributing factor: strip junk whitespace and non-ascii
def clean_factor(val):
    val = str(val).strip()
    val = re.sub(r"\s+", " ", val)          
    val = re.sub(r"[^\x20-\x7E]", "", val)  
    return val if val.lower() not in ("unspecified", "nan", "") else "Unspecified"

if "contributing_factor_vehicle_1" in df.columns:
    df["contributing_factor_vehicle_1"] = df["contributing_factor_vehicle_1"].apply(clean_factor)
    print(f"contributing factors cleaned")

# coordinates: flag anything outside nyc range - nyc latitude is ~40.x, longitude is ~-73.x or -74.x
lat_pat = re.compile(r"^4[0-9]\.\d+$")
lon_pat = re.compile(r"^-7[0-9]\.\d+$")

def valid_coord(val, pat):
    return bool(pat.match(str(val).strip()))

if "latitude" in df.columns:
    df.loc[~df["latitude"].apply(lambda x: valid_coord(x, lat_pat)),  "latitude"]  = None
    df.loc[~df["longitude"].apply(lambda x: valid_coord(x, lon_pat)), "longitude"] = None
    print(f"bad coordinates wiped")

# FEATURE ENGINEERING

# parse date and extract time features
df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")
# replace None with NaN first so to_datetime doesnt try to parse the string "None"
df["crash_time"] = df["crash_time"].where(df["crash_time"].notna(), other=pd.NA)
df["hour"]       = pd.to_datetime(df["crash_time"].astype(str), errors="coerce").dt.hour
df["month"]      = df["crash_date"].dt.month
df["day_of_week"]= df["crash_date"].dt.day_name()

# rush hour flag: 1 if rush hour, 0 if not - cleaner signal than 5 broad buckets
df["is_rush_hour"] = df["hour"].apply(
    lambda h: 1 if pd.notna(h) and (6 <= h < 10 or 16 <= h < 20) else 0
)

# weekend flag: 1 if saturday or sunday
df["is_weekend"] = df["day_of_week"].apply(
    lambda d: 1 if d in ("Saturday", "Sunday") else 0
)

# night driving flag: midnight to 5am
df["is_night"] = df["hour"].apply(
    lambda h: 1 if pd.notna(h) and (0 <= h < 5) else 0
)

# group contributing factors into 6 clean buckets
def group_factor(val):
    val = str(val).upper()
    if re.search(r"INATTENTION|DISTRACT|PHONE|TEXTING|ELECTRONIC|OUTSIDE|DAYDREAM", val):
        return "Distraction"
    if re.search(r"SPEED|RACING|TOO FAST|UNSAFE SPEED", val):
        return "Speeding"
    if re.search(r"ALCOHOL|DRUG|ILLICIT|CANNABIS|MARIJUANA|IMPAIR", val):
        return "Alcohol/Drugs"
    if re.search(r"YIELD|SIGNAL|STOP SIGN|TRAFFIC CONTROL|RIGHT OF WAY", val):
        return "Failure to Yield"
    if re.search(r"PAVEMENT|GLARE|WEATHER|WIND|FOG|RAIN|SNOW|ICE", val):
        return "Weather/Road"
    if re.search(r"FATIGUE|FELL ASLEEP|DROWSY|ASLEEP", val):
        return "Fatigue"
    return "Other"

# safety check - if the column doesnt exist just label everything Other
if "contributing_factor_vehicle_1" in df.columns:
    df["factor_group"] = df["contributing_factor_vehicle_1"].apply(group_factor)
else:
    df["factor_group"] = "Other"
print(f"factor groups: {df['factor_group'].value_counts().to_dict()}")

# count how many vehicles were involved in the crash
# more vehicles = usually worse crash, strong predictor
vehicle_cols = [c for c in df.columns if "vehicle_type_code" in c]
df["num_vehicles"] = df[vehicle_cols].notna().sum(axis=1)

# fill missing numeric columns with 0
numeric_cols = [c for c in df.columns if "number_of" in c]
df[numeric_cols] = df[numeric_cols].fillna(0)

# to improve the modeling, we are adjusting the metric of the SEVERITY SCORE
# start by calculating the total number of injuries and fatalities
df["total_injuries"] = (
    df["number_of_persons_injured"] +
    df["number_of_pedestrians_injured"] +
    df["number_of_cyclist_injured"] +
    df["number_of_motorist_injured"]
)
df["total_fatalities"] = (
    df["number_of_persons_killed"] +
    df["number_of_pedestrians_killed"] +
    df["number_of_cyclist_killed"] +
    df["number_of_motorist_killed"]
)
# let's increase the score for fatalities a bit more than before
df["severity_score"] = (
    df["number_of_persons_injured"] * 1 +
    df["number_of_persons_killed"]  * 10
)
# let's modify the binary target variable
# we say it is severe if it has 2 or more injuries, or at least 1 fatality - this is a more meaningful definition of severity than the original one which only looked at fatalities
df["is_severe"] = (
    (df["total_injuries"] >= 2) | 
    (df["total_fatalities"] >= 1)
).astype(int)

# drop rows missing critical info
df = df.dropna(subset=["hour", "crash_date"])
print(f"\nfinal shape after all cleaning: {df.shape}")
print(f"severe crashes: {df['is_severe'].sum():,} / {len(df):,} ({df['is_severe'].mean():.1%})")

# save it
df.to_csv("data\\collisions_clean.csv", index=False)
print("\nsaved -> collisions_clean.csv")
print(df[["crash_date", "hour", "is_rush_hour", "is_weekend",
          "factor_group", "num_vehicles", "severity_score"]].head(8))
