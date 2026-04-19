import pandas as pd
from datetime import datetime

def get_time_period(hour):
        if pd.isna(hour):
            return 'Unknown'
        elif 5 <= hour < 9:
            return 'Morning_Rush'
        elif 9 <= hour < 12:
            return 'Late_Morning'
        elif 12 <= hour < 16:
            return 'Afternoon'
        elif 16 <= hour < 19:
            return 'Evening_Rush'
        elif 19 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'

def clean_data(df):
    df_clean = df.copy()

    df_clean['CRASH_DATE'] = pd.to_datetime(df_clean['CRASH_DATE'], errors='coerce')
    df_clean['CRASH_TIME'] = pd.to_datetime(df_clean['CRASH_TIME'], format='%H:%M', errors='coerce').dt.time

    df_clean['TIME_PERIOD'] = df_clean['HOUR'].apply(get_time_period)
    df_clean['HOUR'] = pd.to_datetime(df_clean['CRASH_TIME'], format='%H:%M:%S', errors='coerce').dt.hour
    df_clean['HOUR'] = df_clean['HOUR'].fillna(df_clean['HOUR'].median())


    df_clean['DAY_OF_WEEK'] = df_clean['CRASH_DATE'].dt.dayofweek
    df_clean['MONTH'] = df_clean['CRASH_DATE'].dt.month
    df_clean['IS_WEEKEND'] = (df_clean['DAY_OF_WEEK'] >= 5).astype(int)
    df_clean['YEAR'] = df_clean['CRASH_DATE'].dt.year
    

    injury_cols = ['NUMBER_OF_PERSONS_INJURED', 'NUMBER_OF_PEDESTRIANS_INJURED',
                   'NUMBER_OF_CYCLIST_INJURED', 'NUMBER_OF_MOTORIST_INJURED']
    fatality_cols = ['NUMBER_OF_PERSONS_KILLED', 'NUMBER_OF_PEDESTRIANS_KILLED',
                     'NUMBER_OF_CYCLIST_KILLED', 'NUMBER_OF_MOTORIST_KILLED']
    # Fill NaN with 0
    for col in injury_cols + fatality_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)

    df_clean = df_clean.dropna(subset=['LATITUDE', 'LONGITUDE'], how='all')
    
    return df_clean

df_clean = clean_data(df)
print(f"After cleaning: {df_clean.shape}")