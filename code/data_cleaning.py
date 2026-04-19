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

    df_clean['CRASH DATE'] = pd.to_datetime(df_clean['CRASH DATE'], errors='coerce')
    df_clean['CRASH TIME'] = pd.to_datetime(df_clean['CRASH TIME'], format='%H:%M', errors='coerce').dt.time
    
    df_clean['HOUR'] = pd.to_datetime(df_clean['CRASH TIME'], format='%H:%M:%S', errors='coerce').dt.hour
    df_clean['HOUR'] = df_clean['HOUR'].fillna(df_clean['HOUR'].median())
    df_clean['TIME_PERIOD'] = df_clean['HOUR'].apply(get_time_period)


    df_clean['DAY_OF_WEEK'] = df_clean['CRASH DATE'].dt.dayofweek
    df_clean['MONTH'] = df_clean['CRASH DATE'].dt.month
    df_clean['IS_WEEKEND'] = (df_clean['DAY_OF_WEEK'] >= 5).astype(int)
    df_clean['YEAR'] = df_clean['CRASH DATE'].dt.year
    

    injury_cols = ['NUMBER OF PERSONS INJURED', 'NUMBER OF PEDESTRIANS INJURED',
                   'NUMBER OF CYCLIST INJURED', 'NUMBER OF MOTORIST INJURED']
    fatality_cols = ['NUMBER OF PERSONS KILLED', 'NUMBER OF PEDESTRIANS KILLED',
                     'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST KILLED']
    # Fill NaN with 0
    for col in injury_cols + fatality_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    
    df_clean['TOTAL_INJURIES'] = df_clean[injury_cols].sum(axis=1)
    df_clean['TOTAL_FATALITIES'] = df_clean[fatality_cols].sum(axis=1)
    df_clean['SEVERITY_SCORE'] = df_clean['TOTAL_INJURIES'] + (3 * df_clean['TOTAL_FATALITIES'])
    df_clean['IS_SEVERE'] = (df_clean['SEVERITY_SCORE'] > df_clean['SEVERITY_SCORE'].median()).astype(int)

    df_clean = df_clean.dropna(subset=['LATITUDE', 'LONGITUDE'], how='all')
    
    return df_clean

df = pd.read_csv('data\\Collisions_sample_100k.csv', low_memory=False)
print(f"Before cleaning: {df.shape}")
df_clean = clean_data(df)
print(f"After cleaning: {df_clean.shape}")

output_path = 'data\\cleaned_100k.csv'
df_clean.to_csv(output_path, index=False)
print(f"Saved cleaned data to: {output_path}")
    
# Show what new columns were added
new_cols = ['HOUR', 'TIME_PERIOD', 'DAY_OF_WEEK', 'MONTH', 'IS_WEEKEND', 
            'YEAR', 'TOTAL_INJURIES', 'TOTAL_FATALITIES', 'SEVERITY_SCORE', 'IS_SEVERE', 'NUM_VEHICLES']
print("\nNew columns added for modeling:")
for col in new_cols:
    if col in df_clean.columns:
        print(f" - {col}")