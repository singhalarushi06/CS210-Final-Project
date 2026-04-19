import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_recall_fscore_support)
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')
import joblib

from data_cleaning import clean_data, get_time_period

print("NYC VEHICLE COLLISIONS - SEVERITY PREDICTION MODEL")

print("\nLoading dataset...")
df = pd.read_csv('data\\Collisions_sample_100k.csv', low_memory=False)
print(f"Dataset loaded with {df.shape[0]:,} rows and {df.shape[1]} columns.")

# Clean the data
df = clean_data(df)

injury_cols = ['NUMBER_OF_PERSONS_INJURED', 'NUMBER_OF_PEDESTRIANS_INJURED',
               'NUMBER_OF_CYCLIST_INJURED', 'NUMBER_OF_MOTORIST_INJURED']
fatality_cols = ['NUMBER_OF_PERSONS_KILLED', 'NUMBER_OF_PEDESTRIANS_KILLED',
                 'NUMBER_OF_CYCLIST_KILLED', 'NUMBER_OF_MOTORIST_KILLED']
df['TOTAL_INJURED'] = df[injury_cols].sum(axis=1)
df['TOTAL_KILLED'] = df[fatality_cols].sum(axis=1)



