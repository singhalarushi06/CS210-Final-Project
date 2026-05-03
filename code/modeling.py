# Part 4: Modeling + Evaluation
# CS210 - NYC Motor Vehicle Collisions

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

print("NYC VEHICLE COLLISIONS - SEVERITY PREDICTION MODEL")

# Load the cleaned dataset
print("\nLoading dataset...")
df = pd.read_csv('data\\collisions_clean.csv', low_memory=False)
print(f"Dataset loaded with {df.shape[0]:,} rows and {df.shape[1]} columns.")

# Check the distribution of the target variable
print(f"\n   Severity distribution:")
print(f"   - Non-severe (0): {(df['is_severe']==0).sum():,} ({(df['is_severe']==0).mean()*100:.1f}%)")
print(f"   - Severe (1): {(df['is_severe']==1).sum():,} ({(df['is_severe']==1).mean()*100:.1f}%)")



# before modeling, we need to prepare some more features
print("\nPreparing features...")

feature_columns = [
    'hour', 'num_vehicles', 'is_rush_hour', 'is_weekend', 'is_night'
]

# Add categorical features as dummies
categorical_cols = ['factor_group', 'borough']
for col in categorical_cols:
    if col in df.columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        feature_columns.extend(dummies.columns.tolist())

# Ensure all features exist
available_features = [col for col in feature_columns if col in df.columns]

# Create feature matrix X and target y
X = df[available_features].copy()
y = df['is_severe'].copy()

print(f"   Features used: {len(available_features)} features")
print(f"   Feature matrix shape: {X.shape}")

# Handle missing values
print(f"\n   Missing values before imputation: {X.isnull().sum().sum()}")
# Impute missing values with median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
print(f"   After preprocessing shape: {X_scaled.shape}")



# Create an 80/20 train test split
print("\nSpliting the data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training set: {X_train.shape[0]:,} samples")
print(f"   Test set: {X_test.shape[0]:,} samples")
# Find the baseline accuracy without any modeling
print(f"   Baseline accuracy (predicting majority class): {max(y.mean(), 1-y.mean()):.3f}")



print("\nTraining models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n   Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_test = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_test)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_test
    }
    
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      AUC-ROC: {auc:.4f}")

# Compare model performance
print("\nModel Performance Comparison:")
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'AUC-ROC': [results[m]['auc'] for m in results]
}).sort_values('Accuracy', ascending=False)

print("\n", comparison_df.to_string(index=False))
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\n BEST MODEL: {best_model_name}")
print(f"   Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}")
print(f"   AUC-ROC: {comparison_df.iloc[0]['AUC-ROC']:.4f}")
