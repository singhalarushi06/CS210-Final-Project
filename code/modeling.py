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
                             precision_recall_fscore_support, balanced_accuracy_score)
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight

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

# Interaction features (more predictive than individual features)
df['dangerous_combo'] = ((df['is_night'] == 1) & (df['is_weekend'] == 1)).astype(int)
df['rush_hour_weekday'] = ((df['is_rush_hour'] == 1) & (df['is_weekend'] == 0)).astype(int)
# Vehicle risk (more vehicles = higher risk, but non-linear)
df['high_vehicle_count'] = (df['num_vehicles'] >= 3).astype(int)
# Month seasonality (winter months might be worse)
df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)

feature_columns = [
    'hour', 
    'num_vehicles', 
    'is_rush_hour', 
    'is_weekend', 
    'is_night', 
    'dangerous_combo', 
    'rush_hour_weekday', 
    'high_vehicle_count', 
    'is_winter'
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

print(f"   Features used: {len(available_features)}")
print(f"   Feature matrix shape: {X.shape}")

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
print(f"   Preprocessing complete")


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
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', C=0.1),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, min_samples_split=5, min_samples_leaf=2, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)
}

from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

print("\nTraining models with class balancing...")

# Model 1: Logistic Regression (has class_weight parameter)
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

# Model 2: Random Forest (has class_weight parameter)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, 
                           n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# Model 3: Gradient Boosting (NO class_weight parameter - must use sample_weight)
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb.fit(X_train, y_train, sample_weight=sample_weights)  # KEY FIX: using sample_weights
y_pred_gb = gb.predict(X_test)
y_proba_gb = gb.predict_proba(X_test)[:, 1]

# Store results
results = {
    'Logistic Regression': {
        'model': lr,
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_lr),
        'auc': roc_auc_score(y_test, y_proba_lr),
        'predictions': y_pred_lr,
        'probabilities': y_proba_lr
    },
    'Random Forest': {
        'model': rf,
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_rf),
        'auc': roc_auc_score(y_test, y_proba_rf),
        'predictions': y_pred_rf,
        'probabilities': y_proba_rf
    },
    'Gradient Boosting': {
        'model': gb,
        'accuracy': accuracy_score(y_test, y_pred_gb),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_gb),
        'auc': roc_auc_score(y_test, y_proba_gb),
        'predictions': y_pred_gb,
        'probabilities': y_proba_gb
    }
}


# Compare model performance
print("\nModel Performance Comparison:")
comparison_data = []
for name, res in results.items():
    comparison_data.append({
        'Model': name,
        'Accuracy': res['accuracy'],
        'Balanced Accuracy': res['balanced_accuracy'],
        'AUC-ROC': res['auc']
    })
    print(f"\n{name}:")
    print(f"   Accuracy: {res['accuracy']:.4f}")
    print(f"   Balanced Accuracy: {res['balanced_accuracy']:.4f}")  # Better metric for imbalanced
    print(f"   AUC-ROC: {res['auc']:.4f}")

comparison_df = pd.DataFrame(comparison_data).sort_values('Balanced Accuracy', ascending=False)
print("\n" + comparison_df.to_string(index=False))

# Find best model (using balanced accuracy for imbalanced data)
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\n BEST MODEL: {best_model_name}")
print(f"   Balanced Accuracy: {comparison_df.iloc[0]['Balanced Accuracy']:.4f}")
print(f"   AUC-ROC: {comparison_df.iloc[0]['AUC-ROC']:.4f}")