# Part 4: Modeling + Evaluation
# CS210 - NYC Motor Vehicle Collisions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
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

# spliting data
print("\nSpliting Data...")
X_temp = df.drop('is_severe', axis=1)
y = df['is_severe']
X_train, X_test, y_train, y_test = train_test_split(
    X_temp, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training: {len(X_train):,} samples")
print(f"   Test: {len(X_test):,} samples")
print(f"   Baseline (predict majority): {max(y.mean(), 1-y.mean()):.3f}")

# before modeling, we need to prepare some more features
print("\nPreparing features...")
print("\nCreating features using ONLY training data...")

# Create copies to ensure we are only using the training set
train_df = X_train.copy()
test_df = X_test.copy()
train_df['is_severe'] = y_train.values

# Cyclical time features
for df_temp in [train_df, test_df]:
    df_temp['hour_sin'] = np.sin(2 * np.pi * df_temp['hour'] / 24)
    df_temp['hour_cos'] = np.cos(2 * np.pi * df_temp['hour'] / 24)

# Create severity risk scores from historical data
# This uses target encoding - powerful but must be done carefully
# Target encoding

# Borough risk score
borough_risk_train = train_df.groupby('borough')['is_severe'].mean()
train_df['borough_risk'] = train_df['borough'].map(borough_risk_train).fillna(borough_risk_train.mean())
test_df['borough_risk'] = test_df['borough'].map(borough_risk_train).fillna(borough_risk_train.mean())
print("  Added borough_risk (no leakage)")

# Factor group risk score
factor_risk_train = train_df.groupby('factor_group')['is_severe'].mean()
train_df['factor_risk'] = train_df['factor_group'].map(factor_risk_train).fillna(factor_risk_train.mean())
test_df['factor_risk'] = test_df['factor_group'].map(factor_risk_train).fillna(factor_risk_train.mean())
print("  Added factor_risk (no leakage)")

# Vehicle risk score
if 'vehicle_type_code1' in train_df.columns:
    vehicle_risk_train = train_df.groupby('vehicle_type_code1')['is_severe'].mean()
    train_df['vehicle_risk_score'] = train_df['vehicle_type_code1'].map(vehicle_risk_train).fillna(vehicle_risk_train.mean())
    test_df['vehicle_risk_score'] = test_df['vehicle_type_code1'].map(vehicle_risk_train).fillna(vehicle_risk_train.mean())
    print("  Added vehicle_risk_score (no leakage)")

# Interaction features
for df_temp in [train_df, test_df]:
    df_temp['dangerous_combo'] = ((df_temp['is_night'] == 1) & (df_temp['is_weekend'] == 1)).astype(int)
    df_temp['rush_hour_weekday'] = ((df_temp['is_rush_hour'] == 1) & (df_temp['is_weekend'] == 0)).astype(int)
    df_temp['high_vehicle_count'] = (df_temp['num_vehicles'] >= 3).astype(int)
    df_temp['is_winter'] = df_temp['month'].isin([12, 1, 2]).astype(int)
print("  Added interaction features")

feature_columns = [
    'hour_sin', 'hour_cos', 
    'num_vehicles', 
    'is_rush_hour', 
    'is_weekend', 
    'is_night', 
    'dangerous_combo', 
    'rush_hour_weekday', 
    'high_vehicle_count', 
    'is_winter',
    'borough_risk',
    'factor_risk'
]
if 'vehicle_risk_score' in train_df.columns:
    feature_columns.append('vehicle_risk_score')

# Add categorical features as dummies
categorical_cols = ['factor_group', 'borough']
for col in categorical_cols:
    if col in train_df.columns: 
        dummies_train = pd.get_dummies(train_df[col], prefix=col) 
        train_df = pd.concat([train_df, dummies_train], axis=1)
        # For test, create matching columns
        for dummy_col in dummies_train.columns:
            if dummy_col not in test_df.columns:
                test_df[dummy_col] = 0
        feature_columns.extend(dummies_train.columns.tolist())
print(f"   Total features after dummies: {len(feature_columns)}")

# Ensure all features exist
available_features = [col for col in feature_columns if col in train_df.columns]

# Create feature matrix X and target y
X_train_feat = train_df[available_features].copy() 
X_test_feat = test_df[available_features].copy() 
y_train_clean = train_df['is_severe'].copy()
y_test_clean = y_test.copy()

print(f"   Features used: {len(available_features)}")
print(f"   Feature matrix shape: {X_train_feat.shape}")

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train_feat)
X_test_imputed = imputer.transform(X_test_feat)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print(f"   Preprocessing complete")

print("\nTraining models with class balancing...")

# Model 1: Logistic Regression (has class_weight parameter)
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train_clean)
y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

# Model 2: Random Forest (has class_weight parameter)
rf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, 
                           n_jobs=-1, min_samples_split=3, min_samples_leaf=1, 
                           class_weight='balanced_subsample')
rf.fit(X_train_scaled, y_train_clean)
y_pred_rf = rf.predict(X_test_scaled)
y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

# Model 3: Gradient Boosting (NO class_weight parameter - must use sample_weight)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.03, 
                                subsample=0.8, min_samples_split=5, random_state=42)
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_clean)
gb.fit(X_train_scaled, y_train_clean, sample_weight=sample_weights)  # KEY FIX: using sample_weights
y_pred_gb = gb.predict(X_test_scaled)
y_proba_gb = gb.predict_proba(X_test_scaled)[:, 1]

# Also trying AdaBoost as an extra model
ada = AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=42)
ada.fit(X_train_scaled, y_train_clean)
y_pred_ada = ada.predict(X_test_scaled)
y_proba_ada = ada.predict_proba(X_test_scaled)[:, 1]


# Store results
results = {
    'Logistic Regression': {
        'model': lr,
        'accuracy': accuracy_score(y_test_clean, y_pred_lr),
        'balanced_accuracy': balanced_accuracy_score(y_test_clean, y_pred_lr),
        'auc': roc_auc_score(y_test_clean, y_proba_lr),
        'predictions': y_pred_lr,
        'probabilities': y_proba_lr
    },
    'Random Forest': {
        'model': rf,
        'accuracy': accuracy_score(y_test_clean, y_pred_rf),
        'balanced_accuracy': balanced_accuracy_score(y_test_clean, y_pred_rf),
        'auc': roc_auc_score(y_test_clean, y_proba_rf),
        'predictions': y_pred_rf,
        'probabilities': y_proba_rf
    },
    'Gradient Boosting': {
        'model': gb,
        'accuracy': accuracy_score(y_test_clean, y_pred_gb),
        'balanced_accuracy': balanced_accuracy_score(y_test_clean, y_pred_gb),
        'auc': roc_auc_score(y_test_clean, y_proba_gb),
        'predictions': y_pred_gb,
        'probabilities': y_proba_gb
    },
    'AdaBoost': {
        'model': ada,
        'accuracy': accuracy_score(y_test_clean, y_pred_ada),
        'balanced_accuracy': balanced_accuracy_score(y_test_clean, y_pred_ada),
        'auc': roc_auc_score(y_test_clean, y_proba_ada),
        'predictions': y_pred_ada,
        'probabilities': y_proba_ada
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