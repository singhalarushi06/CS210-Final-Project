#Version 6 of the model

**NYC VEHICLE COLLISIONS - SEVERITY PREDICTION MODEL**

## Loading dataset...
Dataset loaded with 100,000 rows and 32 columns.

   Severity distribution:
   - Non-severe (0): 58,143 (58.1%)
   - Severe (1): 41,857 (41.9%)

## Spliting Data...
   Training: 80,000 samples
   Test: 20,000 samples
   Baseline (predict majority): 0.581

## Creating features using ONLY training data...
  Added borough_risk (no leakage)
  Added factor_risk (no leakage)
  Added vehicle_risk_score (no leakage)
  Added interaction features

   Adding targeted features based on SQL query findings...
   Added 3 targeted features (total features: 16)
   Total features after dummies: 29
   Features used: 29
   Feature matrix shape: (80000, 29)
   Preprocessing complete

## Training models with class balancing...
C=0.01: Balanced Accuracy=0.6162
C=0.05: Balanced Accuracy=0.6176
C=0.1: Balanced Accuracy=0.6174
C=0.5: Balanced Accuracy=0.6171
C=1.0: Balanced Accuracy=0.6172
C=2.0: Balanced Accuracy=0.6172
C=5.0: Balanced Accuracy=0.6172

## Model Performance Comparison:

Logistic Regression:
   Accuracy: 0.6319
   Balanced Accuracy: 0.6176
   AUC-ROC: 0.6641

Random Forest:
   Accuracy: 0.6375
   Balanced Accuracy: 0.6067
   AUC-ROC: 0.6530

Gradient Boosting:
   Accuracy: 0.6381
   Balanced Accuracy: 0.6149
   AUC-ROC: 0.6739

AdaBoost:
   Accuracy: 0.6538
   Balanced Accuracy: 0.6014
   AUC-ROC: 0.6585

| Model | Accuracy | Balanced Accuracy | AUC-ROC |
|----|----|----|----|
| Logistic Regression | 0.63190 | 0.617598 | 0.664070 |
| Gradient Boosting | 0.63810 | 0.614864 | 0.673883 |
| Random Forest | 0.63755 | 0.606710 | 0.652992 |
| AdaBoost | 0.65380 | 0.601423 | 0.658456 |

 BEST MODEL: Logistic Regression <br>
   Balanced Accuracy: 0.6176 <br>
   AUC-ROC: 0.6641


------------------------------

Conclusion: Added 3 more features based on the findings from the SQL queries. The features are described below, but the additions to the models helped the balanced accuracy improve even more.
- dangerous combos: those in the evening hours with more vehicles
- accidents where there was a failure to yield (shown to be highest in severity)
- accidents involving vulnerable vehicles (bikes, mopeds, and motorcycles)
