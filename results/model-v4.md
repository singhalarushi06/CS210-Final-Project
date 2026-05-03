#Version 3 of the model

**NYC VEHICLE COLLISIONS - SEVERITY PREDICTION MODEL**

## Loading dataset...
Dataset loaded with 100,000 rows and 30 columns.

   Severity distribution:
   - Non-severe (0): 76,748 (76.7%)
   - Severe (1): 23,252 (23.3%)

## Spliting Data...
   Training: 80,000 samples
   Test: 20,000 samples
   Baseline (predict majority): 0.767


## Creating features using ONLY training data...
  Added borough_risk (no leakage)
  Added factor_risk (no leakage)
  Added interaction features
   Total features after dummies: 24
   Features used: 24
   Feature matrix shape: (80000, 24)
   Preprocessing complete

## Training models with class balancing...

## Model Performance Comparison:

Logistic Regression:
   Accuracy: 0.6402
   Balanced Accuracy: 0.5962
   AUC-ROC: 0.6313

Random Forest:
   Accuracy: 0.7215
   Balanced Accuracy: 0.5476
   AUC-ROC: 0.5976

Gradient Boosting:
   Accuracy: 0.6483
   Balanced Accuracy: 0.5958
   AUC-ROC: 0.6352

AdaBoost:
   Accuracy: 0.7674
   Balanced Accuracy: 0.5053
   AUC-ROC: 0.6327

              Model  Accuracy  Balanced Accuracy  AUC-ROC
Logistic Regression   0.64025           0.596240 0.631324
  Gradient Boosting   0.64830           0.595788 0.635173
      Random Forest   0.72145           0.547577 0.597572
           AdaBoost   0.76745           0.505289 0.632668

 BEST MODEL: Logistic Regression
   Balanced Accuracy: 0.5962
   AUC-ROC: 0.6313

------------------------------

Conclusion: even after making changes to the existing parts, changing it up to use a RobustScaler, adding another model (AdaBoost), the results are relatively the same. It's still better than a fully random baseline of 50%, but we can definitely improve our predictive model. At this point, the only other thing we could do is try adding more modeling approaches.