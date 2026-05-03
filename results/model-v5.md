#Version 5 of the model

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
Training models with class balancing...
C=0.01: Balanced Accuracy=0.5930
C=0.05: Balanced Accuracy=0.5941
C=0.1: Balanced Accuracy=0.5949
C=0.5: Balanced Accuracy=0.5961
C=1.0: Balanced Accuracy=0.5962
C=2.0: Balanced Accuracy=0.5963
C=5.0: Balanced Accuracy=0.5962

Model Performance Comparison:

Logistic Regression:
   Accuracy: 0.6404
   Balanced Accuracy: 0.5961
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



| Model | Accuracy | Balanced Accuracy | AUC-ROC |
| ---- | ---- | ---- | ---- | 
| Logistic Regression | 0.64045 | 0.596145 | 0.631324 |
| Gradient Boosting | 0.64830 | 0.595788 | 0.635173 |
| Random Forest | 0.72145 | 0.547577 | 0.597572 |
| AdaBoost | 0.76745 | 0.505289 | 0.632668 |

 BEST MODEL: Logistic Regression
   Balanced Accuracy: 0.5961
   AUC-ROC: 0.6313

------------------------------

Conclusion: Since the logistic regression model was more common to be the best performing model, we made changes to it to see if we can improve it further. We tested different regularization strengths to figure out if and tweaks can make the performance better. There was minimal change, so we are currently stuck at a plateau. Will brainstrom what further improvements can be made.