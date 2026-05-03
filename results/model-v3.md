#Version 3 of the model

**NYC VEHICLE COLLISIONS - SEVERITY PREDICTION MODEL**

## Loading dataset...
Dataset loaded with 100,000 rows and 30 columns.

   Severity distribution:
   - Non-severe (0): 76,748 (76.7%)
   - Severe (1): 23,252 (23.3%)

## Preparing features...
   Features used: 19
   Feature matrix shape: (100000, 19)
   Preprocessing complete

## Spliting the data...
   Training set: 80,000 samples
   Test set: 20,000 samples
   Baseline accuracy (predicting majority class): 0.767

## Training models with class balancing...

## Model Performance Comparison:

Logistic Regression:
   Accuracy: 0.6623
   Balanced Accuracy: 0.5936
   AUC-ROC: 0.6296

Random Forest:
   Accuracy: 0.6582
   Balanced Accuracy: 0.5928
   AUC-ROC: 0.6321

Gradient Boosting:
   Accuracy: 0.6485
   Balanced Accuracy: 0.5963
   AUC-ROC: 0.6338

              Model  Accuracy  Balanced Accuracy  AUC-ROC
  Gradient Boosting   0.64855           0.596251 0.633752
Logistic Regression   0.66225           0.593633 0.629631
      Random Forest   0.65815           0.592836 0.632068

 BEST MODEL: Gradient Boosting
   Balanced Accuracy: 0.5963
   AUC-ROC: 0.6338


------------------------------

Conclusion is that a after adjusting the models to adjust to the imbalance, we are just stuck around the 60% mark for the balanced accuracy. We can try to make some of the features stronger.