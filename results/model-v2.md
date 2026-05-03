#Version 2 of the model

**NYC VEHICLE COLLISIONS - SEVERITY PREDICTION MODEL**

## Loading dataset...
Dataset loaded with 100,000 rows and 30 columns.

   Severity distribution:
   - Non-severe (0): 76,748 (76.7%)
   - Severe (1): 23,252 (23.3%)

## Preparing features...
   Features used: 15 features
   Feature matrix shape: (100000, 15)

   Missing values before imputation: 0
   After preprocessing shape: (100000, 15)

## Spliting the data...
   Training set: 80,000 samples
   Test set: 20,000 samples
   Baseline accuracy (predicting majority class): 0.767

## Training models...

   Training Logistic Regression...
      Accuracy: 0.7675
      AUC-ROC: 0.6277

   Training Random Forest...
      Accuracy: 0.7677
      AUC-ROC: 0.6304

   Training Gradient Boosting...
      Accuracy: 0.7671
      AUC-ROC: 0.6309

## Model Performance Comparison:

               Model  Accuracy  AUC-ROC
      Random Forest   0.76765 0.630448
Logistic Regression   0.76755 0.627664
  Gradient Boosting   0.76710 0.630894

 BEST MODEL: Random Forest
   Accuracy: 0.7677
   AUC-ROC: 0.6304

------------------------------

Conclusion is that this model still isn't effective. There was slight improvement, but there was still an imbalance in the severity distribution. Let's try adjusting the model directly this time to be able to handle the imbalance.