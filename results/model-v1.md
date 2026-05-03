#Version 1 of the model

**NYC VEHICLE COLLISIONS - SEVERITY PREDICTION MODEL**

## Loading dataset...
Dataset loaded with 100,000 rows and 28 columns.

   Severity distribution:
   - Non-severe (0): 76,274 (76.3%)
   - Severe (1): 23,726 (23.7%)

## Preparing features...
   Features used: 15 features
   Feature matrix shape: (100000, 15)

   Missing values before imputation: 0
   After preprocessing shape: (100000, 15)

## Spliting the data...
   Training set: 80,000 samples
   Test set: 20,000 samples
   Baseline accuracy (predicting majority class): 0.763

## Training models...

   Training Logistic Regression...
      Accuracy: 0.7616
      AUC-ROC: 0.6224

   Training Random Forest...
      Accuracy: 0.7631
      AUC-ROC: 0.6250

   Training Gradient Boosting...
      Accuracy: 0.7633
      AUC-ROC: 0.6279

## Model Performance Comparison:

               Model  Accuracy  AUC-ROC
  Gradient Boosting   0.76335 0.627932
      Random Forest   0.76310 0.624997
Logistic Regression   0.76160 0.622425

## BEST MODEL: Gradient Boosting
   Accuracy: 0.7633
   AUC-ROC: 0.6279

------------------------------

Conclusion is that this model isn't effective as it didnt change much from the baseline accuracy. There was also a major imbalance in the target (Severe vs Non-Severe). Let's try modifying the target variable to make it more balanced. 
