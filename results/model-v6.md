#Version 6 of the model

**NYC VEHICLE COLLISIONS - SEVERITY PREDICTION MODEL**

## Loading dataset...
Dataset loaded with 100,000 rows and 32 columns. <br>

   Severity distribution: <br>
   - Non-severe (0): 58,143 (58.1%) <br>
   - Severe (1): 41,857 (41.9%) <br>

## Spliting Data...
   Training: 80,000 samples <br>
   Test: 20,000 samples <br>
   Baseline (predict majority): 0.581 <br>

## Creating features using ONLY training data...
  Added borough_risk (no leakage) <br>
  Added factor_risk (no leakage) <br>
  Added vehicle_risk_score (no leakage) <br>
  Added interaction features <br>

   Adding targeted features based on SQL query findings... <br>
   Added 3 targeted features (total features: 16) <br>
   Total features after dummies: 29 <br>
   Features used: 29 <br>
   Feature matrix shape: (80000, 29) <br>
   Preprocessing complete <br>

## Training models with class balancing...
C=0.01: Balanced Accuracy=0.6162 <br>
C=0.05: Balanced Accuracy=0.6176 <br>
C=0.1: Balanced Accuracy=0.6174 <br>
C=0.5: Balanced Accuracy=0.6171 <br>
C=1.0: Balanced Accuracy=0.6172 <br>
C=2.0: Balanced Accuracy=0.6172 <br>
C=5.0: Balanced Accuracy=0.6172 <br>

## Model Performance Comparison:

| Model | Accuracy | Balanced Accuracy | AUC-ROC |
|----|----|----|----|
| Logistic Regression | 0.63190 | 0.617598 | 0.664070 |
| Gradient Boosting | 0.63810 | 0.614864 | 0.673883 |
| Random Forest | 0.63755 | 0.606710 | 0.652992 |
| AdaBoost | 0.65380 | 0.601423 | 0.658456 |

 BEST MODEL: Logistic Regression <br>
   Balanced Accuracy: 0.6176 <br>
   AUC-ROC: 0.6641 <br>


------------------------------

Conclusion: Added 3 more features based on the findings from the SQL queries. The features are described below, but the additions to the models helped the balanced accuracy improve even more. <br>
- dangerous combos: those in the evening hours with more vehicles <br>
- accidents where there was a failure to yield (shown to be highest in severity) <br>
- accidents involving vulnerable vehicles (bikes, mopeds, and motorcycles) <br>
