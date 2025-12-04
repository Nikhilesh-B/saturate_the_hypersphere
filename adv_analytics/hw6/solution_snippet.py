import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# This script assumes the following variables are available in the notebook environment:
# df_new, scaler, features, clf, clf_cv, l1_model_cv

print("Calculating performance on Test Sample (Post 2011)...")

# 1. Define Test Data
# Selecting data strictly after 2011
test_df = df_new.loc['2012-01-02':].copy()

X_test = test_df[features]
Y_test = test_df[['POS_RET']]

# 2. Scale Test Data
# Critical: Use the SAME scaler fitted on training data
X_test_scaled = scaler.transform(X_test)
Y_test_arr = Y_test.values.ravel()

# 3. Define Models Dictionary
# clf: Standard Logistic Regression (from Cell 46)
# clf_cv: Logistic Regression with L2 penalty (from Cell 47)
# l1_model_cv: Logistic Regression with L1 penalty (from Cell 49)
models = {
    "Standard Logit": clf,
    "L2 Logit (Ridge)": clf_cv,
    "L1 Logit (Lasso)": l1_model_cv
}

# 4. Compute Metrics
for name, model in models.items():
    print(f"\n=== {name} ===")
    
    # Predict probabilities
    # [:, 1] gives probability of class 1 (Positive Return)
    prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Apply cutoff p = 0.5
    y_pred = (prob >= 0.5).astype(int)
    
    # Confusion Matrix
    # Returns [[TN, FP], [FN, TP]]
    cm = confusion_matrix(Y_test_arr, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate Error Rates
    # Type I Error (False Positive Rate) = FP / (TN + FP) (Predict 1 when 0)
    type_I_error = fp / (tn + fp) if (tn + fp) > 0 else 0
    
    # Type II Error (False Negative Rate) = FN / (FN + TP) (Predict 0 when 1)
    type_II_error = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Overall Error Rate = (FP + FN) / Total
    overall_error = (fp + fn) / (tn + fp + fn + tp)
    
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=['True 0', 'True 1'], columns=['Pred 0', 'Pred 1']))
    print(f"\nType I Error (False Positive Rate): {type_I_error:.4f}")
    print(f"Type II Error (False Negative Rate): {type_II_error:.4f}")
    print(f"Overall Error Rate: {overall_error:.4f}")

