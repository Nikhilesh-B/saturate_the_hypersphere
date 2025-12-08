from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

# This script assumes the following variables are available from your notebook:
# X (scaled training features), Y (training labels), df_new, features, scaler

print("Training Random Forest Model...")

# 1. Initialize and Train Random Forest
# Using 100 trees and fixing random_state for reproducibility
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X, Y.values.ravel())

print("Evaluating on Test Sample (Post 2011)...")

# 2. Prepare Test Data (Ensuring consistency with previous steps)
test_df = df_new.loc['2012-01-02':].copy()
X_test = test_df[features]
Y_test = test_df[['POS_RET']]

# Scale test data using the SAME scaler from training
X_test_scaled = scaler.transform(X_test)
Y_test_arr = Y_test.values.ravel()

# 3. Make Predictions
# Random Forest also outputs probabilities
prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
y_pred_rf = (prob_rf >= 0.5).astype(int)

# 4. Calculate Metrics
cm_rf = confusion_matrix(Y_test_arr, y_pred_rf)
tn, fp, fn, tp = cm_rf.ravel()

type_I_error_rf = fp / (tn + fp) if (tn + fp) > 0 else 0
type_II_error_rf = fn / (fn + tp) if (fn + tp) > 0 else 0
overall_error_rf = (fp + fn) / (tn + fp + fn + tp)

# 5. Print Results
print("\n=== Random Forest Results ===")
print("Confusion Matrix:")
print(pd.DataFrame(cm_rf, index=['True 0', 'True 1'], columns=['Pred 0', 'Pred 1']))
print(f"\nType I Error (False Positive Rate): {type_I_error_rf:.4f}")
print(f"Type II Error (False Negative Rate): {type_II_error_rf:.4f}")
print(f"Overall Error Rate: {overall_error_rf:.4f}")

# Optional: Compare with Logit (if previous results are known/stored)
# You can manually compare these numbers with the output from the previous cell.


