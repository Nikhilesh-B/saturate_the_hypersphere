from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd

# This script assumes the following variables are available from your notebook:
# X (scaled training features), Y (training labels), df_new, features, scaler

print("Training Feed-Forward Neural Network...")

# 1. Initialize and Train Neural Network
# Simple architecture: Two hidden layers with 64 and 32 neurons respectively.
# 'relu' is standard activation. 'adam' is a good default solver.
fnn_model = MLPClassifier(hidden_layer_sizes=(64, 32), 
                          activation='relu', 
                          solver='adam', 
                          max_iter=1000, 
                          random_state=42,
                          early_stopping=True) # Added early stopping to prevent overfitting

fnn_model.fit(X, Y.values.ravel())

print("Evaluating on Test Sample (Post 2011)...")

# 2. Prepare Test Data
test_df = df_new.loc['2012-01-02':].copy()
X_test = test_df[features]
Y_test = test_df[['POS_RET']]

# Scale test data using the SAME scaler from training
X_test_scaled = scaler.transform(X_test)
Y_test_arr = Y_test.values.ravel()

# 3. Make Predictions
prob_fnn = fnn_model.predict_proba(X_test_scaled)[:, 1]
y_pred_fnn = (prob_fnn >= 0.5).astype(int)

# 4. Calculate Metrics
cm_fnn = confusion_matrix(Y_test_arr, y_pred_fnn)
tn, fp, fn, tp = cm_fnn.ravel()

type_I_error_fnn = fp / (tn + fp) if (tn + fp) > 0 else 0
type_II_error_fnn = fn / (fn + tp) if (fn + tp) > 0 else 0
overall_error_fnn = (fp + fn) / (tn + fp + fn + tp)

# 5. Print Results
print("\n=== Feed-Forward Neural Network Results ===")
print("Confusion Matrix:")
print(pd.DataFrame(cm_fnn, index=['True 0', 'True 1'], columns=['Pred 0', 'Pred 1']))
print(f"\nType I Error (False Positive Rate): {type_I_error_fnn:.4f}")
print(f"Type II Error (False Negative Rate): {type_II_error_fnn:.4f}")
print(f"Overall Error Rate: {overall_error_fnn:.4f}")


