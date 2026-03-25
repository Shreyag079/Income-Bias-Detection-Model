import joblib
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


model = joblib.load("income_logistic_regression_model.pkl")
X_train, X_test, y_train, y_test = joblib.load("data_split.pkl")


y_probs = model.predict_proba(X_test)[:, 1]


# 3. Apply Threshold Adjustment

threshold = 0.4   
y_pred_post = (y_probs >= threshold).astype(int)


print("\n--- Post-processing Model Evaluation ---")

accuracy = accuracy_score(y_test, y_pred_post)
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_post))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_post))


joblib.dump(y_pred_post, "post_preds.pkl")
print("\nPost-processing predictions saved!")