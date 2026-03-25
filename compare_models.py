import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from fairness_metrics import compute_fairness

X_train, X_test, y_train, y_test = joblib.load("data_split.pkl")


baseline = joblib.load("income_logistic_regression_model.pkl")
pre = joblib.load("model_preprocessing.pkl")
inproc = joblib.load("model_inprocessing.pkl")
post_preds = joblib.load("post_preds.pkl")
cf_model = joblib.load("model_counterfactual.pkl")


y_base = baseline.predict(X_test)
y_pre = pre.predict(X_test)
y_in = inproc.predict(X_test)
y_cf = cf_model.predict(X_test)


acc_base = accuracy_score(y_test, y_base)
acc_pre = accuracy_score(y_test, y_pre)
acc_in = accuracy_score(y_test, y_in)
acc_post = accuracy_score(y_test, post_preds)
acc_cf = accuracy_score(y_test, y_cf)



def compute_metrics(X, y_true, y_pred):
    f = compute_fairness(X, y_true, y_pred, 'sex')
    
    groups = list(f.keys())
    g1, g2 = groups[0], groups[1]

    # Extract values
    TPR_diff = abs(f[g1]['TPR'] - f[g2]['TPR'])
    FPR_diff = abs(f[g1]['FPR'] - f[g2]['FPR'])
    PPR_diff = abs(f[g1]['PPR'] - f[g2]['PPR'])

    # Equalized Odds
    EO_diff = (TPR_diff + FPR_diff) / 2

    # Disparate Impact
    di = f[g2]['PPR'] / f[g1]['PPR'] if f[g1]['PPR'] > 0 else 0

    return PPR_diff, EO_diff, di



# Counterfactual Fairness
def counterfactual_bias(model, X, num_samples=500):
    flips = 0

    for i in range(num_samples):
        sample = X.iloc[[i]].copy()
        original = model.predict(sample)[0]

        counter = sample.copy()

        # Flip sex
        if counter["sex"].values[0] == "Male":
            counter["sex"] = "Female"
        else:
            counter["sex"] = "Male"

        new_pred = model.predict(counter)[0]

        if original != new_pred:
            flips += 1

    return flips / num_samples


# Compute Metrics for Each Model
dp_base, eo_base, di_base = compute_metrics(X_test, y_test, y_base)
dp_pre, eo_pre, di_pre = compute_metrics(X_test, y_test, y_pre)
dp_in, eo_in, di_in = compute_metrics(X_test, y_test, y_in)
dp_post, eo_post, di_post = compute_metrics(X_test, y_test, post_preds)
dp_cf, eo_cf, di_cf = compute_metrics(X_test, y_test, y_cf)

# Counterfactual (only for trained models)
cf_base = counterfactual_bias(baseline, X_test)
cf_pre = counterfactual_bias(pre, X_test)
cf_in = counterfactual_bias(inproc, X_test)
cf_bias_cf = counterfactual_bias(cf_model, X_test)



results = [
    ["Baseline", acc_base, dp_base, eo_base, di_base, cf_base],
    ["Preprocessing", acc_pre, dp_pre, eo_pre, di_pre, cf_pre],
    ["Inprocessing", acc_in, dp_in, eo_in, di_in, cf_in],
    ["Postprocessing", acc_post, dp_post, eo_post, di_post, "N/A"],
    ["Counterfactual Fairness", acc_cf, dp_cf, eo_cf, di_cf, cf_bias_cf]
]

df = pd.DataFrame(results, columns=[
    "Model",
    "Accuracy",
    "Demographic Parity Diff",
    "Equalized Odds Diff",
    "Disparate Impact",
    "Counterfactual Bias"
])

print("\nFinal Comparison Table:\n")
print(df)