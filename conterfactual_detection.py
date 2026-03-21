import pandas as pd
import joblib


model = joblib.load("income_logistic_regression_model.pkl")


df = pd.read_csv("census_income.csv")


X = df.drop("income", axis=1)


biased_cases = 0
total_checked = 1000

for i in range(total_checked):

    sample = X.iloc[[i]].copy()

    original_pred = model.predict(sample)[0]

    counter = sample.copy()

    # flip sex
    if counter["sex"].values[0] == "Male":
        counter["sex"] = "Female"
    else:
        counter["sex"] = "Male"

    counter_pred = model.predict(counter)[0]

    if original_pred != counter_pred:
        biased_cases += 1

print("Total samples checked:", total_checked)
print("Prediction flips:", biased_cases)
print("Counterfactual bias rate:", biased_cases/total_checked)