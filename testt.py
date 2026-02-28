import pandas as pd
import joblib


model = joblib.load("income_logistic_regression_model.pkl")
print("Model pipeline loaded successfully\n")


df = pd.read_csv("census_income.csv")

# Binary target
df['income_binary'] = df['income'].apply(lambda x: 1 if x == ">50K" else 0)


sample_df = df.sample(5, random_state=42)

X_sample = sample_df.drop(['income', 'income_binary'], axis=1)
y_true = sample_df['income_binary']


y_pred = model.predict(X_sample)


results = X_sample.copy()

results['Actual Income'] = y_true.map({0: "<=50K", 1: ">50K"})

results['Predicted Income'] = (
    pd.Series(y_pred, index=results.index)
    .map({0: "<=50K", 1: ">50K"})
)

print("\n🔍 Sample-level Income Predictions:\n")
print(results)
