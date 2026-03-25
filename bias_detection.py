

import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

# Load trained model
model = joblib.load("income_logistic_regression_model.pkl")

# Load dataset
df = pd.read_csv("census_income.csv")

df['income_binary'] = df['income'].apply(lambda x: 1 if x == ">50K" else 0)

X = df.drop(columns=['income', 'income_binary', 'fnlwgt', 'native-country'])
y = df['income_binary']

# Generate predictions
df['prediction'] = model.predict(X)

# Group Metrics Function
def group_metrics(data, group_col):
    groups = data[group_col].unique()
    
    for group in groups:
        subset = data[data[group_col] == group]
        
        tn, fp, fn, tp = confusion_matrix(
            subset['income_binary'],
            subset['prediction']
        ).ravel()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        positive_rate = subset['prediction'].mean()
        
        print(f"\nGroup: {group}")
        print("Samples:", len(subset))
        print("Recall (TPR):", round(recall, 3))
        print("False Negative Rate:", round(fnr, 3))
        print("Positive Prediction Rate:", round(positive_rate, 3))

# Bias Analysis
print("\nBias Detection by SEX")
group_metrics(df, 'sex')
print("\nBias Detection by RACE")
group_metrics(df, 'race')

