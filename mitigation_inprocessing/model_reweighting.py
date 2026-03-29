import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
preprocessor = joblib.load("income_preprocessor.pkl")

X_train, X_test, y_train, y_test = joblib.load("data_split.pkl")

train_df = X_train.copy()
train_df['income'] = y_train

group_counts = train_df.groupby(['sex', 'income']).size()
total = len(train_df)

weights = []

for i, row in train_df.iterrows():
    group = (row['sex'], row['income'])
    p_group = group_counts[group] / total
    weight = 1 / p_group
    weights.append(weight)

weights = np.array(weights)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])


model.fit(X_train, y_train, classifier__sample_weight=weights)

joblib.dump(model, "model_reweighting.pkl")
print("Reweighting model saved!")