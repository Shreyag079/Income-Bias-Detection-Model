# =====================================
# Logistic Regression for Income Prediction
# =====================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------
# 2. Load dataset
# -------------------------------------

df = pd.read_csv("census_income.csv")

print(df.head())
print(df.info())


# 1 -> High income, 0 -> Low income
df['income'] = df['income'].apply(
    lambda x: 1 if '>50K' in str(x) else 0
)

# -------------------------------------
# 4. Separate features and target
# -------------------------------------
X = df.drop(columns=['income', 'fnlwgt', 'native-country'])
y = df['income']

# -------------------------------------
# 5. Identify column types
# -------------------------------------
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# -------------------------------------
# 6. Preprocessing
# -------------------------------------
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)   

# -------------------------------------
# 7. Train-test split
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------
# 8. Logistic Regression model
# -------------------------------------
logistic_model = LogisticRegression(
    max_iter=1000,
    solver='lbfgs'
)

# -------------------------------------
# 9. Full pipeline
# -------------------------------------
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', logistic_model)
])

# -------------------------------------
# 10. Train model
# -------------------------------------
model.fit(X_train, y_train)

# -------------------------------------
# 11. Predictions
# -------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------------
# 12. Evaluation
# -------------------------------------
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
joblib.dump(model, "income_logistic_regression_model.pkl")
joblib.dump(preprocessor, "income_preprocessor.pkl")

print("Model saved successfully!")
