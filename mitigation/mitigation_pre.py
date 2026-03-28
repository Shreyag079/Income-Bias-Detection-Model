import pandas as pd
import joblib
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


X_train, X_test, y_train, y_test = joblib.load("data_split.pkl")

train_df = X_train.copy()
train_df['income'] = y_train

# Oversample
female_high = train_df[(train_df['sex']=='Female') & (train_df['income']==1)]
male_high = train_df[(train_df['sex']=='Male') & (train_df['income']==1)]

female_high_up = resample(
    female_high,
    replace=True,
    n_samples=len(male_high),
    random_state=42
)

train_bal = pd.concat([train_df, female_high_up])

X_train_bal = train_bal.drop('income', axis=1)
y_train_bal = train_bal['income']

# Preprocessing
categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(include=['int64','float64']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Fresh model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])


model.fit(X_train_bal, y_train_bal)


y_pred = model.predict(X_test)

print("\n--- Preprocessing Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


joblib.dump(model, "model_preprocessing.pkl")
print("Preprocessing mitigation model saved!")


female_high_after = train_bal[
    (train_bal['sex'] == 'Female') & (train_bal['income'] == 1)
]

print("\nAfter Preprocessing Mitigation:")
print("High-income females:", len(female_high_after))

