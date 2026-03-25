import pandas as pd
import joblib


X_train, X_test, y_train, y_test = joblib.load("data_split.pkl")

train_df = X_train.copy()
train_df['income'] = y_train

# Create counterfactuals
cf_df = train_df.copy()

# Flip gender
cf_df['sex'] = cf_df['sex'].apply(lambda x: 'Female' if x=='Male' else 'Male')

# Combine original + counterfactual
augmented = pd.concat([train_df, cf_df])

X_aug = augmented.drop('income', axis=1)
y_aug = augmented['income']


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(include=['int64','float64']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

model_cf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

model_cf.fit(X_aug, y_aug)

joblib.dump(model_cf, "model_counterfactual.pkl")
print("Counterfactual fairness model saved!")