import joblib
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

preprocessor = joblib.load("income_preprocessor.pkl")
X_train, X_test, y_train, y_test = joblib.load("data_split.pkl")

base_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Fairness constraint
constraint = EqualizedOdds()

mitigator = ExponentiatedGradient(
    estimator=base_model,
    constraints=constraint,
    sample_weight_name='classifier__sample_weight'   
)


mitigator.fit(
    X_train,
    y_train,
    sensitive_features=X_train['sex']
)

joblib.dump(mitigator, "model_fairness_constraint.pkl")
print("Fairness constraint model saved!")