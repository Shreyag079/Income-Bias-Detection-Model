import pandas as pd
import numpy as np
import joblib
import os
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.datasets import BinaryLabelDataset
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

X_train, X_test, y_train, y_test = joblib.load("data_split.pkl")

train_df = X_train.copy()
train_df['income'] = y_train.values if hasattr(y_train, 'values') else y_train

# Encode 'sex' as binary
train_df['sex'] = (train_df['sex'] == 'Male').astype(float)

# Encode all remaining categorical columns
cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols = [c for c in cat_cols if c not in ['income', 'sex']]
train_df = pd.get_dummies(train_df, columns=cat_cols, drop_first=True).astype(float)

# Ensure income is binary numeric
if train_df['income'].dtype == object:
    train_df['income'] = (train_df['income'].str.strip() == '>50K').astype(float)
else:
    train_df['income'] = train_df['income'].astype(float)

# Convert to AIF360 BinaryLabelDataset
dataset = BinaryLabelDataset(
    df=train_df,
    label_names=['income'],
    protected_attribute_names=['sex'],
    favorable_label=1.0,
    unfavorable_label=0.0
)

# TF session + Adversarial Debiasing
sess = tf.compat.v1.Session()

adv_model = AdversarialDebiasing(
    privileged_groups=[{'sex': 1.0}],
    unprivileged_groups=[{'sex': 0.0}],
    scope_name='adv_debias',
    sess=sess,
    num_epochs=50
)

adv_model.fit(dataset)

os.makedirs("model_adversarial_tf", exist_ok=True)
saver = tf.compat.v1.train.Saver()
saver.save(sess, "model_adversarial_tf/model.ckpt")
print("TF weights saved to model_adversarial_tf/")

model_metadata = {
    'scope_name': adv_model.scope_name,
    'privileged_groups': adv_model.privileged_groups,
    'unprivileged_groups': adv_model.unprivileged_groups,
    'num_epochs': adv_model.num_epochs,
    'features_dim': train_df.shape[1] - 1,       
    'feature_names': [c for c in train_df.columns if c != 'income'],
    'tf_checkpoint': "model_adversarial_tf/model.ckpt"
}
joblib.dump(model_metadata, "model_adversarial_meta.pkl")
print("Model metadata saved to model_adversarial_meta.pkl")