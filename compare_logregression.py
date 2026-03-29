import pandas as pd
import numpy as np
import joblib
import os
import warnings
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.inprocessing import AdversarialDebiasing
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
warnings.filterwarnings("ignore")

OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
preprocessor = joblib.load("income_preprocessor.pkl")
X_train, X_test, y_train, y_test = joblib.load("data_split.pkl")

y_test_bin = (pd.Series(y_test).str.strip() == '>50K').astype(int).values \
             if pd.api.types.is_object_dtype(y_test) \
             else np.array(y_test, dtype=int)

sex_test  = (X_test['sex'] == 'Male').astype(float).values
sex_train = (X_train['sex'] == 'Male').astype(float).values

# ── Pre-transform for Fairness Constraint ─────────────────────────────────────
X_train_t = preprocessor.transform(X_train)
X_test_t  = preprocessor.transform(X_test)
if sp.issparse(X_train_t): X_train_t = X_train_t.toarray()
if sp.issparse(X_test_t):  X_test_t  = X_test_t.toarray()

# ── Build numeric test df for Adversarial model ───────────────────────────────
def build_numeric_df(X, y):
    df = X.copy()
    df['sex']    = (df['sex'] == 'Male').astype(float)
    df['income'] = (pd.Series(y).str.strip() == '>50K').astype(float) \
                   if pd.api.types.is_object_dtype(y) \
                   else pd.Series(y).astype(float)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ['income', 'sex']]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True).astype(float)
    return df

# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_performance(y_true, y_pred, y_prob=None):
    m = {
        'Accuracy':  round(accuracy_score(y_true, y_pred),                   4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_true, y_pred, zero_division=0),     4),
        'F1 Score':  round(f1_score(y_true, y_pred, zero_division=0),         4),
    }
    if y_prob is not None:
        m['ROC-AUC'] = round(roc_auc_score(y_true, y_prob), 4)
    return m

def compute_fairness(y_true, y_pred, sensitive_col):
    true_df = pd.DataFrame({'income': np.array(y_true, dtype=float),
                             'sex':    np.array(sensitive_col, dtype=float)})
    pred_df = true_df.copy()
    pred_df['income'] = np.array(y_pred, dtype=float)
    ds_true = BinaryLabelDataset(df=true_df, label_names=['income'],
                                  protected_attribute_names=['sex'],
                                  favorable_label=1.0, unfavorable_label=0.0)
    ds_pred = BinaryLabelDataset(df=pred_df, label_names=['income'],
                                  protected_attribute_names=['sex'],
                                  favorable_label=1.0, unfavorable_label=0.0)
    cm = ClassificationMetric(ds_true, ds_pred,
                               privileged_groups=[{'sex': 1.0}],
                               unprivileged_groups=[{'sex': 0.0}])
    return {
        'Statistical Parity Diff': round(cm.statistical_parity_difference(), 4),
        'Equal Opportunity Diff':  round(cm.equal_opportunity_difference(),  4),
        'Avg Abs Odds Diff':       round(cm.average_abs_odds_difference(),   4),
        'Disparate Impact':        round(cm.disparate_impact(),              4),
        'Theil Index':             round(cm.theil_index(),                   4),
    }

results = {}

# ── 1. Baseline ───────────────────────────────────────────────────────────────
print("[1/4] Baseline...")
baseline = Pipeline([('preprocessor', preprocessor),
                     ('classifier',   LogisticRegression(max_iter=1000))])
baseline.fit(X_train, y_train)
bp      = np.array(baseline.predict(X_test), dtype=int)
bp_prob = baseline.predict_proba(X_test)[:, 1]
results['Baseline'] = {**compute_performance(y_test_bin, bp, bp_prob),
                       **compute_fairness(y_test_bin, bp, sex_test)}

# ── 2. Reweighting ────────────────────────────────────────────────────────────
print("[2/4] Reweighting...")
rw_model = joblib.load("model_reweighting.pkl")
rp      = np.array(rw_model.predict(X_test), dtype=int)
rp_prob = rw_model.predict_proba(X_test)[:, 1]
results['Reweighting'] = {**compute_performance(y_test_bin, rp, rp_prob),
                          **compute_fairness(y_test_bin, rp, sex_test)}

# ── 3. Adversarial Debiasing ──────────────────────────────────────────────────
print("[3/4] Adversarial Debiasing...")

meta = joblib.load("model_adversarial_meta.pkl")

# Build AIF360 datasets — must be done before fit/predict
test_df_adv  = build_numeric_df(X_test,  y_test)
train_df_adv = build_numeric_df(X_train, y_train)

train_features = meta['feature_names']

# Align both train and test to the exact columns used during original training
for col in train_features:
    if col not in test_df_adv.columns:
        test_df_adv[col] = 0.0
    if col not in train_df_adv.columns:
        train_df_adv[col] = 0.0

test_df_adv  = test_df_adv[train_features  + ['income']]
train_df_adv = train_df_adv[train_features + ['income']]

# Ensure income is binary numeric
for df in [test_df_adv, train_df_adv]:
    if df['income'].dtype == object:
        df['income'] = (df['income'].str.strip() == '>50K').astype(float)

adv_train_ds = BinaryLabelDataset(
    df=train_df_adv.astype(float),
    label_names=['income'],
    protected_attribute_names=['sex'],
    favorable_label=1.0, unfavorable_label=0.0
)
adv_test_ds = BinaryLabelDataset(
    df=test_df_adv.astype(float),
    label_names=['income'],
    protected_attribute_names=['sex'],
    favorable_label=1.0, unfavorable_label=0.0
)

adv_graph = tf.compat.v1.Graph()
adv_sess  = tf.compat.v1.Session(graph=adv_graph)

with adv_graph.as_default():
    adv_model = AdversarialDebiasing(
        privileged_groups=meta['privileged_groups'],
        unprivileged_groups=meta['unprivileged_groups'],
        scope_name=meta['scope_name'],
        sess=adv_sess,
        num_epochs=1           # ← 1 epoch just to initialize all Python attributes
    )

    # Step 1: fit for 1 epoch — this creates keep_prob, features_ph, etc.
    print("  Initializing model attributes (1 epoch)...")
    adv_model.fit(adv_train_ds)

    # Step 2: restore saved weights over the freshly initialized graph
    # This overwrites the 1-epoch weights with your fully trained 50-epoch weights
    print("  Restoring trained weights from checkpoint...")
    saver = tf.compat.v1.train.Saver()
    saver.restore(adv_sess, meta['tf_checkpoint'])

    print("  Running predictions...")
    adv_pred_ds = adv_model.predict(adv_test_ds)
    ap_bin = adv_pred_ds.labels.flatten().astype(int)

results['Adversarial Debiasing'] = {
    **compute_performance(y_test_bin, ap_bin),
    **compute_fairness(y_test_bin, ap_bin, sex_test)
}

# ── 4. Fairness Constraint ────────────────────────────────────────────────────
print("[4/4] Fairness Constraint...")
mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(max_iter=1000),
    constraints=EqualizedOdds(),
)
mitigator.fit(X_train_t, y_train, sensitive_features=sex_train)
fp     = np.array(mitigator.predict(X_test_t), dtype=int)
try:
    fp_prob = mitigator.predict_proba(X_test_t)[:, 1]
    fp_perf = compute_performance(y_test_bin, fp, fp_prob)
except Exception:
    fp_perf = compute_performance(y_test_bin, fp)
results['Fairness Constraint'] = {**fp_perf,
                                   **compute_fairness(y_test_bin, fp, sex_test)}

# ── Results table ─────────────────────────────────────────────────────────────
print("\n" + "="*75)
results_df = pd.DataFrame(results).T
print(results_df.to_string())
results_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv")
print(f"\nCSV saved → {OUTPUT_DIR}/model_comparison.csv")

# ── Charts ────────────────────────────────────────────────────────────────────
perf_cols     = [c for c in ['Accuracy','Precision','Recall','F1 Score','ROC-AUC']
                 if c in results_df.columns]
fairness_cols = ['Statistical Parity Diff','Equal Opportunity Diff',
                 'Avg Abs Odds Diff','Disparate Impact','Theil Index']

fig, axes = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('Model Comparison: Performance vs Fairness (All 4 Models)',
             fontsize=14, fontweight='bold')

results_df[perf_cols].plot(kind='bar', ax=axes[0], edgecolor='white', width=0.7)
axes[0].set_title('Predictive Performance')
axes[0].set_ylim(0, 1.1)
axes[0].set_xticklabels(results_df.index, rotation=20, ha='right')
axes[0].axhline(0.8, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
axes[0].grid(axis='y', alpha=0.3)

results_df[fairness_cols].plot(kind='bar', ax=axes[1], edgecolor='white', width=0.7)
axes[1].set_title('Fairness Metrics  (→ 0 fairer  |  Disparate Impact → 1 fairer)')
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_xticklabels(results_df.index, rotation=20, ha='right')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/performance_vs_fairness.png", dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(results_df.astype(float), annot=True, fmt='.3f',
            cmap='RdYlGn', linewidths=0.5, ax=ax)
ax.set_title('Full Metrics Heatmap — All 4 Models', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/metrics_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Charts saved → {OUTPUT_DIR}/")

# ── Verdict ───────────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("VERDICT")
print(f"  Best Accuracy        : {results_df['Accuracy'].idxmax()}")
print(f"  Best F1              : {results_df['F1 Score'].idxmax()}")
print(f"  Fairest (Stat Par.)  : {results_df['Statistical Parity Diff'].abs().idxmin()}")
print(f"  Fairest (Equal Opp.) : {results_df['Equal Opportunity Diff'].abs().idxmin()}")
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")