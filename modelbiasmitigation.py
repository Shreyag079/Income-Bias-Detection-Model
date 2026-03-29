
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

OUT_DIR = "evaluation_results/within_model"
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv("census_income.csv")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["income", "sex"]]
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    X = df.drop(columns=["income", "sex"])
    y = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)
    sensitive = df["sex"].map({"Male": 1, "Female": 0})
    return X, y, sensitive



def compute_performance(y_true, y_pred, y_prob=None):
    m = {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        try:
            m["ROC-AUC"] = round(roc_auc_score(y_true, y_prob), 4)
        except Exception:
            m["ROC-AUC"] = np.nan
    else:
        m["ROC-AUC"] = np.nan
    return m


def compute_fairness(y_true, y_pred, sensitive):
    groups = sensitive.unique()
    if len(groups) < 2:
        return {k: np.nan for k in [
            "Statistical Parity Diff", "Equal Opportunity Diff",
            "Avg Abs Odds Diff", "Disparate Impact", "Theil Index"
        ]}
    g0, g1 = sorted(groups)
    mask0, mask1 = (sensitive == g0).values, (sensitive == g1).values

    ppr0 = y_pred[mask0].mean()
    ppr1 = y_pred[mask1].mean()
    tpr0 = y_pred[mask0 & (y_true == 1)].mean() if (y_true[mask0] == 1).any() else 0
    tpr1 = y_pred[mask1 & (y_true == 1)].mean() if (y_true[mask1] == 1).any() else 0
    fpr0 = y_pred[mask0 & (y_true == 0)].mean() if (y_true[mask0] == 0).any() else 0
    fpr1 = y_pred[mask1 & (y_true == 0)].mean() if (y_true[mask1] == 0).any() else 0

    benefit = y_pred.astype(float)
    mu = benefit.mean()
    theil = np.mean(
        np.where(benefit > 0, (benefit / mu) * np.log(benefit / mu + 1e-9), 0)
    ) if mu > 0 else np.nan

    return {
        "Statistical Parity Diff": round(ppr1 - ppr0, 4),
        "Equal Opportunity Diff":  round(tpr1 - tpr0, 4),
        "Avg Abs Odds Diff":       round((abs(tpr1 - tpr0) + abs(fpr1 - fpr0)) / 2, 4),
        "Disparate Impact":        round(ppr1 / ppr0 if ppr0 > 0 else np.nan, 4),
        "Theil Index":             round(theil, 4),
    }


# IN-PROCESSING METHODS

def compute_reweighting(y_train, sensitive_train):
    
    y_train        = y_train.reset_index(drop=True)
    sensitive_train = sensitive_train.reset_index(drop=True)

    weights = np.ones(len(y_train))
    groups  = sensitive_train.unique()
    n_total = len(y_train)
    for g in groups:
        for label in [0, 1]:
            mask = ((sensitive_train == g) & (y_train == label)).values
            n_cell  = mask.sum()
            n_group = (sensitive_train == g).sum()
            n_label = (y_train == label).sum()
            if n_cell > 0:
                expected = (n_group / n_total) * (n_label / n_total) * n_total
                weights[mask] = expected / n_cell
    return weights



def adversarial_predict(clf, X_te_sc, sensitive_te, threshold_0, threshold_1):
    probs  = clf.predict_proba(X_te_sc)[:, 1]
    preds  = np.zeros(len(probs), dtype=int)
    mask0  = (sensitive_te == 0).values
    mask1  = (sensitive_te == 1).values
    preds[mask0] = (probs[mask0] >= threshold_0).astype(int)
    preds[mask1] = (probs[mask1] >= threshold_1).astype(int)
    return preds, probs


def find_adversarial_thresholds(clf, X_val_sc, y_val, sensitive_val):
    """
    Grid-search per-group thresholds to minimise |FPR_group1 - FPR_group0|.
    Falls back to 0.5/0.5 if model has no predict_proba.
    """
    if not hasattr(clf, "predict_proba"):
        return 0.5, 0.5

    probs   = clf.predict_proba(X_val_sc)[:, 1]
    best    = (1.0, 0.5, 0.5)
    thresholds = np.arange(0.2, 0.81, 0.05)

    for t0 in thresholds:
        for t1 in thresholds:
            preds = np.zeros(len(probs), dtype=int)
            m0 = (sensitive_val == 0).values
            m1 = (sensitive_val == 1).values
            preds[m0] = (probs[m0] >= t0).astype(int)
            preds[m1] = (probs[m1] >= t1).astype(int)

            fpr0 = preds[m0 & (np.array(y_val) == 0)].mean() if (np.array(y_val)[m0] == 0).any() else 0
            fpr1 = preds[m1 & (np.array(y_val) == 0)].mean() if (np.array(y_val)[m1] == 0).any() else 0
            gap  = abs(fpr0 - fpr1)
            if gap < best[0]:
                best = (gap, t0, t1)

    return best[1], best[2]



def fairness_constraint_predict(clf, X_te_sc, sensitive_te, threshold_0, threshold_1):
    return adversarial_predict(clf, X_te_sc, sensitive_te, threshold_0, threshold_1)


def find_fairness_thresholds(clf, X_val_sc, y_val, sensitive_val):
    """
    Grid-search per-group thresholds to minimise |PPR_group1 - PPR_group0|
    (Statistical Parity constraint).
    """
    if not hasattr(clf, "predict_proba"):
        return 0.5, 0.5

    probs      = clf.predict_proba(X_val_sc)[:, 1]
    best       = (1.0, 0.5, 0.5)
    thresholds = np.arange(0.2, 0.81, 0.05)

    for t0 in thresholds:
        for t1 in thresholds:
            preds = np.zeros(len(probs), dtype=int)
            m0 = (sensitive_val == 0).values
            m1 = (sensitive_val == 1).values
            preds[m0] = (probs[m0] >= t0).astype(int)
            preds[m1] = (probs[m1] >= t1).astype(int)

            ppr0 = preds[m0].mean()
            ppr1 = preds[m1].mean()
            gap  = abs(ppr0 - ppr1)
            if gap < best[0]:
                best = (gap, t0, t1)

    return best[1], best[2]


# WITHIN-MODEL EVALUATION LOOP

VARIANTS = ["Baseline", "Reweighting", "Adversarial Debiasing", "Fairness Constraint"]


def evaluate_within_model(clf_template, clf_name, X, y, sensitive, n_splits=5):
    scaler  = StandardScaler()
    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = {v: {"perf": [], "fair": []} for v in VARIANTS}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X.iloc[train_idx],   X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx],   y.iloc[test_idx]
        s_tr       = sensitive.iloc[train_idx].reset_index(drop=True)
        s_te       = sensitive.iloc[test_idx].reset_index(drop=True)

        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        val_size  = int(0.2 * len(train_idx))
        X_val_sc  = X_tr_sc[-val_size:]
        y_val     = y_tr.iloc[-val_size:].reset_index(drop=True)
        s_val     = s_tr.iloc[-val_size:].reset_index(drop=True)
        X_tr_sc_  = X_tr_sc[:-val_size]
        y_tr_     = y_tr.iloc[:-val_size].reset_index(drop=True)
        s_tr_     = s_tr.iloc[:-val_size].reset_index(drop=True)

        y_te_arr = np.array(y_te)

        # Baseline 
        import copy
        clf = copy.deepcopy(clf_template)
        clf.fit(X_tr_sc_, y_tr_)
        y_pred = clf.predict(X_te_sc)
        y_prob = clf.predict_proba(X_te_sc)[:, 1] if hasattr(clf, "predict_proba") else None
        fold_results["Baseline"]["perf"].append(compute_performance(y_te_arr, y_pred, y_prob))
        fold_results["Baseline"]["fair"].append(compute_fairness(y_te_arr, y_pred, s_te))

        # Reweighting 
        clf_rw = copy.deepcopy(clf_template)
        weights = compute_reweighting(y_tr_, s_tr_)
        try:
            clf_rw.fit(X_tr_sc_, y_tr_, sample_weight=weights)
        except TypeError:
            
            clf_rw.fit(X_tr_sc_, y_tr_)
        y_pred_rw = clf_rw.predict(X_te_sc)
        y_prob_rw = clf_rw.predict_proba(X_te_sc)[:, 1] if hasattr(clf_rw, "predict_proba") else None
        fold_results["Reweighting"]["perf"].append(compute_performance(y_te_arr, y_pred_rw, y_prob_rw))
        fold_results["Reweighting"]["fair"].append(compute_fairness(y_te_arr, y_pred_rw, s_te))

        # Adversarial Debiasing 
        clf_adv = copy.deepcopy(clf_template)
        clf_adv.fit(X_tr_sc_, y_tr_)
        t0_adv, t1_adv = find_adversarial_thresholds(clf_adv, X_val_sc, y_val, s_val)
        y_pred_adv, y_prob_adv = adversarial_predict(clf_adv, X_te_sc, s_te, t0_adv, t1_adv)
        fold_results["Adversarial Debiasing"]["perf"].append(
            compute_performance(y_te_arr, y_pred_adv,
                                clf_adv.predict_proba(X_te_sc)[:, 1] if hasattr(clf_adv, "predict_proba") else None))
        fold_results["Adversarial Debiasing"]["fair"].append(
            compute_fairness(y_te_arr, y_pred_adv, s_te))

        # Fairness Constraint 
        clf_fc = copy.deepcopy(clf_template)
        clf_fc.fit(X_tr_sc_, y_tr_)
        t0_fc, t1_fc = find_fairness_thresholds(clf_fc, X_val_sc, y_val, s_val)
        y_pred_fc, _ = fairness_constraint_predict(clf_fc, X_te_sc, s_te, t0_fc, t1_fc)
        fold_results["Fairness Constraint"]["perf"].append(
            compute_performance(y_te_arr, y_pred_fc,
                                clf_fc.predict_proba(X_te_sc)[:, 1] if hasattr(clf_fc, "predict_proba") else None))
        fold_results["Fairness Constraint"]["fair"].append(
            compute_fairness(y_te_arr, y_pred_fc, s_te))

        print(f"    Fold {fold}  "
              f"Base Acc={fold_results['Baseline']['perf'][-1]['Accuracy']:.4f}  "
              f"RW Acc={fold_results['Reweighting']['perf'][-1]['Accuracy']:.4f}  "
              f"Adv StatPar={fold_results['Adversarial Debiasing']['fair'][-1]['Statistical Parity Diff']:.4f}  "
              f"FC StatPar={fold_results['Fairness Constraint']['fair'][-1]['Statistical Parity Diff']:.4f}")

    # Average across folds
    rows = {}
    for v in VARIANTS:
        avg_perf = {k: round(np.nanmean([f[k] for f in fold_results[v]["perf"]]), 4)
                    for k in fold_results[v]["perf"][0]}
        avg_fair = {k: round(np.nanmean([f[k] for f in fold_results[v]["fair"]]), 4)
                    for k in fold_results[v]["fair"][0]}
        rows[v] = {**avg_perf, **avg_fair}

    return pd.DataFrame(rows).T


# CHARTS — per model

VARIANT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

def plot_within_model(df, model_name):
    perf_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    fair_cols = ["Statistical Parity Diff", "Equal Opportunity Diff",
                 "Avg Abs Odds Diff", "Disparate Impact", "Theil Index"]
    variants  = df.index.tolist()
    colors    = VARIANT_COLORS[:len(variants)]
    x         = np.arange(len(variants))

    # Performance 
    fig, axes = plt.subplots(1, len(perf_cols), figsize=(18, 4))
    fig.suptitle(f"{model_name} — Performance by Variant", fontsize=13, fontweight="bold")
    for ax, col in zip(axes, perf_cols):
        vals = df[col].fillna(0).values
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(col, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=7.5)
        ax.set_ylim(0, 1.08)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    safe = model_name.replace(" ", "_").lower()
    plt.savefig(f"{OUT_DIR}/{safe}_performance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fairness 
    fig, axes = plt.subplots(1, len(fair_cols), figsize=(22, 4))
    fig.suptitle(f"{model_name} — Fairness by Variant", fontsize=13, fontweight="bold")
    for ax, col in zip(axes, fair_cols):
        vals = df[col].fillna(0).values
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(col, fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=7.5)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.002 if v >= 0 else -0.008),
                    f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{safe}_fairness.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Accuracy vs Stat Parity scatter within each model
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, v in enumerate(variants):
        ax.scatter(
            abs(float(df.loc[v, "Statistical Parity Diff"])),
            float(df.loc[v, "Accuracy"]),
            color=colors[i], s=120, zorder=3, label=v
        )
        ax.annotate(v, (
            abs(float(df.loc[v, "Statistical Parity Diff"])),
            float(df.loc[v, "Accuracy"])
        ), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
    ax.set_xlabel("|Statistical Parity Diff|  (lower = fairer)", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.set_title(f"{model_name} — Accuracy vs Fairness Trade-off", fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{safe}_tradeoff_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"    Charts saved → {OUT_DIR}/{safe}_*.png")


# COMBINED HEATMAP 

def plot_combined_heatmap(all_results):
    """
    Rows = Model × Variant combinations
    Cols = Key metrics
    Color = normalised 0-1 per column (green = good, red = bad)
    """
    key_metrics = ["Accuracy", "F1 Score", "ROC-AUC",
                   "Statistical Parity Diff", "Equal Opportunity Diff", "Avg Abs Odds Diff"]

    rows, row_labels = [], []
    for model_name, df in all_results.items():
        for variant in df.index:
            rows.append(df.loc[variant, key_metrics].values.astype(float))
            row_labels.append(f"{model_name}\n{variant}")

    matrix = np.array(rows)

    norm_matrix = np.zeros_like(matrix)
    for j, col in enumerate(key_metrics):
        col_vals = matrix[:, j]
        if col in ["Accuracy", "F1 Score", "ROC-AUC"]:
            vmin, vmax = np.nanmin(col_vals), np.nanmax(col_vals)
            norm_matrix[:, j] = (col_vals - vmin) / (vmax - vmin + 1e-9)
        else:
            
            abs_vals = np.abs(col_vals)
            vmin, vmax = np.nanmin(abs_vals), np.nanmax(abs_vals)
            norm_matrix[:, j] = 1 - (abs_vals - vmin) / (vmax - vmin + 1e-9)

    fig, ax = plt.subplots(figsize=(13, 0.55 * len(row_labels) + 2))
    im = ax.imshow(norm_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(key_metrics)))
    ax.set_xticklabels(key_metrics, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5)


    for i in range(len(row_labels)):
        for j in range(len(key_metrics)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, color="black")

  
    n_variants = len(VARIANTS)
    for k in range(1, len(all_results)):
        ax.axhline(k * n_variants - 0.5, color="white", linewidth=2)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02,
                 label="Normalised score (green = better)")
    ax.set_title("All Models × All Variants — Key Metric Heatmap", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/combined_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved → {OUT_DIR}/combined_heatmap.png")


def print_model_results(model_name, df):
    perf_cols    = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    fairness_cols = ["Statistical Parity Diff", "Equal Opportunity Diff",
                     "Avg Abs Odds Diff", "Disparate Impact", "Theil Index"]
    print(f"\n{'='*100}")
    print(f"  {model_name.upper()}")
    print(f"{'='*100}")
    print("\n  PERFORMANCE:")
    print(df[perf_cols].to_string())
    print("\n  FAIRNESS:")
    print(df[fairness_cols].to_string())
    print(f"\n  Best Accuracy    : {df['Accuracy'].idxmax()}")
    print(f"  Best F1          : {df['F1 Score'].idxmax()}")
    print(f"  Fairest (StatPar): {df['Statistical Parity Diff'].abs().idxmin()}")
    print(f"  Fairest (EqOpp)  : {df['Equal Opportunity Diff'].abs().idxmin()}")


def save_all(all_results):
    combined = []
    for model_name, df in all_results.items():
        df_copy = df.copy()
        df_copy.insert(0, "Model", model_name)
        df_copy.insert(1, "Variant", df_copy.index)
        combined.append(df_copy)
    out = pd.concat(combined).reset_index(drop=True)
    path = f"{OUT_DIR}/within_model_comparison.csv"
    out.to_csv(path, index=False)
    print(f"\n  Combined CSV saved → {path}")



def build_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=5,
            random_state=42, class_weight="balanced"
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, probability=True,
            random_state=42, class_weight="balanced"
        ),
        "Naive Bayes": GaussianNB(),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
    return models


if __name__ == "__main__":
    print("\n" + "="*65)
    print("  WITHIN EACH MODEL FAIRNESS COMPARISON")
    print("  (Baseline vs Reweighting vs Adversarial vs Fairness Constraint)")
    print("="*65)

    X, y, sensitive = load_data()
    print(f"\n  Dataset: {X.shape[0]} rows × {X.shape[1]} features")

    models      = build_models()
    all_results = {}

    for model_name, clf in models.items():
        print(f"\n{'─'*65}")
        print(f"  ▶  {model_name}")
        print(f"{'─'*65}")
        df = evaluate_within_model(clf, model_name, X, y, sensitive, n_splits=5)
        all_results[model_name] = df
        print_model_results(model_name, df)
        plot_within_model(df, model_name)

    save_all(all_results)
    plot_combined_heatmap(all_results)

    print("\n" + "="*65)
    print("  All outputs saved to: evaluation_results/within_model/")
    print("="*65)