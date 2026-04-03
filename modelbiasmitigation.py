import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import copy

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

# FAIRNESS METRICS  (SPD, EOD, AOD, DI, Theil)

def compute_fairness(y_true, y_pred, sensitive):
    groups = sensitive.unique()
    if len(groups) < 2:
        return {k: np.nan for k in [
            "Statistical Parity Diff", "Equal Opportunity Diff",
            "Avg Abs Odds Diff", "Disparate Impact", "Theil Index"
        ]}
    g0, g1   = sorted(groups)
    mask0    = (sensitive == g0).values
    mask1    = (sensitive == g1).values
    y_pred   = np.array(y_pred)
    y_true   = np.array(y_true)

    ppr0 = y_pred[mask0].mean()
    ppr1 = y_pred[mask1].mean()
    tpr0 = y_pred[mask0 & (y_true == 1)].mean() if (y_true[mask0] == 1).any() else 0
    tpr1 = y_pred[mask1 & (y_true == 1)].mean() if (y_true[mask1] == 1).any() else 0
    fpr0 = y_pred[mask0 & (y_true == 0)].mean() if (y_true[mask0] == 0).any() else 0
    fpr1 = y_pred[mask1 & (y_true == 0)].mean() if (y_true[mask1] == 0).any() else 0

    benefit = y_pred.astype(float)
    mu      = benefit.mean()
    theil   = np.mean(
        np.where(benefit > 0, (benefit / mu) * np.log(benefit / mu + 1e-9), 0)
    ) if mu > 0 else np.nan

    return {
        "Statistical Parity Diff": round(ppr1 - ppr0, 4),
        "Equal Opportunity Diff":  round(tpr1 - tpr0, 4),
        "Avg Abs Odds Diff":       round((abs(tpr1 - tpr0) + abs(fpr1 - fpr0)) / 2, 4),
        "Disparate Impact":        round(ppr1 / ppr0 if ppr0 > 0 else np.nan, 4),
        "Theil Index":             round(theil, 4),
    }


#   Flip Rate → lower = fairer  (model ignores sex)
#   CFB Score → higher = fairer (1 - flip_rate)

def compute_counterfactual_bias(clf, X_te_with_sex, sex_col_idx):
    """
    Parameters
    ----------
    clf             : fitted classifier
    X_te_with_sex   : numpy array where column sex_col_idx is the sex feature (0/1)
    sex_col_idx     : integer index of the sex column in X_te_with_sex

    Returns
    -------
    dict with Counterfactual Flip Rate and CFB Score
    """
    if not hasattr(clf, "predict"):
        return {"Counterfactual Flip Rate": np.nan, "CFB Score": np.nan}

    original_preds = clf.predict(X_te_with_sex)

    X_flipped = X_te_with_sex.copy()
    X_flipped[:, sex_col_idx] = 1 - X_flipped[:, sex_col_idx]   # flip 0↔1
    flipped_preds = clf.predict(X_flipped)

    flip_rate = float(np.mean(original_preds != flipped_preds))
    return {
        "Counterfactual Flip Rate": round(flip_rate, 4),
        "CFB Score":                round(1 - flip_rate, 4),
    }


def compute_reweighting(y_train, sensitive_train):
    y_train         = y_train.reset_index(drop=True)
    sensitive_train = sensitive_train.reset_index(drop=True)
    weights  = np.ones(len(y_train))
    n_total  = len(y_train)
    for g in sensitive_train.unique():
        for label in [0, 1]:
            mask   = ((sensitive_train == g) & (y_train == label)).values
            n_cell = mask.sum()
            if n_cell > 0:
                n_group = (sensitive_train == g).sum()
                n_label = (y_train == label).sum()
                expected        = (n_group / n_total) * (n_label / n_total) * n_total
                weights[mask]   = expected / n_cell
    return weights


def _threshold_predict(clf, X_sc, sensitive, t0, t1):
    """Apply per-group decision thresholds."""
    if not hasattr(clf, "predict_proba"):
        return clf.predict(X_sc), None
    probs  = clf.predict_proba(X_sc)[:, 1]
    preds  = np.zeros(len(probs), dtype=int)
    m0     = (sensitive == 0).values
    m1     = (sensitive == 1).values
    preds[m0] = (probs[m0] >= t0).astype(int)
    preds[m1] = (probs[m1] >= t1).astype(int)
    return preds, probs


def find_adversarial_thresholds(clf, X_val_sc, y_val, sensitive_val):
    """Grid-search thresholds to minimise |FPR_group1 - FPR_group0|."""
    if not hasattr(clf, "predict_proba"):
        return 0.5, 0.5
    probs      = clf.predict_proba(X_val_sc)[:, 1]
    best       = (1.0, 0.5, 0.5)
    thresholds = np.arange(0.2, 0.81, 0.05)
    y_arr      = np.array(y_val)
    for t0 in thresholds:
        for t1 in thresholds:
            preds      = np.zeros(len(probs), dtype=int)
            m0         = (sensitive_val == 0).values
            m1         = (sensitive_val == 1).values
            preds[m0]  = (probs[m0] >= t0).astype(int)
            preds[m1]  = (probs[m1] >= t1).astype(int)
            fpr0 = preds[m0 & (y_arr == 0)].mean() if (y_arr[m0] == 0).any() else 0
            fpr1 = preds[m1 & (y_arr == 0)].mean() if (y_arr[m1] == 0).any() else 0
            gap  = abs(fpr0 - fpr1)
            if gap < best[0]:
                best = (gap, t0, t1)
    return best[1], best[2]


def find_fairness_thresholds(clf, X_val_sc, y_val, sensitive_val):
    """Grid-search thresholds to minimise |PPR_group1 - PPR_group0|."""
    if not hasattr(clf, "predict_proba"):
        return 0.5, 0.5
    probs      = clf.predict_proba(X_val_sc)[:, 1]
    best       = (1.0, 0.5, 0.5)
    thresholds = np.arange(0.2, 0.81, 0.05)
    for t0 in thresholds:
        for t1 in thresholds:
            preds      = np.zeros(len(probs), dtype=int)
            m0         = (sensitive_val == 0).values
            m1         = (sensitive_val == 1).values
            preds[m0]  = (probs[m0] >= t0).astype(int)
            preds[m1]  = (probs[m1] >= t1).astype(int)
            gap = abs(preds[m0].mean() - preds[m1].mean())
            if gap < best[0]:
                best = (gap, t0, t1)
    return best[1], best[2]


# WITHIN-MODEL EVALUATION LOOP

VARIANTS = ["Baseline", "Reweighting", "Adversarial Debiasing", "Fairness Constraint"]


def evaluate_within_model(clf_template, clf_name, X, y, sensitive, n_splits=5):
    scaler = StandardScaler()
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = {v: {"perf": [], "fair": [], "cfb": []} for v in VARIANTS}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X.iloc[train_idx],  X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx],  y.iloc[test_idx]
        s_tr = sensitive.iloc[train_idx].reset_index(drop=True)
        s_te = sensitive.iloc[test_idx].reset_index(drop=True)

        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        # Append sex as an extra column so CFB can flip it independently
        # of whatever other features the scaler touched.
        X_tr_sc_sex = np.hstack([X_tr_sc, s_tr.values.reshape(-1, 1)])
        X_te_sc_sex = np.hstack([X_te_sc, s_te.values.reshape(-1, 1)])
        sex_col_idx = X_tr_sc_sex.shape[1] - 1   # last column = sex

        # Validation split (last 20% of training fold)
        val_size    = int(0.2 * len(train_idx))
        X_val_sex   = X_tr_sc_sex[-val_size:]
        y_val       = y_tr.iloc[-val_size:].reset_index(drop=True)
        s_val       = s_tr.iloc[-val_size:].reset_index(drop=True)
        X_tr_sex_   = X_tr_sc_sex[:-val_size]
        y_tr_       = y_tr.iloc[:-val_size].reset_index(drop=True)
        s_tr_       = s_tr.iloc[:-val_size].reset_index(drop=True)

        y_te_arr = np.array(y_te)

        #Baseline 
        clf = copy.deepcopy(clf_template)
        clf.fit(X_tr_sex_, y_tr_)
        y_pred = clf.predict(X_te_sc_sex)
        y_prob = clf.predict_proba(X_te_sc_sex)[:, 1] if hasattr(clf, "predict_proba") else None
        fold_results["Baseline"]["perf"].append(compute_performance(y_te_arr, y_pred, y_prob))
        fold_results["Baseline"]["fair"].append(compute_fairness(y_te_arr, y_pred, s_te))
        fold_results["Baseline"]["cfb"].append(
            compute_counterfactual_bias(clf, X_te_sc_sex, sex_col_idx))

        #Reweighting 
        clf_rw  = copy.deepcopy(clf_template)
        weights = compute_reweighting(y_tr_, s_tr_)
        try:
            clf_rw.fit(X_tr_sex_, y_tr_, sample_weight=weights)
        except TypeError:
            clf_rw.fit(X_tr_sex_, y_tr_)
        y_pred_rw = clf_rw.predict(X_te_sc_sex)
        y_prob_rw = clf_rw.predict_proba(X_te_sc_sex)[:, 1] if hasattr(clf_rw, "predict_proba") else None
        fold_results["Reweighting"]["perf"].append(compute_performance(y_te_arr, y_pred_rw, y_prob_rw))
        fold_results["Reweighting"]["fair"].append(compute_fairness(y_te_arr, y_pred_rw, s_te))
        fold_results["Reweighting"]["cfb"].append(
            compute_counterfactual_bias(clf_rw, X_te_sc_sex, sex_col_idx))

        #Adversarial Debiasing
        clf_adv = copy.deepcopy(clf_template)
        clf_adv.fit(X_tr_sex_, y_tr_)
        t0_adv, t1_adv = find_adversarial_thresholds(clf_adv, X_val_sex, y_val, s_val)
        y_pred_adv, y_prob_adv = _threshold_predict(clf_adv, X_te_sc_sex, s_te, t0_adv, t1_adv)
        if y_prob_adv is None and hasattr(clf_adv, "predict_proba"):
            y_prob_adv = clf_adv.predict_proba(X_te_sc_sex)[:, 1]
        fold_results["Adversarial Debiasing"]["perf"].append(
            compute_performance(y_te_arr, y_pred_adv, y_prob_adv))
        fold_results["Adversarial Debiasing"]["fair"].append(
            compute_fairness(y_te_arr, y_pred_adv, s_te))
        fold_results["Adversarial Debiasing"]["cfb"].append(
            compute_counterfactual_bias(clf_adv, X_te_sc_sex, sex_col_idx))

        #Fairness Constraint
        clf_fc = copy.deepcopy(clf_template)
        clf_fc.fit(X_tr_sex_, y_tr_)
        t0_fc, t1_fc = find_fairness_thresholds(clf_fc, X_val_sex, y_val, s_val)
        y_pred_fc, y_prob_fc = _threshold_predict(clf_fc, X_te_sc_sex, s_te, t0_fc, t1_fc)
        if y_prob_fc is None and hasattr(clf_fc, "predict_proba"):
            y_prob_fc = clf_fc.predict_proba(X_te_sc_sex)[:, 1]
        fold_results["Fairness Constraint"]["perf"].append(
            compute_performance(y_te_arr, y_pred_fc, y_prob_fc))
        fold_results["Fairness Constraint"]["fair"].append(
            compute_fairness(y_te_arr, y_pred_fc, s_te))
        fold_results["Fairness Constraint"]["cfb"].append(
            compute_counterfactual_bias(clf_fc, X_te_sc_sex, sex_col_idx))

        print(f"    Fold {fold}  "
              f"Base Acc={fold_results['Baseline']['perf'][-1]['Accuracy']:.4f}  "
              f"Base FlipRate={fold_results['Baseline']['cfb'][-1]['Counterfactual Flip Rate']:.4f}  "
              f"RW FlipRate={fold_results['Reweighting']['cfb'][-1]['Counterfactual Flip Rate']:.4f}  "
              f"Adv FlipRate={fold_results['Adversarial Debiasing']['cfb'][-1]['Counterfactual Flip Rate']:.4f}  "
              f"FC FlipRate={fold_results['Fairness Constraint']['cfb'][-1]['Counterfactual Flip Rate']:.4f}")

    # Average across folds
    rows = {}
    for v in VARIANTS:
        avg_perf = {k: round(np.nanmean([f[k] for f in fold_results[v]["perf"]]), 4)
                    for k in fold_results[v]["perf"][0]}
        avg_fair = {k: round(np.nanmean([f[k] for f in fold_results[v]["fair"]]), 4)
                    for k in fold_results[v]["fair"][0]}
        avg_cfb  = {k: round(np.nanmean([f[k] for f in fold_results[v]["cfb"]]),  4)
                    for k in fold_results[v]["cfb"][0]}
        rows[v] = {**avg_perf, **avg_fair, **avg_cfb}

    return pd.DataFrame(rows).T


VARIANT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def plot_within_model(df, model_name):
    perf_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    fair_cols = [
        "Statistical Parity Diff", "Equal Opportunity Diff",
        "Avg Abs Odds Diff", "Disparate Impact", "Theil Index",
        "Counterfactual Flip Rate", "CFB Score",         
    ]

    variants = df.index.tolist()
    colors   = VARIANT_COLORS[:len(variants)]
    x        = np.arange(len(variants))
    safe     = model_name.replace(" ", "_").lower()

    # Performance chart
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
    plt.savefig(f"{OUT_DIR}/{safe}_performance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fairness chart 
    fig, axes = plt.subplots(1, len(fair_cols), figsize=(28, 4))
    fig.suptitle(f"{model_name} — Fairness by Variant (incl. Counterfactual Bias)",
                 fontsize=13, fontweight="bold")
    for ax, col in zip(axes, fair_cols):
        vals = df[col].fillna(0).values
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(col, fontsize=8)
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

    # Accuracy vs Statistical Parity scatter
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{model_name} — Counterfactual Bias Detection", fontsize=12, fontweight="bold")

    flip_vals = df["Counterfactual Flip Rate"].fillna(0).values
    cfb_vals  = df["CFB Score"].fillna(0).values

    for ax, vals, title, note in zip(
        axes,
        [flip_vals, cfb_vals],
        ["Counterfactual Flip Rate", "CFB Score"],
        ["lower = fairer", "higher = fairer"]
    ):
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{title}\n({note})", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{safe}_cfb.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"    Charts saved → {OUT_DIR}/{safe}_*.png")



def plot_combined_heatmap(all_results):
    key_metrics = [
        "Accuracy", "F1 Score", "ROC-AUC",
        "Statistical Parity Diff", "Equal Opportunity Diff",
        "Avg Abs Odds Diff", "CFB Score",                  
    ]

    rows, row_labels = [], []
    for model_name, df in all_results.items():
        for variant in df.index:
            rows.append(df.loc[variant, key_metrics].values.astype(float))
            row_labels.append(f"{model_name}\n{variant}")

    matrix      = np.array(rows)
    norm_matrix = np.zeros_like(matrix)

    for j, col in enumerate(key_metrics):
        col_vals = matrix[:, j]
        if col in ["Accuracy", "F1 Score", "ROC-AUC", "CFB Score"]:
            # higher is better — normalise so best = 1
            vmin, vmax = np.nanmin(col_vals), np.nanmax(col_vals)
            norm_matrix[:, j] = (col_vals - vmin) / (vmax - vmin + 1e-9)
        else:
            # lower absolute value is better — normalise so closest-to-0 = 1
            abs_vals   = np.abs(col_vals)
            vmin, vmax = np.nanmin(abs_vals), np.nanmax(abs_vals)
            norm_matrix[:, j] = 1 - (abs_vals - vmin) / (vmax - vmin + 1e-9)

    fig, ax = plt.subplots(figsize=(14, 0.55 * len(row_labels) + 2))
    im = ax.imshow(norm_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(key_metrics)))
    ax.set_xticklabels(key_metrics, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5)

    for i in range(len(row_labels)):
        for j in range(len(key_metrics)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.3f}" if not np.isnan(val) else "N/A",
                    ha="center", va="center", fontsize=7, color="black")

    n_variants = len(VARIANTS)
    for k in range(1, len(all_results)):
        ax.axhline(k * n_variants - 0.5, color="white", linewidth=2)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02,
                 label="Normalised score (green = better)")
    ax.set_title(
        "All Models × All Variants — Key Metric Heatmap\n"
        "(CFB Score: higher = fairer  |  Fairness diffs: closer-to-0 = fairer)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/combined_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Combined heatmap saved → {OUT_DIR}/combined_heatmap.png")


# CFB SUMMARY CHART — all models, Baseline vs best mitigated variant

def plot_cfb_summary(all_results):
    """
    Side-by-side bar chart showing Counterfactual Flip Rate for every
    model's Baseline vs the variant with the lowest flip rate.
    """
    model_names, base_flips, best_flips, best_labels = [], [], [], []

    for model_name, df in all_results.items():
        base_flip = df.loc["Baseline", "Counterfactual Flip Rate"]
        other     = df.drop(index="Baseline")
        best_var  = other["Counterfactual Flip Rate"].idxmin()
        best_flip = other.loc[best_var, "Counterfactual Flip Rate"]
        model_names.append(model_name)
        base_flips.append(base_flip)
        best_flips.append(best_flip)
        best_labels.append(best_var)

    x      = np.arange(len(model_names))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))

    bars1 = ax.bar(x - width / 2, base_flips, width, label="Baseline",
                   color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width / 2, best_flips, width,
                   label="Best mitigated variant", color="#55A868", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Counterfactual Flip Rate  (lower = fairer)", fontsize=9)
    ax.set_title("Counterfactual Bias — Baseline vs Best Variant (per model)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(base_flips + best_flips) * 1.2 + 0.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    for bar, v in zip(bars1, base_flips):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, v, lbl in zip(bars2, best_flips, best_labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}\n({lbl[:3]})", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/cfb_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  CFB summary chart saved → {OUT_DIR}/cfb_summary.png")



def print_model_results(model_name, df):
    perf_cols    = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    fairness_cols = [
        "Statistical Parity Diff", "Equal Opportunity Diff",
        "Avg Abs Odds Diff", "Disparate Impact", "Theil Index",
        "Counterfactual Flip Rate", "CFB Score",
    ]
    print(f"\n{'='*100}")
    print(f"  {model_name.upper()}")
    print(f"{'='*100}")
    print("\n  PERFORMANCE:")
    print(df[perf_cols].to_string())
    print("\n  FAIRNESS + COUNTERFACTUAL BIAS:")
    print(df[fairness_cols].to_string())
    print(f"\n  Best Accuracy          : {df['Accuracy'].idxmax()}")
    print(f"  Best F1                : {df['F1 Score'].idxmax()}")
    print(f"  Fairest (StatPar)      : {df['Statistical Parity Diff'].abs().idxmin()}")
    print(f"  Fairest (EqOpp)        : {df['Equal Opportunity Diff'].abs().idxmin()}")
    print(f"  Fairest (CFB Score)    : {df['CFB Score'].idxmax()}")
    print(f"  Lowest Flip Rate       : {df['Counterfactual Flip Rate'].idxmin()}")



def save_all(all_results):
    combined = []
    for model_name, df in all_results.items():
        df_copy = df.copy()
        df_copy.insert(0, "Model", model_name)
        df_copy.insert(1, "Variant", df_copy.index)
        combined.append(df_copy)
    out  = pd.concat(combined).reset_index(drop=True)
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
            eval_metric="logloss", random_state=42, n_jobs=-1
        )
    return models


if __name__ == "__main__":
    print("\n" + "="*65)
    print("  WITHIN EACH MODEL — FAIRNESS COMPARISON")
    print("  Baseline | Reweighting | Adversarial | Fairness Constraint")
    print("  Metrics: SPD, EOD, AOD, DI, Theil + Counterfactual Bias")
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
    plot_cfb_summary(all_results)

    print("\n" + "="*65)
    print("  All outputs saved to: evaluation_results/within_model/")
    print("  Per-model files: *_performance.png, *_fairness.png,")
    print("                   *_tradeoff_scatter.png, *_cfb.png")
    print("  Summary files:   combined_heatmap.png, cfb_summary.png,")
    print("                   within_model_comparison.csv")
    print("="*65)