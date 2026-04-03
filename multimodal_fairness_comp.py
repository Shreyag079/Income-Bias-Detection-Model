
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
import json

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[INFO] XGBoost not installed. Run: pip install xgboost")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.datasets import make_classification

# ─── Output directory ───────────────────────────────────────────────────────
OUT_DIR = "evaluation_results"
os.makedirs(OUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING  (replace this block with your real dataset)
# ════════════════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv("census_income.csv")
    
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["income", "sex"]]
    
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    X = df.drop(columns=["income", "sex"])
    y = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)
    sensitive = df["sex"].map({"Male": 1, "Female": 0})  # ensure numeric
    
    return X, y, sensitive


# ════════════════════════════════════════════════════════════════════════════
# 2. FAIRNESS METRICS (same as your logistic regression evaluation)
# ════════════════════════════════════════════════════════════════════════════

def compute_fairness_metrics(y_true, y_pred, sensitive):
    """
    Returns a dict with:
      - statistical_parity_diff
      - equal_opportunity_diff
      - avg_abs_odds_diff
      - disparate_impact
      - theil_index
    """
    groups = sensitive.unique()
    if len(groups) < 2:
        return {k: np.nan for k in [
            "Statistical Parity Diff", "Equal Opportunity Diff",
            "Avg Abs Odds Diff", "Disparate Impact", "Theil Index"
        ]}

    g0, g1 = sorted(groups)
    mask0 = (sensitive == g0).values
    mask1 = (sensitive == g1).values

    # Positive prediction rate per group
    ppr0 = y_pred[mask0].mean()
    ppr1 = y_pred[mask1].mean()

    # True positive rate per group (equal opportunity)
    tpr0 = y_pred[mask0 & (y_true == 1)].mean() if (y_true[mask0] == 1).any() else 0
    tpr1 = y_pred[mask1 & (y_true == 1)].mean() if (y_true[mask1] == 1).any() else 0

    # False positive rate per group
    fpr0 = y_pred[mask0 & (y_true == 0)].mean() if (y_true[mask0] == 0).any() else 0
    fpr1 = y_pred[mask1 & (y_true == 0)].mean() if (y_true[mask1] == 0).any() else 0

    stat_parity_diff = ppr1 - ppr0
    equal_opp_diff   = tpr1 - tpr0
    avg_abs_odds     = (abs(tpr1 - tpr0) + abs(fpr1 - fpr0)) / 2
    disp_impact      = (ppr1 / ppr0) if ppr0 > 0 else np.nan

    # Theil index (individual fairness, benefit-based)
    benefit = y_pred.astype(float)
    mu = benefit.mean()
    if mu > 0:
        theil = np.mean(
            np.where(benefit > 0,
                     (benefit / mu) * np.log(benefit / mu + 1e-9),
                     0)
        )
    else:
        theil = np.nan

    return {
        "Statistical Parity Diff": round(stat_parity_diff, 4),
        "Equal Opportunity Diff":   round(equal_opp_diff, 4),
        "Avg Abs Odds Diff":        round(avg_abs_odds, 4),
        "Disparate Impact":         round(disp_impact, 4),
        "Theil Index":              round(theil, 4),
    }


def compute_performance_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        try:
            metrics["ROC-AUC"] = round(roc_auc_score(y_true, y_prob), 4)
        except Exception:
            metrics["ROC-AUC"] = np.nan
    else:
        metrics["ROC-AUC"] = np.nan
    return metrics


# ════════════════════════════════════════════════════════════════════════════
# 3. MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION LOOP
# ════════════════════════════════════════════════════════════════════════════

def evaluate_all_models(X, y, sensitive, n_splits=5):
    models  = build_models()
    scaler  = StandardScaler()
    results = {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for name, clf in models.items():
        print(f"\n{'─'*55}")
        print(f"  Training  ➜  {name}")
        print(f"{'─'*55}")

        fold_perf     = []
        fold_fairness = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            s_te       = sensitive.iloc[test_idx].reset_index(drop=True)

            # Scale
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            # Fit
            clf.fit(X_tr_sc, y_tr)

            # Predict
            y_pred = clf.predict(X_te_sc)
            y_prob = clf.predict_proba(X_te_sc)[:, 1] if hasattr(clf, "predict_proba") else None

            y_te_arr   = np.array(y_te)
            y_pred_arr = np.array(y_pred)

            perf    = compute_performance_metrics(y_te_arr, y_pred_arr, y_prob)
            fairness = compute_fairness_metrics(
                y_te_arr, y_pred_arr, s_te
            )

            fold_perf.append(perf)
            fold_fairness.append(fairness)
            print(f"  Fold {fold}  Acc={perf['Accuracy']:.4f}  "
                  f"F1={perf['F1 Score']:.4f}  "
                  f"StatParity={fairness['Statistical Parity Diff']:.4f}")

        # Average across folds
        avg_perf = {k: round(np.nanmean([f[k] for f in fold_perf]), 4) for k in fold_perf[0]}
        avg_fair = {k: round(np.nanmean([f[k] for f in fold_fairness]), 4) for k in fold_fairness[0]}
        results[name] = {**avg_perf, **avg_fair}
        print(f"\n  ✔ {name}  →  Avg Acc={avg_perf['Accuracy']}  "
              f"F1={avg_perf['F1 Score']}  AUC={avg_perf['ROC-AUC']}")

    return pd.DataFrame(results).T


# ════════════════════════════════════════════════════════════════════════════
# 5. PRINT RESULTS TABLE
# ════════════════════════════════════════════════════════════════════════════

def print_results(df):
    perf_cols    = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    fairness_cols = [
        "Statistical Parity Diff", "Equal Opportunity Diff",
        "Avg Abs Odds Diff", "Disparate Impact", "Theil Index"
    ]

    print("\n" + "="*110)
    print("PERFORMANCE METRICS (5-fold CV averages)")
    print("="*110)
    print(df[perf_cols].to_string())

    print("\n" + "="*110)
    print("FAIRNESS METRICS (5-fold CV averages)")
    print("="*110)
    print(df[fairness_cols].to_string())

    print("\n" + "="*110)
    print("VERDICT")
    print("="*110)
    print(f"  Best Accuracy          : {df['Accuracy'].idxmax()}")
    print(f"  Best F1                : {df['F1 Score'].idxmax()}")
    print(f"  Best ROC-AUC           : {df['ROC-AUC'].idxmax()}")
    print(f"  Fairest (Stat Par.)    : {df['Statistical Parity Diff'].abs().idxmin()}")
    print(f"  Fairest (Equal Opp.)   : {df['Equal Opportunity Diff'].abs().idxmin()}")
    print(f"  Lowest Theil Index     : {df['Theil Index'].abs().idxmin()}")
    print("="*110)


# ════════════════════════════════════════════════════════════════════════════
# 6. CHARTS
# ════════════════════════════════════════════════════════════════════════════

COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3"
]

def plot_all(df):
    models = df.index.tolist()
    colors = COLORS[:len(models)]

    perf_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    fair_cols = [
        "Statistical Parity Diff", "Equal Opportunity Diff",
        "Avg Abs Odds Diff", "Disparate Impact", "Theil Index"
    ]

    # ── 6a. Performance bar chart ────────────────────────────────────────
    fig, axes = plt.subplots(1, len(perf_cols), figsize=(18, 5))
    fig.suptitle("Model Performance Comparison (5-Fold CV)", fontsize=14, fontweight="bold")
    for ax, col in zip(axes, perf_cols):
        vals = df[col].fillna(0)
        bars = ax.bar(range(len(models)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(col, fontsize=10)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/performance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {OUT_DIR}/performance_comparison.png")

    # ── 6b. Fairness bar chart ───────────────────────────────────────────
    fig, axes = plt.subplots(1, len(fair_cols), figsize=(22, 5))
    fig.suptitle("Model Fairness Comparison (5-Fold CV)", fontsize=14, fontweight="bold")
    for ax, col in zip(axes, fair_cols):
        vals = df[col].fillna(0)
        bars = ax.bar(range(len(models)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(col, fontsize=9)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (0.002 if v >= 0 else -0.008),
                        f"{v:.3f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fairness_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {OUT_DIR}/fairness_comparison.png")

    # ── 6c. Radar / spider chart ─────────────────────────────────────────
    radar_metrics = ["Accuracy", "F1 Score", "ROC-AUC"]
    # Normalise fairness cols to 0-1 for radar (lower abs = better)
    df_radar = df[radar_metrics].copy()
    for col in ["Statistical Parity Diff", "Equal Opportunity Diff"]:
        max_abs = df[col].abs().max()
        df_radar[col + " (inv)"] = 1 - (df[col].abs() / (max_abs + 1e-9))
    radar_labels = radar_metrics + [
        "Stat Parity (inv)", "Equal Opp (inv)"
    ]
    N = len(radar_labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=9)

    for i, model in enumerate(models):
        vals = df_radar.loc[model].values.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=colors[i], linewidth=1.8, label=model)
        ax.fill(angles, vals, color=colors[i], alpha=0.08)

    ax.set_ylim(0, 1)
    ax.set_title("Performance + Fairness Radar", fontsize=12, fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {OUT_DIR}/radar_chart.png")

    # ── 6d. Accuracy vs Fairness scatter ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, model in enumerate(models):
        ax.scatter(
            df.loc[model, "Statistical Parity Diff"].abs(),
            df.loc[model, "Accuracy"],
            color=colors[i], s=120, zorder=3, label=model
        )
        ax.annotate(model, (
            df.loc[model, "Statistical Parity Diff"].abs(),
            df.loc[model, "Accuracy"]
        ), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("|Statistical Parity Difference|  (lower = fairer)", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("Accuracy vs Fairness Trade-off", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="lower right")
    # Ideal corner annotation
    ax.annotate("← Fairer & more accurate", xy=(0.02, 0.02),
                xycoords="axes fraction", fontsize=8, color="gray",
                fontstyle="italic")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/accuracy_vs_fairness_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {OUT_DIR}/accuracy_vs_fairness_scatter.png")


# ════════════════════════════════════════════════════════════════════════════
# 7. SAVE RESULTS
# ════════════════════════════════════════════════════════════════════════════

def save_results(df):
    csv_path = f"{OUT_DIR}/multi_model_comparison.csv"
    df.to_csv(csv_path)
    print(f"\n  CSV saved → {csv_path}")

    json_path = f"{OUT_DIR}/multi_model_comparison.json"
    df.round(4).to_json(json_path, orient="index", indent=2)
    print(f"  JSON saved → {json_path}")


# ════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*65)
    print("  MULTI-MODEL FAIRNESS & PERFORMANCE COMPARISON")
    print("="*65)

    X, y, sensitive = load_data()
    print(f"\n  Dataset: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"  Label distribution: {dict(y.value_counts())}")
    print(f"  Sensitive attr distribution: {dict(sensitive.value_counts())}")

    results_df = evaluate_all_models(X, y, sensitive, n_splits=5)

    print_results(results_df)
    save_results(results_df)

    print("\n  Generating charts...")
    plot_all(results_df)

    print("\n" + "="*65)
    print("  All outputs saved to: evaluation_results/")
    print("="*65)