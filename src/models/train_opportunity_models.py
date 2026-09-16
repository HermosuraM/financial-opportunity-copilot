"""Train one binary opportunity-detection model per label.

Each model predicts a heuristic opportunity label (see
src/features/heuristic_labels.py) from a *noised* snapshot of the feature
table -- balances, spend, and holdings are perturbed and partially dropped to
mimic a real batch-scoring pipeline reading a T-1 data feed, while the label
itself is computed from the clean ground truth. So the task is not "memorize
the rule" but "stay well-calibrated when the inputs are a little stale or
occasionally missing." For each label we fit a logistic-regression baseline
and a HistGradientBoostingClassifier (which handles NaNs natively) and keep
whichever has the higher PR-AUC (the imbalance-robust metric here).
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, RNG_SEED
from src.features.heuristic_labels import OPPORTUNITY_LABELS, add_heuristic_labels, simulate_snapshot_noise

CATEGORICAL_COLS = ["job", "marital", "education", "contact_type", "prior_campaign_outcome"]
ID_COL = "customer_id"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "feature_table.csv")
    df = add_heuristic_labels(df)
    for col in ["has_401k", "has_credit_card", "has_brokerage"]:
        df[col] = df[col].astype(int)
    rng = np.random.default_rng(RNG_SEED)
    noisy = simulate_snapshot_noise(df, rng)
    noisy[OPPORTUNITY_LABELS] = df[OPPORTUNITY_LABELS]
    return noisy


def feature_columns_for(df_encoded: pd.DataFrame, label: str) -> list[str]:
    excluded = set(OPPORTUNITY_LABELS) | {ID_COL}
    return [c for c in df_encoded.columns if c not in excluded]


def train_one(df_encoded: pd.DataFrame, label: str) -> dict:
    cols = feature_columns_for(df_encoded, label)
    X = df_encoded[cols]
    y = df_encoded[label]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RNG_SEED, stratify=y
    )

    logreg = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    logreg.fit(X_train, y_train)

    hgb = HistGradientBoostingClassifier(random_state=RNG_SEED, class_weight="balanced", max_depth=6)
    hgb.fit(X_train, y_train)

    candidates = {"logistic_regression": logreg, "hist_gradient_boosting": hgb}
    scored = {}
    for name, model in candidates.items():
        proba = model.predict_proba(X_test)[:, 1]
        scored[name] = {
            "model": model,
            "proba": proba,
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "brier": brier_score_loss(y_test, proba),
        }

    best_name = max(scored, key=lambda n: scored[n]["pr_auc"])
    best = scored[best_name]

    joblib.dump({"model": best["model"], "feature_columns": cols, "model_name": best_name}, MODELS_DIR / f"{label}.joblib")

    plot_diagnostics(label, best_name, best["model"], X_test, y_test, best["proba"])

    return {
        "label": label,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "base_rate": float(y.mean()),
        "best_model": best_name,
        "metrics": {
            name: {k: round(v, 4) for k, v in s.items() if k != "model" and k != "proba"}
            for name, s in scored.items()
        },
    }


def plot_diagnostics(label, model_name, model, X_test, y_test, proba) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[0])
    axes[0].set_title(f"{label}\nROC ({model_name})")

    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[1])
    axes[1].set_title("Precision-Recall")

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    axes[2].plot(mean_pred, frac_pos, marker="o", label="model")
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Observed frequency")
    axes[2].set_title("Calibration")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{label}_diagnostics.png", dpi=130)
    plt.close(fig)

    if hasattr(model, "predict_proba"):
        result = permutation_importance(
            model, X_test, y_test, scoring="average_precision", n_repeats=8, random_state=RNG_SEED, n_jobs=-1
        )
        order = np.argsort(result.importances_mean)[-10:]
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        ax2.barh(np.array(X_test.columns)[order], result.importances_mean[order])
        ax2.set_xlabel("Permutation importance (PR-AUC drop)")
        ax2.set_title(f"{label}: top features")
        fig2.tight_layout()
        fig2.savefig(FIGURES_DIR / f"{label}_importance.png", dpi=130)
        plt.close(fig2)


def main() -> None:
    df = load_dataset()
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS)

    results = [train_one(df_encoded, label) for label in OPPORTUNITY_LABELS]

    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"{'label':<24}{'best_model':<24}{'roc_auc':>9}{'pr_auc':>9}{'base_rate':>11}")
    for r in results:
        m = r["metrics"][r["best_model"]]
        print(f"{r['label']:<24}{r['best_model']:<24}{m['roc_auc']:>9.3f}{m['pr_auc']:>9.3f}{r['base_rate']:>11.3f}")
    print(f"\nSaved models -> {MODELS_DIR}")
    print(f"Saved figures -> {FIGURES_DIR}")
    print(f"Saved metrics -> {REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
