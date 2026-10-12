"""Train the telco churn model and save a single sklearn Pipeline.

Usage:
    python train.py                 # picks the best model by cross-validated ROC-AUC
    python train.py --model rf      # force a model: lr | rf | rf_balanced

Outputs (in models/):
    churn_pipeline.joblib   preprocessing + model in one object (what app.py loads)
    metrics.json            every number quoted in the README
    drivers.png             top features driving predictions
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
PLOT_PATH = ROOT / "models" / "drivers.png"

SEED = 42
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = CATEGORICAL + NUMERIC


def load_data():
    df = pd.read_csv(DATA_PATH)
    # 11 rows have a blank TotalCharges (customers with tenure 0, never billed): drop them.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    y = (df["Churn"] == "Yes").astype(int)
    return df[FEATURES], y, n_before - len(df)


def make_pipeline(classifier):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    return Pipeline([("prep", pre), ("clf", classifier)])


def candidates():
    return {
        "lr": make_pipeline(LogisticRegression(max_iter=2000)),
        "rf": make_pipeline(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)),
        "rf_balanced": make_pipeline(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=SEED, n_jobs=-1)),
    }


def top_decile_capture(y_true, proba, frac=0.2):
    """Share of all churners found in the riskiest `frac` of customers."""
    k = int(len(y_true) * frac)
    order = np.argsort(-proba)[:k]
    return float(np.asarray(y_true)[order].sum() / np.asarray(y_true).sum())


def evaluate(pipe, X_te, y_te):
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    return {
        "accuracy": round(accuracy_score(y_te, pred), 3),
        "precision": round(precision_score(y_te, pred), 3),
        "recall": round(recall_score(y_te, pred), 3),
        "f1": round(f1_score(y_te, pred), 3),
        "roc_auc": round(roc_auc_score(y_te, proba), 3),
        "top20pct_captures_churners": round(top_decile_capture(y_te, proba), 3),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def feature_drivers(pipe, top=12):
    names = pipe.named_steps["prep"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        values, kind = clf.feature_importances_, "importance"
    else:
        values, kind = clf.coef_[0], "coefficient (log-odds, + raises churn risk)"
    names = [n.split("__", 1)[1] for n in names]
    order = np.argsort(-np.abs(values))[:top]
    return kind, [(names[i], float(values[i])) for i in order]


def save_plot(kind, drivers):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [d[0] for d in drivers][::-1]
    vals = [d[1] for d in drivers][::-1]
    colors = ["#B42318" if v > 0 else "#0F766E" for v in vals]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, vals, color=colors)
    ax.set_title("Top drivers of churn prediction")
    ax.set_xlabel(kind)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lr", "rf", "rf_balanced"], default=None)
    args = parser.parse_args()

    X, y, dropped = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, pipe in candidates().items():
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        pipe.fit(X_tr, y_tr)
        results[name] = {"cv_roc_auc": round(float(cv_auc), 3), **evaluate(pipe, X_te, y_te)}
        fitted[name] = pipe
        print(f"{name:12s} cvAUC={cv_auc:.3f} test={results[name]}")

    chosen = args.model or max(results, key=lambda k: results[k]["cv_roc_auc"])
    print("\nFinal model:", chosen)

    # Refit the chosen model on ALL data for deployment; metrics above come from the held-out split.
    final = candidates()[chosen].fit(X, y)
    kind, drivers = feature_drivers(final)
    save_plot(kind, drivers)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps({
        "dataset": {"rows_used": int(len(X)), "rows_dropped_blank_total_charges": dropped,
                    "churn_rate": round(float(y.mean()), 4),
                    "majority_class_accuracy": round(float(1 - y_te.mean()), 3)},
        "split": "80/20 stratified, random_state=42; cross-validation is 5-fold on the 80% train part",
        "candidates": results,
        "chosen_model": chosen,
        "top_drivers": {"kind": kind, "features": drivers},
        "sklearn_version": sklearn.__version__,
    }, indent=2))
    print("Saved", MODEL_PATH, "and", METRICS_PATH)


if __name__ == "__main__":
    main()
