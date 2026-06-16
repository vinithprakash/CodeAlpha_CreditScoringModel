"""
CodeAlpha Internship - Task 1: Credit Scoring Model
train.py — Feature engineering, model training, comparison & saving

Trains three classifiers (Logistic Regression, Decision Tree, Random Forest),
evaluates each with Accuracy / Precision / Recall / F1 / ROC-AUC, and saves:
  - model/credit_model.pkl     -> deployed pipeline (Logistic Regression)
  - model/feature_info.json    -> feature metadata for the Streamlit app
  - model/metrics.json         -> comparison metrics for all 3 models
  - utils/*.png                -> evaluation plots
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

RANDOM_STATE = 42
DATA_PATH = "dataset/credit_data.csv"

NUMERIC_FEATURES = [
    "Age", "Job", "Monthly_Income", "Employment_Years",
    "Credit_amount", "Duration", "Existing_Loans", "Dependents",
    "Debt_to_Income", "Credit_per_Month",
]
CATEGORICAL_FEATURES = [
    "Sex", "Housing", "Saving_accounts", "Checking_account",
    "Purpose", "Payment_History",
]
TARGET = "Risk"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = (df[TARGET] == "good").astype(int)  # 1 = good credit, 0 = bad credit
    return df, X, y


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def evaluate_model(name, pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Bad", "Good"]))

    return metrics, y_pred, y_proba


def plot_model_comparison(results):
    os.makedirs("utils", exist_ok=True)
    metrics_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    models = list(results.keys())

    x = np.arange(len(metrics_names))
    width = 0.25

    plt.figure(figsize=(11, 6))
    for i, model in enumerate(models):
        values = [results[model][m] for m in metrics_names]
        plt.bar(x + i * width, values, width, label=model)

    plt.xticks(x + width, [m.replace("_", " ").title() for m in metrics_names])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Comparison — Credit Scoring")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("utils/model_comparison.png", dpi=150)
    plt.close()
    print("Saved: utils/model_comparison.png")


def plot_confusion(y_test, y_pred, name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("utils/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: utils/confusion_matrix.png")


def plot_roc_curves(curves):
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — All Models")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("utils/roc_curve.png", dpi=150)
    plt.close()
    print("Saved: utils/roc_curve.png")


def plot_feature_importance(pipe):
    """Use Logistic Regression coefficients as feature 'importance' / direction."""
    preprocessor = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefs = model.coef_[0]

    order = np.argsort(np.abs(coefs))[-12:]  # top 12
    names = feature_names[order]
    values = coefs[order]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    plt.figure(figsize=(9, 7))
    plt.barh(range(len(values)), values, color=colors)
    plt.yticks(range(len(values)), [n.split("__")[-1] for n in names])
    plt.xlabel("Coefficient (impact on 'Good Credit' probability)")
    plt.title("Top Feature Influences — Logistic Regression")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("utils/feature_importance.png", dpi=150)
    plt.close()
    print("Saved: utils/feature_importance.png")


def save_feature_info(df):
    info = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": {
            col: sorted(df[col].unique().tolist())
            for col in CATEGORICAL_FEATURES
        },
        "numeric_ranges": {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
            }
            for col in NUMERIC_FEATURES
        },
    }
    with open("model/feature_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("Saved: model/feature_info.json")


def main():
    os.makedirs("model", exist_ok=True)
    os.makedirs("utils", exist_ok=True)

    print("Loading dataset …")
    df, X, y = load_data()
    print(f"Total samples: {len(df)} | Good: {(y==1).sum()} | Bad: {(y==0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE),
    }

    results = {}
    roc_data = {}
    pipelines = {}

    for name, clf in candidates.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        pipe.fit(X_train, y_train)
        metrics, y_pred, y_proba = evaluate_model(name, pipe, X_test, y_test)
        results[name] = metrics
        pipelines[name] = pipe

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, metrics["roc_auc"])

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_model_comparison(results)
    plot_roc_curves(roc_data)

    best_pred = pipelines["Logistic Regression"].predict(X_test)
    plot_confusion(y_test, best_pred, "Logistic Regression")
    plot_feature_importance(pipelines["Logistic Regression"])

    # ── Save deployed model (Logistic Regression — most interpretable) ────────
    joblib.dump(pipelines["Logistic Regression"], "model/credit_model.pkl")
    print("\nSaved: model/credit_model.pkl  (deployed model for the app)")

    save_feature_info(df)

    with open("model/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: model/metrics.json")

    print("\n" + "=" * 50)
    print("  TRAINING COMPLETE")
    print("=" * 50)
    for name, m in results.items():
        print(f"  {name:20s} | Acc: {m['accuracy']:.3f} | F1: {m['f1_score']:.3f} | AUC: {m['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
