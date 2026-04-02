"""
model.py
--------
Model training functions for the
Credit Default Prediction project.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score


def get_scale_pos_weight(y_train: pd.Series) -> float:
    """Calculate scale_pos_weight for XGBoost."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    return neg / pos


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> LogisticRegression:
    """Train Logistic Regression baseline model."""
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Logistic Regression trained.")
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestClassifier:
    """Train Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Random Forest trained.")
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> XGBClassifier:
    """Train XGBoost model."""
    spw = get_scale_pos_weight(y_train)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric='auc',
        verbosity=0
    )
    model.fit(X_train, y_train)
    print(f"XGBoost trained. scale_pos_weight={spw:.2f}")
    return model


def train_xgboost_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> XGBClassifier:
    """Train XGBoost model with SMOTE oversampling."""
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric='auc',
        verbosity=0
    )
    model.fit(X_res, y_res)
    print("XGBoost + SMOTE trained.")
    return model


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str = "Model"
) -> dict:
    """Print AUC-ROC score for a given model."""
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"{name} AUC-ROC: {auc:.4f}")
    return {"model": name, "auc_roc": auc}


def run_training(
    data_path: str,
    model_path: str
) -> None:
    """
    Full training pipeline.
    Loads processed data, trains all models, saves to model_path.
    """
    data_path  = Path(data_path)
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(data_path / "X_train.csv")
    X_test  = pd.read_csv(data_path / "X_test.csv")
    y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
    y_test  = pd.read_csv(data_path / "y_test.csv").squeeze()

    models = {
        "logistic_regression": train_logistic_regression(X_train, y_train),
        "random_forest":       train_random_forest(X_train, y_train),
        "xgboost":             train_xgboost(X_train, y_train),
        "xgboost_smote":       train_xgboost_smote(X_train, y_train),
    }

    print("\nEvaluation on test set:")
    for name, model in models.items():
        evaluate_model(model, X_test, y_test, name)
        joblib.dump(model, model_path / f"{name}.pkl")

    print(f"\nAll models saved to {model_path}")


if __name__ == "__main__":
    run_training(
        data_path="data/processed",
        model_path="outputs/models"
    )