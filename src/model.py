"""
model.py
--------
Model training functions for the
Credit Default Prediction project.

Models:
    1. Logistic Regression  — baseline, class_weight='balanced'
    2. Random Forest        — RandomizedSearchCV (n_iter=30, 5-fold CV)
    3. XGBoost              — RandomizedSearchCV + early stopping (val set)
    4. XGBoost + SMOTE      — best XGBoost params, SMOTE oversampling
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score


# ── Helper ────────────────────────────────────────────────────

def get_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Calculate scale_pos_weight for XGBoost.
    = negative cases / positive cases
    Equivalent to class_weight='balanced' for tree boosters.
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"scale_pos_weight: {spw:.2f} (neg={neg:,}, pos={pos:,})")
    return spw


# ── Model 1: Logistic Regression (baseline) ───────────────────

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> LogisticRegression:
    """
    Train Logistic Regression as baseline model.
    class_weight='balanced' handles class imbalance.
    No hyperparameter tuning — serves as reference point.
    """
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Logistic Regression trained.")
    return model


# ── Model 2: Random Forest + RandomizedSearchCV ───────────────

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    cv_folds: int = 5
) -> RandomForestClassifier:
    """
    Train Random Forest with RandomizedSearchCV hyperparameter tuning.

    CV is performed entirely within the training set.
    Val and test sets are not involved at this stage.

    Best params found: n_estimators=300, min_samples_split=10,
                       min_samples_leaf=4, max_depth=10
    Best CV AUC-ROC: 0.8571
    """
    param_dist = {
        'n_estimators':      [100, 200, 300],
        'max_depth':         [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [1, 2, 4],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    print(f"Running RandomizedSearchCV for Random Forest (n_iter={n_iter})...")
    search.fit(X_train, y_train)

    print(f"Best parameters : {search.best_params_}")
    print(f"Best CV AUC-ROC : {search.best_score_:.4f}")

    return search.best_estimator_


# ── Model 3: XGBoost + RandomizedSearchCV + Early Stopping ────

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_iter: int = 30,
    cv_folds: int = 5,
    early_stopping_rounds: int = 50
) -> tuple[XGBClassifier, dict]:
    """
    Two-stage XGBoost training:

    Stage 1 — RandomizedSearchCV (5-fold CV within train set)
        Finds best hyperparameters.

    Stage 2 — Early stopping (val set)
        Retrains with best params, monitors val AUC each round.
        Stops when val AUC does not improve for 50 consecutive rounds.
        Val set is used ONLY here — not for tuning or final evaluation.

    Best params found:
        subsample=0.9, reg_lambda=1, reg_alpha=0.1,
        max_depth=3, learning_rate=0.05, colsample_bytree=0.9
    Best iteration (early stopping): 305
    Best validation AUC-ROC: 0.8659

    Returns:
        model: trained XGBClassifier
        best_params: dict of best hyperparameters (used by SMOTE variant)
    """
    spw = get_scale_pos_weight(y_train)

    param_dist = {
        'max_depth':        [3, 4, 5, 6],
        'learning_rate':    [0.01, 0.05, 0.1],
        'subsample':        [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha':        [0, 0.1, 0.5],
        'reg_lambda':       [1, 1.5, 2],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Stage 1: RandomizedSearchCV
    search = RandomizedSearchCV(
        XGBClassifier(
            n_estimators=300,
            scale_pos_weight=spw,
            random_state=42,
            eval_metric='auc',
            verbosity=0
        ),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    print(f"Running RandomizedSearchCV for XGBoost (n_iter={n_iter})...")
    search.fit(X_train, y_train)
    best_params = search.best_params_

    print(f"Best parameters : {best_params}")
    print(f"Best CV AUC-ROC : {search.best_score_:.4f}")

    # Stage 2: Retrain with best params + early stopping on val set
    print("Retraining with best parameters + early stopping...")
    model = XGBClassifier(
        **best_params,
        n_estimators=1000,               # high ceiling — early stopping decides
        scale_pos_weight=spw,
        early_stopping_rounds=early_stopping_rounds,
        random_state=42,
        eval_metric='auc',
        verbosity=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],       # val set monitors training — NOT test set
        verbose=False
    )

    print(f"Best iteration (early stopping) : {model.best_iteration}")
    print(f"Best validation AUC-ROC         : {model.best_score:.4f}")

    return model, best_params


# ── Model 4: XGBoost + SMOTE ──────────────────────────────────

def train_xgboost_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
    best_iteration: int
) -> XGBClassifier:
    """
    Train XGBoost with SMOTE oversampling using best hyperparameters
    found during XGBoost tuning.

    SMOTE is applied to the training set ONLY.
    Val and test sets are never oversampled — doing so would
    introduce data leakage and inflate performance metrics.

    Note: scale_pos_weight is not needed here as SMOTE
    balances the class distribution directly.

    Note on performance: SMOTE underperforms scale_pos_weight on
    this dataset because several features are discrete count variables
    (e.g. NumberOfTimes90DaysLate). Interpolating between integer
    counts generates unrealistic synthetic samples.
    """
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    print(f"After SMOTE : {pd.Series(y_res).value_counts().to_dict()}")

    model = XGBClassifier(
        **best_params,
        n_estimators=best_iteration,     # use optimal iteration from early stopping
        random_state=42,
        eval_metric='auc',
        verbosity=0
        # no scale_pos_weight — SMOTE handles class balance
    )

    model.fit(X_res, y_res)
    print("XGBoost + SMOTE trained.")
    return model


# ── Evaluation helper ─────────────────────────────────────────

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str = "Model"
) -> dict:
    """Compute and print AUC-ROC on test set."""
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"{name:<25} AUC-ROC: {auc:.4f}")
    return {"model": name, "auc_roc": auc}


# ── Full pipeline ─────────────────────────────────────────────

def run_training(
    data_path: str,
    model_path: str
) -> None:
    """
    Full training pipeline.

    Loads processed data (train / val / test),
    trains all four models, evaluates on test set,
    and saves models to model_path.
    """
    data_path  = Path(data_path)
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train = pd.read_csv(data_path / "X_train.csv")
    X_val   = pd.read_csv(data_path / "X_val.csv")
    X_test  = pd.read_csv(data_path / "X_test.csv")
    y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
    y_val   = pd.read_csv(data_path / "y_val.csv").squeeze()
    y_test  = pd.read_csv(data_path / "y_test.csv").squeeze()

    print(f"Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}\n")

    # Train models
    lr     = train_logistic_regression(X_train, y_train)
    rf     = train_random_forest(X_train, y_train)
    xgb, best_params = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_sm = train_xgboost_smote(X_train, y_train, best_params, xgb.best_iteration)

    # Evaluate on test set
    print("\nTest set evaluation:")
    results = {
        "logistic_regression": lr,
        "random_forest":       rf,
        "xgboost":             xgb,
        "xgboost_smote":       xgb_sm,
    }
    for name, model in results.items():
        evaluate_model(model, X_test, y_test, name)

    # Save models
    for name, model in results.items():
        joblib.dump(model, model_path / f"{name}.pkl")

    print(f"\nAll models saved to {model_path}")


if __name__ == "__main__":
    run_training(
        data_path="data/processed",
        model_path="outputs/models"
    )
