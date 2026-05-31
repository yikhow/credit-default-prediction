"""
predict.py
----------
Prediction functions for the
Credit Default Prediction project.

Loads the trained XGBoost model and fitted artefacts (scaler, imputer)
to score a single new borrower and return a risk assessment.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path


FEATURE_COLS = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents'
]

# DebtRatio 99th percentile from training data (computed in preprocessing)
DEBT_RATIO_CAP = 4985.11

RISK_THRESHOLDS = {
    'LOW':    (0.00, 0.20),
    'MEDIUM': (0.20, 0.50),
    'HIGH':   (0.50, 1.00),
}


def load_artifacts(
    model_path: str = "outputs/models",
    artefact_path: str = "outputs/artefacts"
) -> tuple:
    """
    Load trained XGBoost model, fitted scaler, and fitted imputer.

    Scaler and imputer are loaded from artefact_path (not data/processed)
    as they are inference artefacts, not data files.

    Returns:
        model, scaler, imputer
    """
    model   = joblib.load(Path(model_path)   / "xgboost.pkl")
    scaler  = joblib.load(Path(artefact_path) / "scaler.pkl")
    imputer = joblib.load(Path(artefact_path) / "imputer.pkl")
    return model, scaler, imputer


def preprocess_input(
    data: dict,
    imputer,
    scaler
) -> pd.DataFrame:
    """
    Apply the same preprocessing steps as the training pipeline
    to a single borrower's input data.

    Steps (must match 02_preprocessing.ipynb exactly):
    1. Cap RevolvingUtilization at 1
    2. Cap DebtRatio at training 99th percentile (4985.11)
    3. Log1p transform MonthlyIncome
    4. Impute missing values (median, fitted on training data)
    5. Scale (StandardScaler, fitted on training data)

    Args:
        data: dict of raw feature values (may contain None/NaN)
        imputer: fitted SimpleImputer from training
        scaler: fitted StandardScaler from training

    Returns:
        Scaled feature DataFrame ready for model.predict_proba()
    """
    df = pd.DataFrame([data], columns=FEATURE_COLS)

    # 1. Cap outliers
    df['RevolvingUtilizationOfUnsecuredLines'] = \
        df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=1)
    df['DebtRatio'] = df['DebtRatio'].clip(upper=DEBT_RATIO_CAP)

    # 2. Log transform
    df['MonthlyIncome'] = np.log1p(df['MonthlyIncome'])

    # 3. Impute missing values (using median fitted on training data)
    impute_cols = ['MonthlyIncome', 'NumberOfDependents']
    df[impute_cols] = imputer.transform(df[impute_cols])

    # 4. Scale (using scaler fitted on training data)
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=FEATURE_COLS
    )

    return df_scaled


def get_risk_label(probability: float) -> str:
    """
    Convert default probability to risk label.

    Thresholds:
        LOW:    [0.00, 0.20)
        MEDIUM: [0.20, 0.50)
        HIGH:   [0.50, 1.00]
    """
    for label, (low, high) in RISK_THRESHOLDS.items():
        if low <= probability < high:
            return label
    return 'HIGH'


def predict(
    data: dict,
    model_path: str = "outputs/models",
    artefact_path: str = "outputs/artefacts"
) -> dict:
    """
    Predict default risk for a single borrower.

    Args:
        data: dict of raw borrower feature values
        model_path: directory containing trained model pkl files
        artefact_path: directory containing scaler.pkl and imputer.pkl

    Returns:
        dict with:
            - default_probability: float (0 to 1)
            - risk_label: str ('LOW', 'MEDIUM', or 'HIGH')
            - top_risk_factors: list of 3 most important features
    """
    model, scaler, imputer = load_artifacts(model_path, artefact_path)

    X = preprocess_input(data, imputer, scaler)
    probability = model.predict_proba(X)[0][1]
    risk_label  = get_risk_label(probability)

    # Top 3 features by model importance (static — same for all predictions)
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    top_factors = sorted(
        importance.items(), key=lambda x: x[1], reverse=True
    )[:3]

    return {
        "default_probability": round(float(probability), 4),
        "risk_label":          risk_label,
        "top_risk_factors":    [f for f, _ in top_factors],
    }


if __name__ == "__main__":
    sample_borrower = {
        'RevolvingUtilizationOfUnsecuredLines':  0.85,
        'age':                                    32,
        'NumberOfTime30-59DaysPastDueNotWorse':   2,
        'DebtRatio':                              0.45,
        'MonthlyIncome':                          4500,
        'NumberOfOpenCreditLinesAndLoans':         8,
        'NumberOfTimes90DaysLate':                 1,
        'NumberRealEstateLoansOrLines':            0,
        'NumberOfTime60-89DaysPastDueNotWorse':   0,
        'NumberOfDependents':                      1,
    }

    result = predict(sample_borrower)

    print("\n" + "=" * 45)
    print("  CREDIT DEFAULT RISK ASSESSMENT")
    print("=" * 45)
    print(f"  Default Probability : {result['default_probability']:.1%}")
    print(f"  Risk Label          : {result['risk_label']}")
    print(f"  Top Risk Factors    :")
    for factor in result['top_risk_factors']:
        print(f"    • {factor}")
    print("=" * 45)
