"""
predict.py
----------
Prediction functions for the
Credit Default Prediction project.
Loads trained model and preprocessors to score new borrower data.
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

RISK_THRESHOLDS = {
    'LOW':    (0.00, 0.20),
    'MEDIUM': (0.20, 0.50),
    'HIGH':   (0.50, 1.00),
}


def load_artifacts(model_path: str, data_path: str) -> tuple:
    """Load model, scaler, and imputer from disk."""
    model_path = Path(model_path)
    data_path  = Path(data_path)

    model   = joblib.load(model_path / "xgboost.pkl")
    scaler  = joblib.load(data_path  / "scaler.pkl")
    imputer = joblib.load(data_path  / "imputer.pkl")

    return model, scaler, imputer


def preprocess_input(
    data: dict,
    imputer,
    scaler
) -> pd.DataFrame:
    """
    Preprocess a single borrower's input data.

    Args:
        data: dict of feature values
        imputer: fitted SimpleImputer
        scaler: fitted StandardScaler

    Returns:
        Scaled feature DataFrame ready for prediction
    """
    df = pd.DataFrame([data], columns=FEATURE_COLS)

    # Cap outliers
    df['RevolvingUtilizationOfUnsecuredLines'] = \
        df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=1)
    debt_cap = 3415  # 99th percentile from training data
    df['DebtRatio'] = df['DebtRatio'].clip(upper=debt_cap)

    # Log transform
    df['MonthlyIncome'] = np.log1p(df['MonthlyIncome'])

    # Impute missing values
    impute_cols = ['MonthlyIncome', 'NumberOfDependents']
    df[impute_cols] = imputer.transform(df[impute_cols])

    # Scale
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=FEATURE_COLS
    )

    return df_scaled


def get_risk_label(probability: float) -> str:
    """Convert default probability to risk label."""
    for label, (low, high) in RISK_THRESHOLDS.items():
        if low <= probability < high:
            return label
    return 'HIGH'


def predict(
    data: dict,
    model_path: str = "outputs/models",
    data_path: str  = "data/processed"
) -> dict:
    """
    Predict default risk for a single borrower.

    Args:
        data: dict of borrower feature values
        model_path: path to saved models
        data_path: path to saved preprocessors

    Returns:
        dict with probability, risk label, and top risk factors
    """
    model, scaler, imputer = load_artifacts(model_path, data_path)

    X = preprocess_input(data, imputer, scaler)
    probability = model.predict_proba(X)[0][1]
    risk_label  = get_risk_label(probability)

    # Identify top risk factors
    importance = dict(zip(
        FEATURE_COLS, model.feature_importances_
    ))
    top_factors = sorted(
        importance.items(), key=lambda x: x[1], reverse=True
    )[:3]

    return {
        "default_probability": round(float(probability), 4),
        "risk_label":          risk_label,
        "top_risk_factors":    [f for f, _ in top_factors],
    }


if __name__ == "__main__":
    # Example prediction
    sample_borrower = {
        'RevolvingUtilizationOfUnsecuredLines':   0.85,
        'age':                                     32,
        'NumberOfTime30-59DaysPastDueNotWorse':    2,
        'DebtRatio':                               0.45,
        'MonthlyIncome':                           4500,
        'NumberOfOpenCreditLinesAndLoans':          8,
        'NumberOfTimes90DaysLate':                  1,
        'NumberRealEstateLoansOrLines':             0,
        'NumberOfTime60-89DaysPastDueNotWorse':    0,
        'NumberOfDependents':                       1,
    }

    result = predict(sample_borrower)

    print("\n" + "="*45)
    print("  CREDIT DEFAULT RISK ASSESSMENT")
    print("="*45)
    print(f"  Default Probability : {result['default_probability']:.1%}")
    print(f"  Risk Label          : {result['risk_label']}")
    print(f"  Top Risk Factors    :")
    for factor in result['top_risk_factors']:
        print(f"    • {factor}")
    print("="*45)