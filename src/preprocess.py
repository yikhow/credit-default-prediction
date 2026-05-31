"""
preprocess.py
-------------
Data cleaning and preprocessing functions for the
Credit Default Prediction project.

Pipeline:
    load → deduplicate → remove invalid age → cap outliers
    → log transform → impute → 3-way split (60/20/20) → scale → save
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ── Individual steps ──────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    df = pd.read_csv(filepath, index_col=0)
    print(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates(keep='first')
    print(f"Duplicates removed: {before - len(df)} ({len(df):,} rows remaining)")
    return df


def remove_invalid_age(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where age = 0 (biologically impossible)."""
    before = len(df)
    df = df[df['age'] > 0]
    print(f"Invalid age rows removed: {before - len(df)} ({len(df):,} rows remaining)")
    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap extreme outliers:
    - RevolvingUtilizationOfUnsecuredLines: clip at 1
      (utilization rate cannot exceed 100%)
    - DebtRatio: clip at 99th percentile
      (extreme values inconsistent with real financial behaviour)
    """
    df = df.copy()

    # Cap revolving utilization at 1
    before = (df['RevolvingUtilizationOfUnsecuredLines'] > 1).sum()
    df['RevolvingUtilizationOfUnsecuredLines'] = \
        df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=1)
    print(f"RevolvingUtilization capped at 1: {before} values modified")

    # Cap DebtRatio at 99th percentile
    debt_99 = df['DebtRatio'].quantile(0.99)
    before = (df['DebtRatio'] > debt_99).sum()
    df['DebtRatio'] = df['DebtRatio'].clip(upper=debt_99)
    print(f"DebtRatio capped at 99th pct ({debt_99:.2f}): {before} values modified")

    return df


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p transformation to MonthlyIncome.
    log1p(x) = log(1 + x) — safely handles zero values.
    """
    df = df.copy()
    df['MonthlyIncome'] = np.log1p(df['MonthlyIncome'])
    print("Log1p transformation applied to MonthlyIncome.")
    return df


def impute_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, SimpleImputer]:
    """
    Impute missing values using median strategy:
    - MonthlyIncome  (19.6% missing after dedup/age removal)
    - NumberOfDependents (2.6% missing)

    Median is used over mean as both features are right-skewed
    with extreme values — median is more robust to outliers.

    Returns:
        df: imputed DataFrame
        imputer: fitted SimpleImputer (saved for inference)
    """
    df = df.copy()
    cols = ['MonthlyIncome', 'NumberOfDependents']

    print(f"Missing before imputation: "
          f"MonthlyIncome={df['MonthlyIncome'].isnull().sum()}, "
          f"NumberOfDependents={df['NumberOfDependents'].isnull().sum()}")

    imputer = SimpleImputer(strategy='median')
    df[cols] = imputer.fit_transform(df[cols])

    print(f"Missing after imputation: "
          f"MonthlyIncome={df['MonthlyIncome'].isnull().sum()}, "
          f"NumberOfDependents={df['NumberOfDependents'].isnull().sum()}")

    return df, imputer


def split_and_scale(
    df: pd.DataFrame,
    target: str = 'SeriousDlqin2yrs',
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Three-way stratified split (60% train / 20% val / 20% test)
    followed by StandardScaler fitted on train only.

    Val set is used for XGBoost early stopping.
    Test set is held out for final evaluation only.

    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test, scaler
    """
    X = df.drop(columns=[target])
    y = df[target]

    # Step 1: Split off test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Step 2: Split remaining 80% → train (60%) and val (20%)
    # test_size=0.25 of 80% = 20% of total
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.25,
        random_state=random_state,
        stratify=y_temp
    )

    print(f"Train : {X_train.shape[0]:,} rows ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Val   : {X_val.shape[0]:,} rows ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"Test  : {X_test.shape[0]:,} rows ({X_test.shape[0]/len(X)*100:.1f}%)")

    # Scale: fit on train only, transform val and test
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns)

    print("StandardScaler fitted on train, applied to val and test.")
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


# ── Full pipeline ─────────────────────────────────────────────

def run_preprocessing(
    input_path: str,
    output_path: str,
    artefact_path: str = "outputs/artefacts"
) -> None:
    """
    Full preprocessing pipeline.

    Saves:
    - Processed datasets to output_path/
      (X_train, X_val, X_test, y_train, y_val, y_test)
    - Fitted artefacts to artefact_path/
      (scaler.pkl, imputer.pkl)
    """
    output_path   = Path(output_path)
    artefact_path = Path(artefact_path)
    output_path.mkdir(parents=True, exist_ok=True)
    artefact_path.mkdir(parents=True, exist_ok=True)

    # Pipeline
    df = load_data(input_path)
    df = remove_duplicates(df)
    df = remove_invalid_age(df)
    df = cap_outliers(df)
    df = log_transform(df)
    df, imputer = impute_missing(df)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(df)

    # Save datasets
    X_train.to_csv(output_path / "X_train.csv", index=False)
    X_val.to_csv(output_path   / "X_val.csv",   index=False)
    X_test.to_csv(output_path  / "X_test.csv",  index=False)
    y_train.to_csv(output_path / "y_train.csv", index=False)
    y_val.to_csv(output_path   / "y_val.csv",   index=False)
    y_test.to_csv(output_path  / "y_test.csv",  index=False)

    # Save artefacts
    joblib.dump(scaler,  artefact_path / "scaler.pkl")
    joblib.dump(imputer, artefact_path / "imputer.pkl")

    print(f"\nPreprocessing complete.")
    print(f"  Datasets  → {output_path}")
    print(f"  Artefacts → {artefact_path}")


if __name__ == "__main__":
    run_preprocessing(
        input_path="data/cs-training.csv",
        output_path="data/processed",
        artefact_path="outputs/artefacts"
    )
