"""
preprocess.py
-------------
Data cleaning and preprocessing functions for the
Credit Default Prediction project.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    df = pd.read_csv(filepath, index_col=0)
    print(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"Duplicates removed: {removed}")
    return df


def remove_invalid_age(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where age = 0 (biologically impossible)."""
    before = len(df)
    df = df[df['age'] > 0]
    removed = before - len(df)
    print(f"Invalid age rows removed: {removed}")
    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap extreme outliers:
    - RevolvingUtilizationOfUnsecuredLines: clip at 1
    - DebtRatio: clip at 99th percentile
    """
    df = df.copy()

    # Cap revolving utilization at 1
    df['RevolvingUtilizationOfUnsecuredLines'] = \
        df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=1)

    # Cap DebtRatio at 99th percentile
    debt_99 = df['DebtRatio'].quantile(0.99)
    df['DebtRatio'] = df['DebtRatio'].clip(upper=debt_99)

    print(f"Outliers capped. DebtRatio 99th pct: {debt_99:.2f}")
    return df


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p transformation to MonthlyIncome."""
    df = df.copy()
    df['MonthlyIncome'] = np.log1p(df['MonthlyIncome'])
    print("Log transformation applied to MonthlyIncome.")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using median strategy:
    - MonthlyIncome
    - NumberOfDependents
    """
    df = df.copy()
    cols = ['MonthlyIncome', 'NumberOfDependents']

    imputer = SimpleImputer(strategy='median')
    df[cols] = imputer.fit_transform(df[cols])

    print(f"Missing values imputed for: {cols}")
    return df, imputer


def split_and_scale(
    df: pd.DataFrame,
    target: str = 'SeriousDlqin2yrs',
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split data into train/test sets and apply StandardScaler.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def run_preprocessing(
    input_path: str,
    output_path: str
) -> None:
    """
    Full preprocessing pipeline.
    Saves processed files to output_path.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    df = remove_duplicates(df)
    df = remove_invalid_age(df)
    df = cap_outliers(df)
    df = log_transform(df)
    df, imputer = impute_missing(df)

    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    X_train.to_csv(output_path / "X_train.csv", index=False)
    X_test.to_csv(output_path / "X_test.csv", index=False)
    y_train.to_csv(output_path / "y_train.csv", index=False)
    y_test.to_csv(output_path / "y_test.csv", index=False)

    import joblib
    joblib.dump(scaler,  output_path / "scaler.pkl")
    joblib.dump(imputer, output_path / "imputer.pkl")

    print(f"\nPreprocessing complete. Files saved to {output_path}")


if __name__ == "__main__":
    run_preprocessing(
        input_path="data/cs-training.csv",
        output_path="data/processed"
    )