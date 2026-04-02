"""
app.py
------
Streamlit dashboard for Credit Default Risk Prediction.
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from src.predict import predict, FEATURE_COLS, RISK_THRESHOLDS

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Default Risk Predictor",
    page_icon="💳",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────
st.title("💳 Credit Default Risk Predictor")
st.markdown(
    "Enter borrower information to assess the probability "
    "of loan default using an XGBoost model trained on "
    "150,000 historical borrower records."
)
st.divider()

# ── Input Form ────────────────────────────────────────────────
st.subheader("📋 Borrower Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age", min_value=18, max_value=100, value=40)
    monthly_income = st.number_input(
        "Monthly Income (USD)", min_value=0, value=5000, step=100)
    debt_ratio = st.slider(
        "Debt Ratio", min_value=0.0, max_value=1.0,
        value=0.3, step=0.01)

with col2:
    revolving_util = st.slider(
        "Revolving Utilization", min_value=0.0, max_value=1.0,
        value=0.3, step=0.01)
    open_credit = st.number_input(
        "Open Credit Lines & Loans",
        min_value=0, max_value=60, value=5)
    real_estate = st.number_input(
        "Real Estate Loans", min_value=0, max_value=20, value=0)

with col3:
    late_30_59 = st.number_input(
        "Times 30-59 Days Late", min_value=0, max_value=20, value=0)
    late_60_89 = st.number_input(
        "Times 60-89 Days Late", min_value=0, max_value=20, value=0)
    late_90 = st.number_input(
        "Times 90+ Days Late", min_value=0, max_value=20, value=0)
    dependents = st.number_input(
        "Number of Dependents", min_value=0, max_value=20, value=0)

st.divider()

# ── Prediction ────────────────────────────────────────────────
if st.button("🔍 Assess Default Risk", use_container_width=True):

    borrower_data = {
        'RevolvingUtilizationOfUnsecuredLines':  revolving_util,
        'age':                                    age,
        'NumberOfTime30-59DaysPastDueNotWorse':   late_30_59,
        'DebtRatio':                              debt_ratio,
        'MonthlyIncome':                          monthly_income,
        'NumberOfOpenCreditLinesAndLoans':         open_credit,
        'NumberOfTimes90DaysLate':                 late_90,
        'NumberRealEstateLoansOrLines':            real_estate,
        'NumberOfTime60-89DaysPastDueNotWorse':   late_60_89,
        'NumberOfDependents':                      dependents,
    }

    with st.spinner("Analysing borrower profile..."):
        result = predict(borrower_data)

    prob  = result['default_probability']
    label = result['risk_label']
    factors = result['top_risk_factors']

    # ── Result display ────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}[label]
        st.markdown(f"### Default Probability")
        st.markdown(
            f"<h1 style='color:{color}'>{prob:.1%}</h1>",
            unsafe_allow_html=True
        )

    with col_b:
        st.markdown("### Risk Label")
        st.markdown(
            f"<h1 style='color:{color}'>{label}</h1>",
            unsafe_allow_html=True
        )

    with col_c:
        st.markdown("### Top Risk Factors")
        for factor in factors:
            st.markdown(f"• `{factor}`")

    st.divider()

    # ── Probability gauge ─────────────────────────────────────
    st.subheader("📊 Risk Probability Breakdown")

    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.barh(0, prob, color=color, height=0.4)
    ax.barh(0, 1 - prob, left=prob,
            color='lightgrey', height=0.4)
    ax.axvline(x=0.20, color='orange',
               linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=0.50, color='red',
               linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Default Probability")
    ax.text(0.10, 0.6, 'LOW',    transform=ax.transAxes,
            color='green',  fontsize=9)
    ax.text(0.40, 0.6, 'MEDIUM', transform=ax.transAxes,
            color='orange', fontsize=9)
    ax.text(0.75, 0.6, 'HIGH',   transform=ax.transAxes,
            color='red',    fontsize=9)
    st.pyplot(fig)

    # ── Recommendation ────────────────────────────────────────
    st.subheader("📋 Recommendation")

    if label == "LOW":
        st.success(
            "✅ **Approve**: Low default risk. "
            "Standard loan terms recommended."
        )
    elif label == "MEDIUM":
        st.warning(
            "⚠️ **Review**: Moderate default risk. "
            "Recommend manual underwriting review before approval. "
            "Consider requesting additional documentation."
        )
    else:
        st.error(
            "❌ **Decline or Enhanced Review**: High default risk. "
            "Recommend rejection or significantly reduced credit limit "
            "with enhanced monitoring conditions."
        )

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: XGBoost | Trained on: Give Me Some Credit (Kaggle) | "
    "AUC-ROC: 0.8664 | Recall: 0.7897"
)