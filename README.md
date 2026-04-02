# Credit Default Prediction

A machine learning project that predicts the probability of loan 
default using historical borrower data. Built to demonstrate 
end-to-end data analysis skills — from raw data to an interactive 
prediction dashboard.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![AUC](https://img.shields.io/badge/AUC--ROC-0.8664-brightgreen)

---

## 📌 Problem Statement

Given a borrower's financial history, predict whether they will 
experience serious delinquency (90+ days late) within 2 years.

**Business context:** In credit risk management, failing to identify 
a defaulter (false negative) is significantly more costly than 
rejecting a creditworthy borrower. This project prioritises Recall 
to minimise missed defaults.

---

## 📊 Dataset

- **Source**: [Give Me Some Credit — Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)
- **Size**: 150,000 borrower records, 10 features
- **Class imbalance**: 6.7% default rate

---

## 🔍 Project Structure
```
credit-default-prediction/
├── data/
│   ├── cs-training.csv          # Raw data (download from Kaggle)
│   └── processed/               # Cleaned data (auto-generated)
├── notebooks/
│   ├── 01_EDA.ipynb             # Exploratory data analysis
│   ├── 02_preprocessing.ipynb  # Data cleaning & preparation
│   ├── 03_modeling.ipynb        # Model training & comparison
│   └── 04_evaluation.ipynb      # Model evaluation & insights
├── src/
│   ├── preprocess.py            # Preprocessing functions
│   ├── model.py                 # Training functions
│   └── predict.py               # Prediction pipeline
├── outputs/
│   └── models/                  # Saved model files
├── app.py                       # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Workflow

### 1. Preprocessing
```bash
python src/preprocess.py
```
Cleans raw data and saves processed files to `data/processed/`.

### 2. Model Training
```bash
python src/model.py
```
Trains all four models and saves to `outputs/models/`.

### 3. Prediction (single borrower)
```bash
python src/predict.py
```
Runs a sample prediction using the trained XGBoost model.

### 4. Launch Dashboard
```bash
streamlit run app.py
```

---

## 📈 Results

| Model | AUC-ROC | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.8377 | 0.1898 | 0.7532 | 0.3032 |
| Random Forest | 0.8379 | 0.5542 | 0.1608 | 0.2493 |
| **XGBoost** ✅ | **0.8664** | 0.2164 | **0.7897** | 0.3397 |
| XGBoost + SMOTE | 0.8573 | 0.3681 | 0.4960 | 0.4226 |

**Selected model: XGBoost**  
Highest AUC-ROC (0.8664) and Recall (0.7897).  
In credit risk, missing a defaulter is more costly than a 
false alarm — high Recall is prioritised.

---

## 🔑 Key Findings

- **RevolvingUtilizationOfUnsecuredLines** is the strongest 
  predictor of default across all models
- **Past delinquency behaviour** (90DaysLate, 30-59DaysLate) 
  are the next most important signals
- Class imbalance (6.7% positive) requires careful handling — 
  accuracy alone is a misleading metric; AUC-PR is used 
  alongside AUC-ROC for honest evaluation

---

## 🛠️ Tech Stack

- **Language**: Python 3.10
- **Data**: pandas, NumPy
- **ML**: scikit-learn, XGBoost, imbalanced-learn
- **Visualisation**: matplotlib, seaborn
- **Dashboard**: Streamlit
- **Environment**: Anaconda

---

## 🚀 Quick Start
```bash
# 1. Clone the repo
git clone https://github.com/yikhow/credit-default-prediction.git
cd credit-default-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle and place in data/
#    https://www.kaggle.com/c/GiveMeSomeCredit

# 4. Run full pipeline
python src/preprocess.py
python src/model.py

# 5. Launch dashboard
streamlit run app.py
```

---

## 📁 Notebooks

| Notebook | Description |
|---|---|
| `01_EDA.ipynb` | Data exploration, missing values, outlier detection |
| `02_preprocessing.ipynb` | Cleaning, imputation, scaling, train/test split |
| `03_modeling.ipynb` | Training LR, RF, XGBoost, XGBoost+SMOTE |
| `04_evaluation.ipynb` | Confusion matrix, ROC/PR curves, feature importance, business recommendations |
```