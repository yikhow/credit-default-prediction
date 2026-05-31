# Credit Default Prediction

A machine learning project that predicts the probability of loan 
default using historical borrower data. Built to demonstrate 
end-to-end data analysis skills — from raw data to an interactive 
prediction dashboard.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![AUC](https://img.shields.io/badge/AUC--ROC-0.8662-brightgreen)

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
│   └── processed/               # Cleaned & split data (auto-generated)
│       ├── X_train.csv / y_train.csv   # 60% — model training
│       ├── X_val.csv   / y_val.csv     # 20% — XGBoost early stopping
│       └── X_test.csv  / y_test.csv    # 20% — final evaluation only
├── notebooks/
│   ├── 01_EDA.ipynb             # Exploratory data analysis
│   ├── 02_preprocessing.ipynb   # Data cleaning & 3-way split
│   ├── 03_modeling.ipynb        # Model training & comparison
│   └── 04_evaluation.ipynb      # Model evaluation & insights
├── src/
│   ├── preprocess.py            # Preprocessing functions
│   ├── model.py                 # Training functions
│   └── predict.py               # Prediction pipeline
├── outputs/
│   ├── models/                  # Saved model pkl files
│   └── artefacts/               # Fitted scaler & imputer pkl files
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

| Model | AUC-ROC | AUC-PR | Precision | Recall | F1-Score | Tuning |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8376 | 0.3500 | 0.1895 | 0.7532 | 0.3028 | None (baseline) |
| Random Forest | 0.8602 | 0.3818 | 0.2345 | 0.7313 | 0.3552 | RandomizedSearchCV |
| **XGBoost** ✅ | **0.8662** | **0.4125** | 0.2152 | **0.7852** | 0.3378 | RandomizedSearchCV + Early Stopping |
| XGBoost + SMOTE | 0.8580 | 0.3715 | 0.3648 | 0.5080 | 0.4246 | Best XGBoost params |

**Selected model: XGBoost (tuned)**  
Highest AUC-ROC (0.8662), AUC-PR (0.4125), and Recall (0.7852).  
In credit risk, missing a defaulter is more costly than a  
false alarm — high Recall and low False Negative Rate (21.48%) are prioritised.

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
| `02_preprocessing.ipynb` | Cleaning, imputation, scaling, 3-way split (60/20/20) |
| `03_modeling.ipynb` | Training LR, RF (tuned), XGBoost (tuned + early stopping), XGBoost+SMOTE |
| `04_evaluation.ipynb` | Confusion matrix, ROC/PR curves, feature importance, business recommendations |
```