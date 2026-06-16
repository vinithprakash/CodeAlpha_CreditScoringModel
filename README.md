# 💳 CreditIQ — AI Credit Risk Analyzer
### CodeAlpha Machine Learning Internship — Task 1: Credit Scoring Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Logistic%20Regression-orange?logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-81.5%25-brightgreen)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.851-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

**CreditIQ** predicts whether a loan applicant is a **Good** or **Bad** credit
risk based on financial and demographic information — and explains *why* it
made that decision, in plain language and visuals.

Instead of a plain script with printed numbers, this project ships as a
**full interactive web dashboard** built with Streamlit:

- 🎯 **FICO-style gauge** showing a 300–850 predicted credit score
- 🟢🔴 **Live "why" explanation** — top factors pushing the score up or down
- 📊 **Model comparison** — Logistic Regression vs Decision Tree vs Random Forest
- 📁 **Dataset explorer** with live charts

---

## 🖼️ App Preview (Tabs)

| Tab | What it shows |
|---|---|
| 🔍 **Risk Predictor** | Input form → predicted score, risk badge, gauge chart, and a per-applicant explanation chart |
| 📊 **Model Insights** | Accuracy / Precision / Recall / F1 / ROC-AUC comparison table, ROC curves, confusion matrix, feature-influence chart |
| 📁 **Dataset Explorer** | Raw data preview, class balance, risk-by-purpose chart, summary statistics |

---

## 🗂️ Project Structure

```
CodeAlpha_CreditScoringModel/
│
├── app.py                     # Streamlit dashboard (main entry point)
├── train.py                   # Trains & compares 3 models, saves the best
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── .streamlit/
│   └── config.toml            # Dark "CreditIQ" theme
│
├── dataset/
│   ├── generate_dataset.py    # Creates the synthetic credit dataset
│   └── credit_data.csv        # 1,000 applicant records 
│
├── model/
│   ├── credit_model.pkl        # Trained pipeline (deployed model)
│   ├── feature_info.json       # Feature names, categories & ranges
│   └── metrics.json             # Evaluation metrics for all 3 models
│
└── utils/
    ├── model_comparison.png    # Bar chart comparing models
    ├── roc_curve.png            # ROC curves for all 3 models
    ├── confusion_matrix.png    # Confusion matrix (Logistic Regression)
    └── feature_importance.png # Coefficient-based feature influence chart
```

---

## 📊 Dataset

Since the project must run **fully offline** (no UCI download required), a
**realistic synthetic dataset** is generated locally — structurally identical
to the classic German Credit dataset, but created from a transparent scoring
formula so the patterns are genuine and explainable.

| Property | Value |
|---|---|
| Records | 1,000 |
| Good Credit | ~69% |
| Bad Credit | ~31% |
| Numeric features | Age, Job (skill level), Monthly Income, Employment Years, Credit Amount, Duration, Existing Loans, Dependents, Debt-to-Income, Credit per Month |
| Categorical features | Sex, Housing, Saving Accounts, Checking Account, Purpose, Payment History |
| Target | `Risk` → `good` / `bad` |

> Want the *real* UCI German Credit dataset instead? Download it and place it
> as `dataset/credit_data.csv` with matching column names — `train.py` works
> unchanged.

---

## 🧠 Model & Approach

| Step | Details |
|---|---|
| Feature Engineering | One-Hot Encoding (categorical) + Standard Scaling (numeric); derived `Debt_to_Income` and `Credit_per_Month` ratios |
| Algorithms Compared | Logistic Regression, Decision Tree, Random Forest |
| Deployed Model | **Logistic Regression** (best ROC-AUC + fully explainable via coefficients) |
| Evaluation Metrics | Accuracy, Precision, Recall, F1-Score, ROC-AUC |

### 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ✅ | 0.815 | 0.834 | 0.913 | 0.872 | **0.851** |
| Random Forest | 0.770 | 0.799 | 0.891 | 0.843 | 0.815 |
| Decision Tree | 0.690 | 0.771 | 0.783 | 0.777 | 0.624 |

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset
```bash
python dataset/generate_dataset.py
```
> Creates `dataset/credit_data.csv` (1,000 records). Already included, but
> re-run anytime for a fresh sample.

### 3. Train the Model
```bash
python train.py
```
> Trains all 3 models, prints a full comparison report, and saves:
> - `model/credit_model.pkl`
> - `model/feature_info.json`
> - `model/metrics.json`
> - All plots in `utils/`

### 4. Launch the Dashboard
```bash
streamlit run app.py
```
> Opens automatically at **http://localhost:8501**

---

## 🎮 How to Use the App

1. Open the **🔍 Risk Predictor** tab.
2. Fill in the applicant's personal, banking, and loan details.
3. Click **🚀 Analyze Credit Risk**.
4. View:
   - A **risk badge** (Excellent / Good / Fair / Poor)
   - A **gauge chart** showing the predicted credit score (300–850)
   - Probability of Good vs Bad credit
   - A **factor chart** showing exactly which inputs helped or hurt the score

---

## 🔍 Explainability — How "Why" Works

The deployed Logistic Regression model produces a coefficient for every
(scaled) input feature. For any applicant, multiplying their transformed
feature values by these coefficients gives each feature's **contribution to
the prediction** — positive values push toward "Good Credit", negative values
push toward "Bad Credit". This is the same logic used in real-world
**credit scorecards**.

---

## 🔧 Key Concepts Used

- **Logistic Regression** for binary classification with probability outputs
- **One-Hot Encoding & Standard Scaling** via `ColumnTransformer` + `Pipeline`
- **Feature Engineering** — Debt-to-Income & Credit-per-Month ratios
- **Model Evaluation** — Precision, Recall, F1, ROC-AUC, Confusion Matrix
- **Model Explainability** — coefficient-based contribution analysis
- **Interactive Dashboards** — Streamlit + Plotly gauge & bar charts

---

## 🔮 Future Improvements

- Add SHAP values for deeper, model-agnostic explanations
- Support batch predictions via CSV upload
- Add hyperparameter tuning (GridSearchCV) for Random Forest / XGBoost
- Deploy on Streamlit Community Cloud for a live public demo

---

## 👨‍💻 Author

**Vinith Prakash B**
Machine Learning Intern — CodeAlpha
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

---

## 🏢 About CodeAlpha

CodeAlpha is a leading software development company driving innovation
through AI and intelligent systems.
🌐 [www.codealpha.tech](https://www.codealpha.tech)

---

## 📄 License

This project is licensed under the MIT License.
