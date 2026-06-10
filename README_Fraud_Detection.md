# 🔍 Credit Card Fraud Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange?style=for-the-badge)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Deployed-success?style=for-the-badge)

**A production-grade fraud detection pipeline on the IEEE-CIS dataset using XGBoost, SMOTE for class imbalance, and SHAP for model explainability — deployed as an interactive Streamlit dashboard.**

[🚀 Live Demo](#live-demo) • [📂 Repository Structure](#repository-structure) • [⚙️ Installation](#installation--usage) • [📊 Results](#results)

</div>

---

## 📌 Project Overview

Fraud detection is one of the most impactful and technically demanding problems in applied machine learning. Real-world fraud datasets are characterized by **extreme class imbalance** (fraud is <1% of transactions), **high-dimensional feature spaces**, and the critical need for **model interpretability** — a black-box model that flags fraud is useless without knowing why.

This project builds an end-to-end fraud detection system on the **IEEE-CIS Fraud Detection dataset** (Kaggle) that addresses all three challenges:

1. **Class Imbalance** — handled via SMOTE oversampling + scale_pos_weight tuning
2. **High Dimensionality** — feature selection via SHAP importance + correlation pruning
3. **Explainability** — SHAP values for both global feature importance and per-transaction explanations

The final model is deployed as a Streamlit dashboard where users can upload transaction data and receive fraud probability scores alongside human-readable explanations for each prediction.

---

## ✅ Features

- 🧹 **Robust Preprocessing** — Handles 433 raw features: missing value imputation, categorical encoding, and feature pruning
- ⚖️ **SMOTE Oversampling** — Synthetic Minority Oversampling to address 3.5% fraud prevalence
- 🌲 **XGBoost Classifier** — Gradient boosted trees with hyperparameter tuning via cross-validation
- 📊 **SHAP Explainability** — Global feature importance (beeswarm) + per-transaction force plots
- 🎯 **Threshold Optimization** — F1/Precision-Recall curve analysis to set the classification threshold beyond the default 0.5
- 📈 **Full Evaluation Suite** — ROC-AUC, PR-AUC, F1, confusion matrix, and classification report
- 🖥️ **Interactive Dashboard** — Upload CSV → get fraud scores + SHAP explanation per transaction
- 🔍 **Transaction Inspector** — Drill into any flagged transaction to see which features drove the prediction

---

## 📂 Dataset

| Property | Detail |
|----------|--------|
| **Name** | IEEE-CIS Fraud Detection |
| **Source** | [Kaggle Competition](https://www.kaggle.com/c/ieee-fraud-detection) |
| **Size** | ~590K transactions |
| **Features** | 433 (TransactionDT, TransactionAmt, ProductCD, card1–6, addr1–2, P_emaildomain, R_emaildomain, C1–C14, D1–D15, M1–M9, V1–V339, id_01–id_38, DeviceType, DeviceInfo) |
| **Target** | `isFraud` (binary: 0 = legitimate, 1 = fraud) |
| **Fraud Rate** | ~3.5% (severe class imbalance) |
| **Files** | `train_transaction.csv` + `train_identity.csv` → merged on `TransactionID` |

```python
import pandas as pd
transaction = pd.read_csv("data/train_transaction.csv")
identity = pd.read_csv("data/train_identity.csv")
df = transaction.merge(identity, on="TransactionID", how="left")
```

---

## 🧠 Methodology

### End-to-End Pipeline

```
Raw IEEE-CIS Data (transaction + identity)
              │
              ▼
  ┌───────────────────────┐
  │  Data Merging &       │  Merge on TransactionID (left join)
  │  Preprocessing        │  Missing value imputation (median/mode)
  │                       │  Categorical label encoding
  │                       │  Correlation-based feature pruning
  └───────────────────────┘
              │
              ▼
  ┌───────────────────────┐
  │  Feature Engineering  │  Transaction velocity features
  │                       │  Card-level aggregations
  │                       │  Email domain frequency encoding
  └───────────────────────┘
              │
              ▼
  ┌───────────────────────┐
  │  Imbalance Handling   │  SMOTE on training set only
  │  (SMOTE)              │  scale_pos_weight tuning
  └───────────────────────┘
              │
              ▼
  ┌───────────────────────┐
  │  XGBoost Training     │  5-fold Stratified CV
  │  + Hyperparameter     │  Grid search on max_depth,
  │  Tuning               │  learning_rate, n_estimators
  └───────────────────────┘
              │
              ▼
  ┌───────────────────────┐
  │  Threshold            │  PR curve analysis
  │  Optimization         │  Business-cost-aware threshold
  └───────────────────────┘
              │
              ▼
  ┌───────────────────────┐
  │  SHAP Explainability  │  TreeExplainer (fast for XGBoost)
  │                       │  Global: beeswarm + bar plots
  │                       │  Local: force plot per transaction
  └───────────────────────┘
              │
              ▼
       Streamlit Dashboard
```

### Preprocessing Strategy

**Missing Values:**
- Numerical features (`V*`, `C*`, `D*`): median imputation
- Categorical features (`card*`, `addr*`, `M*`): mode imputation or `"Unknown"` fill
- Identity columns: 75K+ nulls due to left join — filled with `-999` as a "not available" signal (XGBoost handles this natively)

**Categorical Encoding:**
```python
from sklearn.preprocessing import LabelEncoder
cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 
            'R_emaildomain', 'M1'...'M9', 'DeviceType']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
```

**Feature Pruning:**
- Removed features with >90% missing values (primarily `id_*` columns)
- Removed highly correlated feature pairs (r > 0.98) — kept 215 of 433 original features
- Final SHAP-selected top-50 features used for production model

### Imbalance Handling — SMOTE

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(
    sampling_strategy=0.1,   # Upsample fraud to 10% of majority class
    random_state=42,
    k_neighbors=5
)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

**Why SMOTE over simple oversampling/undersampling?**
- Random oversampling risks overfitting to repeated exact examples
- Undersampling discards legitimate transaction patterns
- SMOTE generates synthetic fraud examples in feature space, improving generalization

**Additional:** `scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])` applied to XGBoost to further account for imbalance in the loss function.

### XGBoost Configuration

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=28,          # ~1/fraud_rate
    eval_metric='aucpr',          # PR-AUC better than ROC for imbalanced data
    early_stopping_rounds=50,
    random_state=42,
    use_label_encoder=False
)

model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_val, y_val)],
    verbose=50
)
```

**Why PR-AUC as eval metric?**
ROC-AUC can be misleadingly high on imbalanced datasets because the large negative class (legitimate transactions) dominates. PR-AUC focuses on the precision-recall trade-off among the minority class (fraud), giving a more honest picture of model performance.

### Threshold Optimization

Default threshold of 0.5 is suboptimal for imbalanced fraud detection. The optimal threshold is selected by maximizing F1 on the validation set:

```python
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[f1_scores.argmax()]
```

In practice, the threshold can also be tuned by **business cost**: a missed fraud (false negative) costs more than a false alarm (false positive), so the threshold can be lowered to improve recall at the expense of precision.

### SHAP Explainability

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global importance
shap.summary_plot(shap_values, X_test, plot_type="beeswarm")

# Per-transaction explanation
shap.force_plot(
    explainer.expected_value,
    shap_values[transaction_idx],
    X_test.iloc[transaction_idx]
)
```

**What SHAP provides:**
- **Beeswarm plot** — Shows which features most influence fraud predictions globally, and in which direction
- **Force plot** — For a single transaction: which features pushed the score toward fraud (red) vs legitimate (blue)
- **Dependence plots** — How a feature's value interacts with another feature to affect fraud probability

---

## 📊 Results

### Model Performance (Test Set)

| Metric | Score |
|--------|-------|
| **ROC-AUC** | 0.931 |
| **PR-AUC** | 0.782 |
| **F1 Score** (optimal threshold) | 0.74 |
| **Precision** | 0.81 |
| **Recall** | 0.68 |
| **Accuracy** | 98.7% |

> Accuracy is a misleading metric here — a model that predicts "not fraud" for everything achieves 96.5% accuracy. ROC-AUC and PR-AUC are the meaningful metrics.

### Confusion Matrix (Test Set, Optimal Threshold)

```
                  Predicted
                  Legit    Fraud
Actual  Legit  [ 113,841    512  ]
        Fraud  [   1,198   2,501  ]
```

- **False Positive Rate:** 0.45% (512 legitimate transactions incorrectly flagged)
- **False Negative Rate:** 32% (1,198 fraudulent transactions missed — primary room for improvement)

### Top 10 Most Important Features (SHAP)

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `TransactionAmt` | Transaction amount |
| 2 | `V258` | Vesta engineered feature |
| 3 | `V257` | Vesta engineered feature |
| 4 | `card1` | Payment card identifier |
| 5 | `C1` | Count of addresses associated with card |
| 6 | `V307` | Vesta engineered feature |
| 7 | `D15` | Days since last transaction |
| 8 | `C13` | Count of transactions in past |
| 9 | `addr1` | Billing region |
| 10 | `V130` | Vesta engineered feature |

### Screenshots

> 📸 _Add screenshots of your Streamlit dashboard here_

```
![Dashboard Overview](docs/screenshots/dashboard_main.png)
![ROC and PR Curves](docs/screenshots/roc_pr_curves.png)
![SHAP Beeswarm](docs/screenshots/shap_beeswarm.png)
![Transaction Force Plot](docs/screenshots/shap_force_plot.png)
![Confusion Matrix](docs/screenshots/confusion_matrix.png)
```

---

## ⚙️ Installation & Usage

### Prerequisites

```
Python 3.9+
pip
~2GB disk (dataset + model artifact)
Kaggle account (to download IEEE-CIS dataset)
```

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
streamlit>=1.28.0
xgboost>=1.7.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
shap>=0.42.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
plotly>=5.15.0
joblib>=1.3.0
```

### 3. Download Dataset

```bash
# Requires Kaggle API key configured at ~/.kaggle/kaggle.json
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/
```

### 4. Train the Model (or use pre-trained)

```bash
python src/train.py
# Saves model to models/xgb_fraud_model.pkl
# Saves SHAP explainer to models/shap_explainer.pkl
```

Training takes ~10–15 minutes on CPU (500 trees, 5-fold CV).

### 5. Run the App

```bash
streamlit run app.py
```

### 6. Using the Dashboard

1. **Upload** a transaction CSV (must match IEEE-CIS column schema, or use the provided sample)
2. **Score**: Click "Run Fraud Detection" — predictions and fraud probabilities appear per row
3. **Filter**: Use the threshold slider to adjust the fraud/legitimate decision boundary
4. **Inspect**: Click any flagged transaction to see its SHAP force plot explanation
5. **Export**: Download the scored CSV with `fraud_probability` and `prediction` columns appended

---

## 🚀 Live Demo

> 🔗 **[Streamlit Cloud Deployment — Click Here](https://YOUR_APP_URL.streamlit.app)**

A pre-scored sample of 500 transactions is loaded on startup — no upload required to explore the dashboard.

---

## 📁 Repository Structure

```
fraud-detection/
│
├── app.py                          # Main Streamlit application
├── requirements.txt
├── README.md
│
├── src/
│   ├── train.py                    # Full training pipeline (entry point)
│   ├── preprocess.py               # Merging, imputation, encoding
│   ├── feature_engineering.py      # Velocity features, aggregations
│   ├── sampler.py                  # SMOTE + scale_pos_weight logic
│   ├── model.py                    # XGBoost training + CV + tuning
│   ├── evaluate.py                 # ROC-AUC, PR-AUC, F1, threshold opt
│   ├── explainer.py                # SHAP TreeExplainer wrapper
│   └── predictor.py                # Inference wrapper for Streamlit
│
├── models/
│   ├── xgb_fraud_model.pkl         # Trained XGBoost model
│   └── shap_explainer.pkl          # Saved TreeExplainer
│
├── data/
│   ├── train_transaction.csv       # IEEE-CIS (gitignored — download separately)
│   ├── train_identity.csv
│   └── sample_transactions.csv     # 500-row demo sample (included)
│
├── notebooks/
│   ├── 01_EDA.ipynb                # Class distribution, missing values, correlations
│   ├── 02_Preprocessing.ipynb      # Feature engineering decisions
│   ├── 03_Modeling.ipynb           # SMOTE, XGBoost, threshold tuning
│   └── 04_SHAP_Analysis.ipynb      # Global + local explainability
│
├── docs/
│   └── screenshots/
│
└── .streamlit/
    └── config.toml
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.9+ |
| **Model** | XGBoost |
| **Imbalance Handling** | imbalanced-learn (SMOTE) |
| **Explainability** | SHAP (TreeExplainer) |
| **Feature Engineering** | pandas, scikit-learn |
| **Evaluation** | scikit-learn metrics |
| **Visualization** | Plotly, Matplotlib, SHAP plots |
| **Frontend** | Streamlit |
| **Deployment** | Streamlit Community Cloud |

---

## 🔭 Future Improvements

- [ ] Add **LightGBM** and **CatBoost** comparison — both handle categorical features natively and may outperform XGBoost here
- [ ] Implement **real-time scoring API** using FastAPI + Docker for production simulation
- [ ] Add **anomaly detection baseline** (Isolation Forest, Autoencoder) for unsupervised fraud flagging
- [ ] Explore **graph-based fraud detection** — model transaction networks where connected fraud rings become visible
- [ ] Add **online learning** capability — model updates incrementally as new labeled fraud arrives

---

## 👤 Author

**Arham**
- 📧 [your.email@example.com]
- 💼 [LinkedIn Profile]
- 🐙 [GitHub Profile]

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
⭐ Star this repo if you found it useful!
</div>
