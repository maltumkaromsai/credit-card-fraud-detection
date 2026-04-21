# Credit Card Fraud Detection
Detecting fraudulent credit card transactions by training and comparing Logistic Regression, Random Forest, and XGBoost with SMOTE on 284K real-world transactions. Best model achieved ROC-AUC 0.918.

## Problem Statement
Credit card fraud is a major issue in the financial industry. The goal of this project is to build a machine learning model that can accurately detect fraudulent transactions from a highly imbalanced real-world dataset.

## Dataset
- **Source:** Kaggle — Credit Card Fraud Detection Dataset
- **Size:** 284,807 transactions
- **Fraud rate:** 0.17% (highly imbalanced)

## Tech Stack
- Python, Pandas, NumPy, Matplotlib
- Scikit-learn, XGBoost
- SMOTE (imbalanced-learn)
- Google Colab

## Approach
1. Exploratory Data Analysis (EDA) on 284K transactions
2. Applied SMOTE to handle class imbalance — balanced fraud samples from 394 to 227,451
3. Trained and compared three models:
   - Logistic Regression
   - Random Forest
   - XGBoost
4. Hyperparameter tuning using RandomizedSearchCV
5. Validated using Stratified K-Fold Cross-Validation

## Results
| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 13% | 90% | 23% | 0.944 |
| Random Forest | 84% | 85% | 84% | 0.923 |
| XGBoost | 77% | 87% | 82% | 0.933 |
| **Tuned Random Forest (Best)** | **85%** | **84%** | **84%** | **0.918** |

> After hyperparameter tuning with RandomizedSearchCV and Stratified K-Fold Cross-Validation, the Tuned Random Forest delivered the most balanced and reliable performance for real-world fraud detection.
