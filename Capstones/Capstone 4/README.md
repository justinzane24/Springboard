# Credit Card Fraud Detection

## Executive Summary

This project tackles the real-world challenge of detecting fraudulent credit card transactions — where fraud makes up just 0.17% of the data. Using anonymized transaction data, 
I implemented and compared several machine learning techniques to improve fraud detection while reducing false negatives (missed frauds), which are highly costly in practice.

The dataset consists of 284,807 credit card transactions, with 492 labeled as fraudulent. Each transaction includes 28 anonymized principal components (PCA features), along with `Time` and `Amount`.

After extensive preprocessing — including scaling and handling class imbalance with SMOTE — I tested four models: **Logistic Regression**, **Random Forest**, **XGBoost**, and 
**Multilayer Perceptron (MLP)** for anomaly detection. I focused on **PR-AUC**, **Recall**, and **Precision** to measure performance due to the imbalance and cost sensitivity.

Using Random Forest as an example, I demonstrated how to choose the best number of trees to balance performance and computational cost with PR-AUC as the metric. In a similar fashion, I showed how to perform
threshold tuning for both Random Forest and XGBoost.

Finally, I used the Kolmogorov-Smirnov (KS) test to provide businesses with a cost-effective solution to manual review resource limitations. By ordering predicted probabilities above a capture threshold,
we can select a small sample of the overall dataset that contains a percentage (e.g., 90%) of all fraudulent transactions.

**Results:**
- **Random Forest** achieved the best overall performance: **PR-AUC ~0.67**, **Recall ~0.83**, **Precision ~0.81**.
- To maximize anomaly while minimizing false positives, a threshold of 0.30 should be used for the Random Forest model.
- KS analysis for Random Forest showed a business need only manually inspect the top 0.4% (228 of 56,962) of the data to capture 90% of all fraud.

**Recommendations:**  
Deploy the Random Forest model in a real-time system with human review of high-risk alerts. Future improvements could involve cost-sensitive loss functions, real-world threshold optimization, 
and a dashboard for monitoring prediction drift.

---

## Project Structure
---

## Dataset

- **Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions over two days (2013)
- **Fraud cases**: 492 (0.17%)
- **Features**:
  - `V1` through `V28`: PCA components
  - `Amount`: Transaction amount
  - `Time`: Seconds elapsed since first transaction
  - `Class`: 1 = fraud, 0 = normal

---

## Models & Methodology

- **Preprocessing**:
  - Feature scaling with `StandardScaler`
  - Class imbalance handled with **SMOTE (Synthetic Minority Oversampling)**
- **Models Tested**:
  - Logistic Regression (baseline)
  - Random Forest (best performance)
  - XGBoost 
  - Multilayer Perceptron
- **Evaluation Metrics**:
  - PR-AUC
  - Precision, Recall, F1-Score
  - Classification Reports
---

## Key Cross-Validation Results

| Model             | PR-AUC | Precision | Recall |
|------------------|---------|-----------|--------|
| Logistic Regression | 0.7531   | 0.0562      | 0.9138   |
| **Random Forest**       | **0.8449**   | **0.8984**      | **0.8199**   |
| XGBoost         | 0.8331 | 0.6840  | 0.8401 |
| Multilayer Perceptron         | 0.8170    | 0.7249      | 0.8122   |

> *Note: Best model was selected based on balanced trade-off between high fraud capture and low false positives.*

---

##  Tools & Libraries

- Python 3.12.4 (Jupyter Notebook)
- `pandas`, `numpy` – data handling
- `scikit-learn`, `imbalanced-learn` – ML + oversampling
- `xgboost` – gradient boosting classifier
- `matplotlib`, `seaborn` – data visualization

---

## Next Steps

- Deploy the Random Forest model as a scoring API (e.g., Flask/FastAPI)
- Add economic cost function for threshold tuning (false negative = $$$)
- Implement model monitoring + concept drift detection
- Package into a full pipeline using `scikit-learn` or `mlflow`

---

## Author

**Justin Feathers**   
✉ justinzane@gmail.com
