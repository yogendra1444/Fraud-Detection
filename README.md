# 💳 Fraud Detection using Machine Learning

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using data analytics and machine learning techniques. We utilize the `creditcard.csv` dataset which contains anonymized transaction features and a binary `Class` column indicating fraud (1) or not (0).

---

## 📂 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Rows**: 284,807
- **Features**: 30 (including `Time`, `Amount`, and 28 PCA-transformed variables)
- **Target**: `Class` (0 = Non-Fraud, 1 = Fraud)

---

## 🛠️ Tools & Libraries

- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn (Logistic Regression, Random Forest, MLPClassifier)
- Jupyter Notebook / JupyterLab

---

## 🔍 Project Steps

1. **Data Loading & Exploration**
   - Loaded dataset using Pandas
   - Performed basic EDA and visualizations

2. **Preprocessing**
   - Scaled `Amount` and `Time` features using `StandardScaler`
   - Split data into training and test sets

3. **Model Training**
   - Logistic Regression
   - Random Forest Classifier

4. **Model Evaluation**
   - Accuracy, Confusion Matrix, Classification Report
   - Visualizations using Seaborn
   - Feature importance plot for Random Forest

---

## 📈 Results

| Model              | Accuracy  |
|-------------------|-----------|
| Logistic Regression | ~99.3%    |
| Random Forest       | ~99.9%    |

> Note: Due to high class imbalance, precision/recall/F1-score for fraud class is more important than overall accuracy.

---

## 📊 Key Insights

- Fraudulent transactions are extremely rare, making this a highly imbalanced dataset.
- Feature engineering is less critical here due to pre-processed PCA components.
- Random Forest performed better in identifying fraud with fewer false positives.
- Top features based on importance help in understanding model behavior.

---

## ✅ Future Improvements

- Use SMOTE or undersampling to handle class imbalance
- Try advanced models like XGBoost, LightGBM
- Deploy model using Flask or Streamlit (optional)

---

## 📁 File Structure

