# Predicting and Analyzing E-Commerce Customer Loyalty

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit_Learn%20%7C%20XGBoost-orange.svg)
![Data Analysis](https://img.shields.io/badge/Data_Analysis-Pandas%20%7C%20NumPy-brightgreen.svg)

## 📌 Project Overview
In today's highly competitive e-commerce landscape, treating all customers uniformly results in low engagement and poor marketing ROI. This project utilizes machine learning to transform raw transactional data into actionable customer intelligence. 

The goal of this project is twofold:
1. **Predictive Modeling:** Forecast the probability of a customer making a purchase within the next 90 days.
2. **Customer Segmentation:** Cluster customers into distinct behavioral groups to enable highly targeted marketing interventions and reduce churn.

## 📊 Dataset
The analysis uses the **Online Retail II** dataset (sourced from Kaggle), containing over 1,000,000 transactional records between 2009–2011 for a UK-based online retailer. 
* Data was heavily cleaned (removing cancellations and missing IDs), resulting in a finalized dataset of **805,549 completed orders**. 
* A strict temporal split was utilized to prevent data leakage and simulate a real-world predictive scenario.

## 🛠️ Methodology & Pipeline

### 1. Advanced Feature Engineering
Expanded the classic **RFM** (Recency, Frequency, Monetary) framework to capture deeper behavioral nuances. Engineered features include:
* **Tenure:** Days since the customer's first purchase.
* **Variety:** Count of unique items purchased.
* **Average Order Value (AOV):** Total revenue divided by total orders.
* **Inter-Purchase Time:** Average time elapsed between purchases.
*(Note: Log transformations were applied to handle right-skewed data and normalize distributions).*

### 2. Predictive Modeling (Classification)
Trained and evaluated multiple classification models to predict 90-day repurchase probability. Models were optimized using 3-fold Cross-Validation and Grid Search:
* **Logistic Regression (Baseline):** L2 Regularization applied to handle multicollinearity.
* **Random Forest:** Chosen as the deployment model for its balanced Precision/Recall and interpretability.
* **XGBoost:** Yielded the highest precision, ideal for high-cost, hyper-targeted campaigns.

### 3. Customer Segmentation (Clustering)
Applied **K-Means Clustering** (optimized via the Elbow Method at `k=4`) and Principal Component Analysis (PCA) to group customers based on behavioral patterns.

## 🏆 Key Findings & Business Impact
* **Best Performing Model:** **Random Forest** achieved an **AUC-ROC of 0.817**, with a well-balanced precision (0.71) and recall (0.71).
* **Actionable Segments:** Successfully identified 4 distinct customer tiers: *High-Value (Champions)*, *At-Risk*, *Occasional*, and *New*.
* **Behavioral Insights:** Feature importance analysis revealed that **Recency** and **Frequency** are significantly stronger indicators of future loyalty than total monetary spend alone. Marketing efforts should prioritize habitual purchasing over maximizing single-order values.

# Authors
- Quynh Trinh
- Julia Raygoza
- Taige Mueller
