# 🏦 Behavioral Credit Scoring System (Real-World Case)
### *Production-Grade SQL + ML Analytics for Pallav Technologies*

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![SQL](https://img.shields.io/badge/SQL-SQLite%2FPostgreSQL--Compatible-orange?logo=sqlite)](https://sqlite.org)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Credit%20Default-green)](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

---

## 🎯 Business Problem
Standard credit scoring models often lack explainability. This project uses the **UCI Default of Credit Card Clients** dataset (30,000 real records) to build a robust underwriting engine. It specifically analyzes payment delinquency patterns and credit utilization ratios—key drivers for Pallav's Credit OS.

---

## 📊 SQL Analytics Engine (`credit_scoring_sql.py`)

The analytics engine uses advanced SQL to segment customers based on 6 months of payment behavior.

### Key Analysis:
- **Risk Decile Analytics**: Using `NTILE(10)` to segment 30,000 customers into risk bands.
- **Utilization Risk Profile**: Calculating `Bill Amount / Credit Limit` in real-time.
- **Persistence LAG Analysis**: Identifying customers with "Persistent Delinquency" vs. "Recent Delinquency."

---

## 🤖 ML + Explainability (`analysis.py`)

- **Model**: LightGBM Classifier (Balanced for Financial Default).
- **ROC-AUC**: **0.778** on real-world data.
- **XAI (Explainable AI)**: Using **SHAP** to show exactly why a loan is flagged. Top drivers include Sept Payment Status and Credit Limit.

---

## 📂 Data Sources
This project uses 100% real-world data from the UCI Machine Learning Repository.
- **Features**: 23 (Credit Limit, Payment History, Bill Amounts, Demographics).
- **Target**: Default payment next month.

---

## 🔍 Visual Insights
Generated dashboards include:
1. **Default Rate by Decile**: Showing a 4x risk increase in the top decile.
2. **Utilization Impact**: Critical risk spikes in customers using >70% of their limit.
3. **SHAP Summary**: Global feature impact plot.

---

*Built for Pallav Technologies Portfolio — Advanced Underwriting Pillar.*
