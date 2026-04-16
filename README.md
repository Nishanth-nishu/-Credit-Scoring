# 🏦 Behavioral Credit Scoring & Decision System
### *Lending Approval Strategy + Expected Loss Modeling*

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![SQL](https://img.shields.io/badge/SQL-Advanced%20Window%20Functions-orange?logo=sqlite)](https://sqlite.org)
[![Financial Analysis](https://img.shields.io/badge/Impact-Expected%20Loss-red)](https://en.wikipedia.org/wiki/Expected_loss)

---

## 🎯 Business Decision Layer
This system moves beyond simple prediction to **automated lending orchestration**. It classifies applicants based on Probability of Default (PD) into clear business actions.

- **Approval Logic**:
    - **PD < 0.3**: ✅ **APPROVE** (Standard processing)
    - **PD 0.3 - 0.6**: 🔍 **MANUAL REVIEW** (Higher risk, needs underwriter audit)
    - **PD > 0.6**: ❌ **REJECT** (High risk of default)

---

## 📈 Financial Impact Analysis
We calculate the **Expected Loss (EL)** for each applicant to quantify the portfolio's risk exposure before disbursement.

**Formula**: $EL = PD \times LGD \times Exposure$
- **PD**: Probability of Default (Model Output)
- **LGD**: Loss Given Default (Baseline: 45% for unsecured credit)
- **Exposure**: The requested Credit Limit.

---

## 🛠 SQL -> ML Pipeline Connectivity
A critical component of this project is the seamless link between **SQL Engineering** and **Machine Learning**:
1. **SQL Layer**: Transforms raw payment logs into behavioral features (Credit Utilization, Delinquency Persistence Index).
2. **ML Layer**: Ingests these specialized features to train a high-fidelity LightGBM classifier.
3. **Outcome**: Higher precision by using stateful behavioral data rather than just static snapshots.

---

## 📂 Data & Methodology
- **Dataset**: UCI Credit Default (30,000 real records).
- **Core Model**: LightGBM (Gradient Boosting) optimized for Credit Risk.
- **Explainability**: SHAP (Shapley Additive Explanations) used to justify every rejection for compliance.

---

## 🔍 Visual Insights
- **Decision Distribution**: Breakdown of Approved vs. Rejected applicants.
- **Exposure vs. EL**: Comparison of total credit capacity vs. total risk-weighted loss.
- **SHAP Summary**: Determinants of credit scores in a production environment.

---

*Fintech Data Analyst Portfolio — Decision Engineering Pillar.*
