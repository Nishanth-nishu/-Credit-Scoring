# Alternative Credit Scoring System (-Credit-Scoring)

This project demonstrates an AI-powered credit underwriting solution designed for thin-file borrowers in emerging markets. It was built specifically for a Data Analyst role at **Pallav Technologies**.

## Key Features
- **Alternative Data Integration**: Evaluates risk using non-traditional proxies like utility payment punctuality and app behavioral data.
- **Advanced Modeling**: Utilizes **LightGBM** for high-efficiency classification on unbalanced datasets.
- **Explainable AI (XAI)**: Implements **SHAP** values to provide human-readable explanations for loan rejection/approval.

## Results
- **AUC-ROC**: 0.73 (Baseline)
- **Primary Predictors**: Utility Payment Punctuality, Location Stability Index.

## Project Structure
- `analysis.py`: Main modeling and interpretability script.
- `data_generator.py`: Synthetic alternative data generator.
- `alt_credit_data.csv`: Sample dataset.
- `*.png`: Visualizations of feature importance and SHAP summary.

---
*Created for Pallav Technologies Data Analyst Portfolio.*
