"""
=============================================================================
PROJECT 1: Behavioral Credit Scoring System
Machine Learning & Business Decision Engine
=============================================================================
Model: LightGBM Classification on UCI Real-World Credit Data
Business Layers:
  - Approval Strategy: Automated decisioning (Approve/Reject/Review)
  - Financial Impact: Expected Loss (EL) calculation
  - Decision Narrative: Explainability via SHAP
=============================================================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(".")
DATA_PATH = Path("uci_credit_default.csv")

# Industry standard Loss Given Default (LGD) for unsecured credit
CONSTANT_LGD = 0.45 

def run_business_decision_engine():
    print("🚀 Loading data and initializing Fintech Decision Engine...")
    df = pd.read_csv(DATA_PATH)

    # Descriptive feature naming for SHAP and Business reporting
    mapping = {
        'X1': 'Credit_Limit', 'X5': 'Age',
        'X6': 'Pay_Status_Sep', 'X12': 'Bill_Sep', 'X18': 'Pay_Amt_Sep',
        'X7': 'Pay_Status_Aug', 'X13': 'Bill_Aug', 'X19': 'Pay_Amt_Aug',
        'X8': 'Pay_Status_Jul', 'X14': 'Bill_Jul', 'X20': 'Pay_Amt_Jul',
        'Y': 'Target'
    }
    
    features = [f'X{i}' for i in range(1, 24)]
    X = df[features].rename(columns=mapping)
    y = df['Y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("🛠 Training Credit Risk Model...")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)

    # -----------------------------------------------------------------------
    # 1. DECISION LAYER: APPROVAL STRATEGY
    # -----------------------------------------------------------------------
    print("⚖️ Applying Business Approval Strategy...")
    y_prob = model.predict_proba(X_test)[:, 1]
    
    def get_decision(pd_val):
        if pd_val > 0.60: return "REJECT"
        if pd_val > 0.30: return "MANUAL REVIEW"
        return "APPROVE"

    decisions_df = X_test.copy()
    decisions_df['PD_Score'] = y_prob
    decisions_df['Decision'] = decisions_df['PD_Score'].apply(get_decision)

    # -----------------------------------------------------------------------
    # 2. FINANCIAL IMPACT: EXPECTED LOSS (EL)
    # -----------------------------------------------------------------------
    # Formula: EL = PD * LGD * Exposure (Credit Limit)
    decisions_df['Expected_Loss'] = decisions_df['PD_Score'] * CONSTANT_LGD * decisions_df['Credit_Limit']
    
    print("\n📦 Business Decision Sample (Top 5 Applicants):")
    print(decisions_df[['PD_Score', 'Decision', 'Expected_Loss']].head())

    # Save operational output
    decisions_df.to_csv(BASE / "credit_decisions_report.csv", index=False)

    # -----------------------------------------------------------------------
    # 3. ANALYSIS & VISUALIZATION
    # -----------------------------------------------------------------------
    # Plot 1: Decision Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Decision', data=decisions_df, palette="husl")
    plt.title("Lending Decision Distribution", fontsize=15)
    plt.savefig(BASE / "approval_strategy_distribution.png")
    
    # Plot 2: Total Exposure vs Expected Loss
    plt.figure(figsize=(10, 6))
    total_el = decisions_df['Expected_Loss'].sum()
    total_exposure = decisions_df['Credit_Limit'].sum()
    labels = ['Total Exposure', 'Expected Loss (Risk)']
    values = [total_exposure, total_el]
    plt.bar(labels, values, color=['#3498db', '#e74c3c'])
    plt.title(f"Portfolio Risk Exposure (Total EL: ₹{int(total_el):,})", fontsize=15)
    plt.savefig(BASE / "financial_impact_kpi.png")

    # Plot 3: SHAP Explainability
    print("\n🔮 Generating SHAP Interpretability Plots...")
    explainer = shap.TreeExplainer(model)
    shap_val = explainer.shap_values(X_test)
    shap_vals_to_plot = shap_val[1] if isinstance(shap_val, list) else shap_val
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_vals_to_plot, X_test, show=False)
    plt.title("Business Score Determinants (Decision Logic)", fontsize=16)
    plt.savefig(BASE / "shap_summary.png", bbox_inches='tight')

    print(f"✨ Business engine complete. Reports saved to {BASE}")

if __name__ == "__main__":
    run_business_decision_engine()
