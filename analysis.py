"""
=============================================================================
PROJECT 1: Behavioral Credit Scoring System (Real-World Case)
Machine Learning & Explainability Engine
=============================================================================
Model: LightGBM Classification on UCI Real-World Credit Data
Explainability: SHAP (Shapley Additive Explanations)
Aims to provide deep insights for Pallav Technologies' Credit Operating System.
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

def train_scoring_model():
    print("🚀 Loading UCI Credit dataset for ML training...")
    df = pd.read_csv(DATA_PATH)

    # Descriptive feature naming for SHAP interpretability
    mapping = {
        'X1': 'Credit_Limit', 'X5': 'Age',
        'X6': 'Pay_Status_Sep', 'X12': 'Bill_Sep', 'X18': 'Pay_Amt_Sep',
        'X7': 'Pay_Status_Aug', 'X13': 'Bill_Aug', 'X19': 'Pay_Amt_Aug',
        'X8': 'Pay_Status_Jul', 'X14': 'Bill_Jul', 'X20': 'Pay_Amt_Jul',
        'Y': 'Target'
    }
    
    # We select key behavioral features for the model
    # X6-X11 (Pay Status), X1-X5 (Limit, Demographics), X12-X17 (Bill), X18-X23 (Pay)
    features = [f'X{i}' for i in range(1, 24)]
    X = df[features].rename(columns=mapping)
    y = df['Y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("🛠 Training LightGBM Model (optimizing for Credit Risk)...")
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        random_state=42,
        class_weight='balanced', # Crucial for financial default data
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=50)])

    # Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n📈 Model Performance Summary:")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Explainability with SHAP
    print("\n🔮 Computing SHAP values for model explainability...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Note: For LightGBM, shap_values is a list for multi-class [class0, class1] or just a single array for binary
    # Recently, it often returns class0 and class1 separately and it's class 1 we care about for "Default"
    if isinstance(shap_values, list):
        shap_vals_to_plot = shap_values[1]
    else:
        shap_vals_to_plot = shap_values

    # Plot 1: SHAP Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_vals_to_plot, X_test, show=False)
    plt.title("Credit Score Determinants (SHAP Global Interpretability)", fontsize=16)
    plt.savefig(BASE / "shap_summary.png", bbox_inches='tight')
    plt.close()

    # Plot 2: Feature Importance (Built-in)
    plt.figure(figsize=(12, 8))
    lgb.plot_importance(model, max_num_features=15, importance_type='gain', ax=plt.gca(), color='skyblue')
    plt.title("Top Predictors for Loan Default (Information Gain)", fontsize=16)
    plt.savefig(BASE / "feature_importance.png", bbox_inches='tight')
    plt.close()

    print(f"✨ Model and explanations generated at {BASE}")

if __name__ == "__main__":
    train_scoring_model()
