import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import lightgbm as lgb
import os

# Set Plotting Style
sns.set_theme(style="whitegrid", palette="muted")

def run_credit_scoring_analysis():
    # 1. Load Data
    df = pd.read_csv('/scratch/nishanth.r/pallavi/project_1_credit_scoring/alt_credit_data.csv')
    print("Dataset Loaded. Shape:", df.shape)

    # 2. EDA & Feature Analysis (Business Context)
    # Correlation with Default
    corr = df.drop('user_id', axis=1).corr()['default_label'].sort_values()
    print("\nFeature Correlations with Default:")
    print(corr)

    # 3. Data Preparation
    X = df.drop(['user_id', 'default_label'], axis=1)
    y = df['default_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Model Training (LightGBM)
    # LightGBM is preferred in Fintech for high efficiency and handling non-linear alternative data
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        importance_type='gain'
    )
    
    model.fit(X_train, y_train)

    # 5. Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC-ROC Score: {auc:.4f}")

    # 6. Feature Importance (Business Insights for Pallav)
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance)
    plt.title('Alternative Data Feature Importance (Credit Score Influence)')
    plt.tight_layout()
    plt.savefig('/scratch/nishanth.r/pallavi/project_1_credit_scoring/feature_importance.png')
    print("\nSaved Feature Importance plot.")

    # 7. Model Interpretation with SHAP (Research Validated Recommendation)
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig('/scratch/nishanth.r/pallavi/project_1_credit_scoring/shap_summary.png')
        print("Saved SHAP Summary plot.")
    except Exception as e:
        print(f"SHAP explanation failed or skipped: {e}")

if __name__ == "__main__":
    run_credit_scoring_analysis()
