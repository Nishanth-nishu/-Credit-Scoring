"""
=============================================================================
PROJECT 1: Behavioral Credit Scoring System
SQL Analytics & Feature Engineering Engine
=============================================================================
Dataset: UCI Default of Credit Card Clients (30,000 real records)
Business Case:
  - This engine performs the heavy lifting for raw data transformation.
  - SQL-engineered features (Utilization ratios, delinquency lags) are the
    primary inputs for the subsequent Machine Learning risk model.
=============================================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

BASE = Path(".")
DATA_DIR = Path(".")
DB   = BASE / "credit_scoring.db"

# ---------------------------------------------------------------------------
# 1. SCHEMA DESIGN
# ---------------------------------------------------------------------------
DDL = """
DROP TABLE IF EXISTS credit_history;

CREATE TABLE credit_history (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    limit_bal              REAL    NOT NULL,
    sex                    INTEGER,
    education              INTEGER,
    marriage               INTEGER,
    age                    INTEGER,
    pay_status_sep         INTEGER,
    pay_status_aug         INTEGER,
    pay_status_jul         INTEGER,
    pay_status_jun         INTEGER,
    pay_status_may         INTEGER,
    pay_status_apr         INTEGER,
    bill_amt_sep           REAL,
    bill_amt_aug           REAL,
    bill_amt_jul           REAL,
    bill_amt_jun           REAL,
    bill_amt_may           REAL,
    bill_amt_apr           REAL,
    pay_amt_sep            REAL,
    pay_amt_aug            REAL,
    pay_amt_jul            REAL,
    pay_amt_jun            REAL,
    pay_amt_may            REAL,
    pay_amt_apr            REAL,
    default_label          INTEGER NOT NULL,
    ingested_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (SQL -> ML CONNECTION)
# ---------------------------------------------------------------------------

QUERIES = {

    # Highlight 1: Behavioral Risk Features (Feature Engineering for ML)
    "ml_feature_engineering_preview": """
        SELECT
            id,
            limit_bal,
            -- Utilization Ratio: How much limit is the user exhausting?
            CASE WHEN limit_bal > 0 THEN bill_amt_sep / limit_bal ELSE 0 END AS feature_utilization,
            -- Delinquency Intensity: Weighted sum of payment delays
            (pay_status_sep * 1.5 + pay_status_aug * 1.0) AS feature_delinquency_weight,
            -- Outcome for Training
            default_label AS target
        FROM credit_history
        LIMIT 10;
    """,

    # Highlight 2: Portfolio Health Profile
    "risk_decile_analysis": """
        WITH scored AS (
            SELECT
                id,
                limit_bal,
                default_label,
                (pay_status_sep * 2.0 + pay_status_aug) AS raw_agg_score
            FROM credit_history
        ),
        deciled AS (
            SELECT
                *,
                NTILE(10) OVER (ORDER BY raw_agg_score DESC) AS risk_decile
            FROM scored
        )
        SELECT
            risk_decile,
            COUNT(*)                                AS total_customers,
            ROUND(AVG(default_label) * 100, 2)     AS default_rate_pct,
            ROUND(AVG(limit_bal), 0)                AS avg_limit_balance
        FROM deciled
        GROUP BY risk_decile
        ORDER BY risk_decile;
    """,

    # Highlight 3: Portfolio Aggregates
    "portfolio_kpis": """
        SELECT
            COUNT(*)                                         AS total_records,
            ROUND(AVG(limit_bal), 0)                        AS avg_credit_limit,
            ROUND(AVG(default_label) * 100, 2)              AS overall_default_rate_pct
        FROM credit_history;
    """
}

# ---------------------------------------------------------------------------
# 3. EXECUTION ENGINE
# ---------------------------------------------------------------------------

def process_and_load():
    csv_path = DATA_DIR / "uci_credit_default.csv"
    if not csv_path.exists():
        print(f"❌ Data file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Descriptive Rename Mapping
    mapping = {
        'X1': 'limit_bal', 'X2': 'sex', 'X3': 'education', 'X4': 'marriage', 'X5': 'age',
        'X6': 'pay_status_sep', 'X7': 'pay_status_aug', 'X8': 'pay_status_jul',
        'X9': 'pay_status_jun', 'X10': 'pay_status_may', 'X11': 'pay_status_apr',
        'X12': 'bill_amt_sep', 'X13': 'bill_amt_aug', 'X14': 'bill_amt_jul',
        'X15': 'bill_amt_jun', 'X16': 'bill_amt_may', 'X17': 'bill_amt_apr',
        'X18': 'pay_amt_sep', 'X19': 'pay_amt_aug', 'X20': 'pay_amt_jul',
        'X21': 'pay_amt_jun', 'X22': 'pay_amt_may', 'X23': 'pay_amt_apr',
        'Y': 'default_label'
    }
    df = df.rename(columns=mapping)

    con = sqlite3.connect(DB)
    con.executescript(DDL)
    df.to_sql("credit_history", con, if_exists="append", index=False)
    con.commit()
    print(f"✅ Real-world data loaded. {len(df)} records in SQLite.")
    return con

def run_analytics(con):
    results = {}
    for name, sql in QUERIES.items():
        try:
            df = pd.read_sql_query(sql, con)
            results[name] = df
            print(f"\n📊 {name.upper().replace('_', ' ')}")
            print(df.head(10).to_string(index=False))
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
    return results

def plot_visuals(results):
    sns.set_theme(style="darkgrid", palette="muted")
    fig = plt.figure(figsize=(20, 12), facecolor="#f0f2f7")
    fig.suptitle("Fintech Operations — Portfolio Behavioral Credit Dashboard",
                 fontsize=22, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Plot 1: Decile Analysis
    ax1 = fig.add_subplot(gs[0, 0])
    d1 = results["risk_decile_analysis"]
    sns.barplot(x="risk_decile", y="default_rate_pct", data=d1, ax=ax1, palette="Reds_r")
    ax1.set_title("Default Rate by Risk Decile (NTILE 10)", fontsize=14)

    # Plot 2: Exposure Sum
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    kpi = results["portfolio_kpis"].iloc[0]
    summary_text = (
        f"  Total Portfolio View     : {int(kpi['total_records']):,}\n\n"
        f"  Average Credit Limit     : ₹{int(kpi['avg_credit_limit']):,}\n"
        f"  Base Default Rate        : {kpi['overall_default_rate_pct']}%\n"
    )
    ax2.text(0.1, 0.45, summary_text, fontsize=18, family="monospace",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax2.set_title("Portfolio Summary KPIs", fontsize=16, fontweight="bold")

    # Plot 3: SQL Engineered Feature Distribution (Example)
    ax3 = fig.add_subplot(gs[1, 0])
    d_feat = results["ml_feature_engineering_preview"]
    sns.scatterplot(x="feature_utilization", y="feature_delinquency_weight", hue="target", data=d_feat, ax=ax3, s=100)
    ax3.set_title("Engineered Features vs Target (SQL-ML Link)", fontsize=14)

    out_path = BASE / "credit_scoring_sql_dashboard.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"✅ SQL dashboard generated: {out_path}")

if __name__ == "__main__":
    connection = process_and_load()
    if connection:
        analytical_results = run_analytics(connection)
        plot_visuals(analytical_results)
        connection.close()
