"""
=============================================================================
PROJECT 1: Behavioral Credit Scoring System (Real-World Case)
SQL Analytics Engine — Production-Grade Edition
=============================================================================
Dataset: UCI Default of Credit Card Clients (30,000 real records)
Demonstrates:
  - Deep SQL analysis on historical payment delinquency (PAY_0-6)
  - Credit Utilization modeling (Bill Amount vs Balance Limit)
  - Aggregated Risk Decile analysis on real financial behaviors
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
# 1. SCHEMA DESIGN (Real-World Credit Schema)
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

CREATE INDEX idx_default_limit ON credit_history(default_label, limit_bal);
CREATE INDEX idx_age_default    ON credit_history(age, default_label);
"""

# ---------------------------------------------------------------------------
# 2. ANALYTICAL SQL QUERIES (Production-Grade Analytics)
# ---------------------------------------------------------------------------

QUERIES = {

    # Query 1: Risk Decile Analysis (Real Data)
    "risk_decile_analysis": """
        WITH scored AS (
            SELECT
                id,
                limit_bal,
                default_label,
                -- Use weighted payment delinquency as a proxy score
                (pay_status_sep * 2.0 + pay_status_aug * 1.5 + pay_status_jul) AS delinquency_score
            FROM credit_history
        ),
        deciled AS (
            SELECT
                *,
                NTILE(10) OVER (ORDER BY delinquency_score DESC) AS risk_decile
            FROM scored
        )
        SELECT
            risk_decile,
            COUNT(*)                                AS total_customers,
            SUM(default_label)                      AS defaults,
            ROUND(AVG(default_label) * 100, 2)     AS default_rate_pct,
            ROUND(AVG(limit_bal), 0)                AS avg_limit_balance
        FROM deciled
        GROUP BY risk_decile
        ORDER BY risk_decile;
    """,

    # Query 2: Credit Utilization vs Default (Dynamic Calculation)
    "utilization_risk_profile": """
        WITH util_agg AS (
            SELECT
                id,
                default_label,
                limit_bal,
                -- Calculate Sept Utilization ratio
                CASE WHEN limit_bal > 0 THEN bill_amt_sep / limit_bal ELSE 0 END AS util_ratio
            FROM credit_history
        ),
        util_buckets AS (
            SELECT
                *,
                CASE
                    WHEN util_ratio <= 0.1 THEN '0-10% (Low)'
                    WHEN util_ratio <= 0.3 THEN '10-30% (Standard)'
                    WHEN util_ratio <= 0.7 THEN '30-70% (High)'
                    ELSE                        '70%+ (Critical)'
                END AS util_band
            FROM util_agg
        )
        SELECT
            util_band,
            COUNT(*)                            AS customer_count,
            ROUND(AVG(default_label) * 100, 2) AS default_rate_pct,
            RANK() OVER (ORDER BY AVG(default_label) DESC) AS risk_rank
        FROM util_buckets
        GROUP BY util_band
        ORDER BY default_rate_pct DESC;
    """,

    # Query 3: Delinquency Persistence (LAG Analysis)
    "delinquency_trend_impact": """
        WITH trend AS (
            SELECT
                id,
                default_label,
                pay_status_sep,
                pay_status_aug,
                CASE 
                    WHEN pay_status_sep > 0 AND pay_status_aug > 0 THEN 'Persistent Delinquency'
                    WHEN pay_status_sep > 0 AND pay_status_aug <= 0 THEN 'Recent Delinquency'
                    WHEN pay_status_sep <= 0 AND pay_status_aug > 0 THEN 'Recovering'
                    ELSE 'Healthy'
                END AS delinquency_type
            FROM credit_history
        )
        SELECT
            delinquency_type,
            COUNT(*) AS count,
            ROUND(AVG(default_label) * 100, 2) AS default_rate_pct
        FROM trend
        GROUP BY delinquency_type
        ORDER BY default_rate_pct DESC;
    """,

    # Query 4: Portfolio Summary (KPI Dashboard)
    "portfolio_summary": """
        SELECT
            COUNT(*)                                         AS total_records,
            ROUND(AVG(limit_bal), 0)                        AS avg_credit_limit,
            SUM(default_label)                               AS total_defaults,
            ROUND(AVG(default_label) * 100, 2)              AS overall_default_rate_pct,
            ROUND(AVG(age), 1)                              AS avg_customer_age
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
    print(f"✅ Real-world data loaded into SQLite: {len(df)} records.")
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
    fig = plt.figure(figsize=(20, 12), facecolor="#f0f2f6")
    fig.suptitle("Pallav Technologies — Behavioral Credit Risk Dashboard (UCI Real Data)",
                 fontsize=22, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Plot 1: Decile Analysis
    ax1 = fig.add_subplot(gs[0, 0])
    d1 = results["risk_decile_analysis"]
    sns.barplot(x="risk_decile", y="default_rate_pct", data=d1, ax=ax1, palette="Reds_r")
    ax1.set_title("Default Rate by Delinquency Decile (NTILE Analytics)", fontsize=14)
    ax1.set_ylabel("Default Rate (%)")

    # Plot 2: Utilization Bands
    ax2 = fig.add_subplot(gs[0, 1])
    d2 = results["utilization_risk_profile"]
    sns.barplot(x="util_band", y="default_rate_pct", data=d2.sort_values("default_rate_pct"), ax=ax2, palette="viridis")
    ax2.set_title("Default Risk vs Credit Utilization Ratio", fontsize=14)
    ax2.set_ylabel("Default Rate (%)")

    # Plot 3: Delinquency Persistence
    ax3 = fig.add_subplot(gs[1, 0])
    d3 = results["delinquency_trend_impact"]
    sns.barplot(x="delinquency_type", y="default_rate_pct", data=d3, ax=ax3, palette="magma")
    ax3.set_title("Impact of Persistent vs Recent Delinquency", fontsize=14)
    ax3.set_ylabel("Default Rate (%)")

    # Plot 4: KPI Text box
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    kpi = results["portfolio_summary"].iloc[0]
    summary_text = (
        f"  Total Records Analyze: {int(kpi['total_records']):,}\n\n"
        f"  Overall Default Rate  : {kpi['overall_default_rate_pct']}%\n"
        f"  Avg Credit Limit      : ${kpi['avg_credit_limit']:,}\n"
        f"  Total Delinquent Cases: {int(kpi['total_defaults']):,}\n"
        f"  Avg Customer Age      : {kpi['avg_customer_age']} Yrs\n"
    )
    ax4.text(0.1, 0.5, summary_text, fontsize=16, family="monospace",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
    ax4.set_title("Portfolio Summary KPIs", fontsize=16, fontweight="bold")

    out_path = BASE / "credit_scoring_sql_dashboard.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"✅ Professional dashboard generated: {out_path}")

if __name__ == "__main__":
    connection = process_and_load()
    if connection:
        analytical_results = run_analytics(connection)
        plot_visuals(analytical_results)
        connection.close()
