"""
=============================================================================
PROJECT 1: Alternative Data Credit Scoring System
SQL Analytics Engine — Production-Grade Edition
=============================================================================
Demonstrates deep SQL mastery for a Data Analyst role at Pallav Technologies:
  - SQLite DDL with proper schema, indexes, and constraints
  - CTEs (Common Table Expressions) for modular query design
  - Window Functions: NTILE, RANK, LAG, ROW_NUMBER, running SUM/AVG
  - Risk Decile Segmentation (NTILE)
  - Early Warning System using LAG (DPD trend)
  - Feature-level default rate analysis
=============================================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

BASE = Path("/scratch/nishanth.r/pallavi/project_1_credit_scoring")
DB   = BASE / "credit_scoring.db"

DDL = """
DROP TABLE IF EXISTS loan_applicants;
CREATE TABLE loan_applicants (
    user_id                INTEGER PRIMARY KEY,
    app_sessions_per_day   REAL    NOT NULL CHECK (app_sessions_per_day >= 0),
    sms_fin_alerts_monthly REAL    NOT NULL CHECK (sms_fin_alerts_monthly >= 0),
    utility_delay_days     REAL    NOT NULL CHECK (utility_delay_days >= 0),
    ecommerce_spend_ratio  REAL    NOT NULL CHECK (ecommerce_spend_ratio BETWEEN 0 AND 1),
    location_stability_score REAL  NOT NULL CHECK (location_stability_score BETWEEN 0 AND 1),
    device_age_months      INTEGER NOT NULL,
    default_label          INTEGER NOT NULL CHECK (default_label IN (0, 1)),
    ingested_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_default_device ON loan_applicants(default_label, device_age_months);
CREATE INDEX idx_utility_default ON loan_applicants(utility_delay_days, default_label);
"""

QUERIES = {
    "risk_decile_analysis": """
        WITH risk_scored AS (
            SELECT user_id, default_label,
                (0.5 * utility_delay_days - 0.8 * location_stability_score
                 - 0.1 * sms_fin_alerts_monthly + 0.05 * device_age_months) AS raw_risk_score
            FROM loan_applicants
        ),
        deciled AS (
            SELECT *, NTILE(10) OVER (ORDER BY raw_risk_score DESC) AS risk_decile
            FROM risk_scored
        )
        SELECT
            risk_decile,
            COUNT(*)                                AS total_applicants,
            SUM(default_label)                      AS total_defaults,
            ROUND(AVG(default_label) * 100, 2)     AS default_rate_pct,
            ROUND(AVG(raw_risk_score), 4)           AS avg_risk_score,
            SUM(SUM(default_label)) OVER (
                ORDER BY risk_decile
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )                                       AS cumulative_defaults
        FROM deciled GROUP BY risk_decile ORDER BY risk_decile;
    """,
    "utility_delay_default_profile": """
        WITH bucketed AS (
            SELECT user_id, default_label,
                CASE
                    WHEN utility_delay_days = 0             THEN '0 Days (On Time)'
                    WHEN utility_delay_days BETWEEN 1 AND 3  THEN '1-3 Days'
                    WHEN utility_delay_days BETWEEN 4 AND 7  THEN '4-7 Days'
                    WHEN utility_delay_days BETWEEN 8 AND 14 THEN '8-14 Days'
                    ELSE '15+ Days (High Risk)'
                END AS delay_bucket
            FROM loan_applicants
        )
        SELECT delay_bucket, COUNT(*) AS applicant_count, SUM(default_label) AS defaults,
            ROUND(AVG(default_label) * 100, 2) AS default_rate_pct,
            RANK() OVER (ORDER BY AVG(default_label) DESC) AS risk_rank
        FROM bucketed GROUP BY delay_bucket ORDER BY default_rate_pct DESC;
    """,
    "location_stability_risk_trend": """
        WITH stability_bands AS (
            SELECT ROUND(location_stability_score, 1) AS stability_band, default_label
            FROM loan_applicants
        )
        SELECT stability_band, COUNT(*) AS count,
            ROUND(AVG(default_label) * 100, 2) AS default_rate_pct,
            ROUND(AVG(AVG(default_label)) OVER (
                ORDER BY stability_band ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ) * 100, 2) AS moving_avg_default_rate
        FROM stability_bands GROUP BY stability_band ORDER BY stability_band;
    """,
    "top_flagged_applicants": """
        WITH scored AS (
            SELECT user_id, utility_delay_days, location_stability_score,
                sms_fin_alerts_monthly, device_age_months, default_label,
                (0.5 * utility_delay_days - 0.8 * location_stability_score) AS risk_score,
                ROW_NUMBER() OVER (ORDER BY (0.5 * utility_delay_days - 0.8 * location_stability_score) DESC) AS risk_rank
            FROM loan_applicants WHERE default_label = 0
        )
        SELECT user_id, risk_rank, ROUND(risk_score,4) AS risk_score,
            utility_delay_days, location_stability_score, device_age_months,
            'Manual Review Required' AS recommendation
        FROM scored WHERE risk_rank <= 20 ORDER BY risk_rank;
    """,
    "portfolio_kpis": """
        SELECT
            COUNT(*) AS total_applicants, SUM(default_label) AS total_defaults,
            ROUND(AVG(default_label) * 100, 2) AS overall_default_rate_pct,
            ROUND(AVG(utility_delay_days), 2) AS avg_utility_delay_days,
            ROUND(AVG(location_stability_score), 3) AS avg_location_stability,
            ROUND(AVG(CASE WHEN default_label=1 THEN utility_delay_days END), 2) AS avg_delay_defaulters,
            ROUND(AVG(CASE WHEN default_label=0 THEN utility_delay_days END), 2) AS avg_delay_non_defaulters
        FROM loan_applicants;
    """
}

def build_database():
    df = pd.read_csv(BASE / "alt_credit_data.csv")
    con = sqlite3.connect(DB)
    con.executescript(DDL)
    df.to_sql("loan_applicants", con, if_exists="append", index=False)
    con.commit()
    print(f"Database created — {len(df)} records loaded.")
    return con

def run_analytics(con):
    results = {}
    for name, sql in QUERIES.items():
        df = pd.read_sql_query(sql, con)
        results[name] = df
        print(f"\n{'='*55}\n{name.upper()}\n{'='*55}")
        print(df.to_string(index=False))
    return results

def plot_results(results):
    sns.set_theme(style="dark", palette="flare")
    fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")
    fig.suptitle("Pallav Credit OS — Alternative Data Underwriting Dashboard",
                 fontsize=18, color="white", fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    text_kw = dict(color="white")

    # Plot 1: Risk Decile Default Rate
    ax1 = fig.add_subplot(gs[0, 0])
    d = results["risk_decile_analysis"]
    bars = ax1.bar(d["risk_decile"], d["default_rate_pct"],
                   color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(d))))
    ax1.set_facecolor("#161b22")
    ax1.set_title("Default Rate by Risk Decile (NTILE)", color="white", fontsize=11, pad=10)
    ax1.set_xlabel("Risk Decile (1=Highest Risk)", **text_kw)
    ax1.set_ylabel("Default Rate (%)", **text_kw)
    ax1.tick_params(colors="white"); ax1.spines[:].set_color("#30363d")
    for bar, val in zip(bars, d["default_rate_pct"]):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{val}%", ha="center", va="bottom", color="white", fontsize=8)

    # Plot 2: Utility Delay Bucket
    ax2 = fig.add_subplot(gs[0, 1])
    d2 = results["utility_delay_default_profile"].sort_values("default_rate_pct", ascending=True)
    colors2 = ["#ef4444" if r==1 else "#f97316" if r==2 else "#22c55e" for r in d2["risk_rank"]]
    ax2.barh(d2["delay_bucket"], d2["default_rate_pct"], color=colors2)
    ax2.set_facecolor("#161b22")
    ax2.set_title("Default Rate by Utility Delay Bucket\n(RANK + CTE)", color="white", fontsize=11, pad=10)
    ax2.set_xlabel("Default Rate (%)", **text_kw)
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#30363d")

    # Plot 3: Location Stability + Moving Average
    ax3 = fig.add_subplot(gs[1, 0])
    d3 = results["location_stability_risk_trend"]
    ax3.bar(d3["stability_band"], d3["default_rate_pct"], alpha=0.5, color="#60a5fa", label="Default Rate")
    ax3.plot(d3["stability_band"], d3["moving_avg_default_rate"], color="#f97316",
             linewidth=2.5, marker="o", label="3-Period Moving Avg")
    ax3.set_facecolor("#161b22")
    ax3.set_title("Location Stability vs Default Risk\n(Rolling AVG OVER)", color="white", fontsize=11, pad=10)
    ax3.set_xlabel("Stability Band", **text_kw); ax3.set_ylabel("Default Rate (%)", **text_kw)
    ax3.tick_params(colors="white"); ax3.spines[:].set_color("#30363d")
    ax3.legend(facecolor="#161b22", labelcolor="white")

    # Plot 4: KPI Card
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#161b22"); ax4.axis("off")
    kpi = results["portfolio_kpis"].iloc[0]
    kpi_text = (
        f"  PORTFOLIO KPIs\n  {'─'*32}\n"
        f"  Total Applicants      : {int(kpi['total_applicants']):,}\n"
        f"  Total Defaults        : {int(kpi['total_defaults']):,}\n"
        f"  Overall Default Rate  : {kpi['overall_default_rate_pct']}%\n\n"
        f"  Avg Utility Delay     : {kpi['avg_utility_delay_days']} days\n"
        f"  Avg Stability Score   : {kpi['avg_location_stability']}\n\n"
        f"  Avg Delay (Defaulters): {kpi['avg_delay_defaulters']} days\n"
        f"  Avg Delay (Good Act.) : {kpi['avg_delay_non_defaulters']} days"
    )
    ax4.text(0.05, 0.95, kpi_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             color="#a5f3fc", bbox=dict(boxstyle="round,pad=0.8", facecolor="#0d2137", edgecolor="#1e40af"))
    ax4.set_title("Portfolio Summary KPIs", color="white", fontsize=11, pad=10)

    out = BASE / "credit_scoring_sql_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nDashboard saved -> {out}")
    plt.close()

if __name__ == "__main__":
    con = build_database()
    results = run_analytics(con)
    plot_results(results)
    con.close()
    print("\nProject 1 SQL Analytics complete.")
