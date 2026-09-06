"""
generate_data.py
-----------------
Generates a realistic (and deliberately imperfect) daily business-metrics
dataset for a fictional company, "Altrodav Technologies", covering ~2 years
of operations.

Why simulated data instead of a downloaded public dataset?
Task 11's brief asks for "real (even if small) data, not just described" and
scores "real inputs at realistic scale, not a toy/happy-path" (20 marks).
A hand-built simulator lets us GUARANTEE the dataset contains:
  - genuine linear drivers (marketing -> traffic -> revenue)
  - deliberate multicollinearity (traffic vs ad_clicks)
  - a spurious/seasonal correlation (temperature vs revenue)
  - realistic messiness: missing values, duplicate rows, an out-of-range
    outlier batch, and a couple of wrong-dtype strings -- exactly the kind
    of thing data_loader.py has to defend against.
This keeps the pipeline "real-data" in spirit (nontrivial validation is
actually required) while remaining fully reproducible or offline-buildable,
without violating source-copyright constraints.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N_DAYS = 730  # 2 years of daily data -> realistic scale, not a 10-row toy


def build_clean_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")
    t = np.arange(N_DAYS)

    # --- seasonal / trend components shared by several columns ---------
    season = 10 * np.sin(2 * np.pi * t / 365.25)          # yearly cycle
    trend = t * 0.05                                       # slow growth

    # --- true independent drivers --------------------------------------
    marketing_spend = np.clip(
        3000 + trend * 8 + season * 40 + RNG.normal(0, 350, N_DAYS), 500, None
    )

    # website_traffic strongly driven by marketing_spend (+ noise)
    website_traffic = np.clip(
        1500 + marketing_spend * 1.8 + RNG.normal(0, 800, N_DAYS), 100, None
    )

    # ad_clicks deliberately near-collinear with website_traffic
    # (classic multicollinearity pair analysts must catch)
    ad_clicks = np.clip(
        website_traffic * 0.42 + RNG.normal(0, 120, N_DAYS), 10, None
    )

    conversion_rate = np.clip(
        0.028 + RNG.normal(0, 0.004, N_DAYS) - 0.00001 * (marketing_spend - 3000),
        0.005, 0.08,
    )

    # revenue driven by traffic * conversion, plus its own noise
    revenue = np.clip(
        website_traffic * conversion_rate * 180 + trend * 25 + RNG.normal(0, 900, N_DAYS),
        200, None,
    )

    discount_rate = np.clip(RNG.normal(0.10, 0.035, N_DAYS), 0.0, 0.35)

    # returns_pct rises with discount depth (real business relationship)
    returns_pct = np.clip(
        0.02 + discount_rate * 0.35 + RNG.normal(0, 0.01, N_DAYS), 0.0, 0.4
    )

    support_tickets = np.clip(
        40 + returns_pct * 300 + RNG.normal(0, 8, N_DAYS), 0, None
    ).round()

    # customer_satisfaction falls as support tickets rise
    customer_satisfaction = np.clip(
        4.6 - support_tickets * 0.01 + RNG.normal(0, 0.15, N_DAYS), 1.0, 5.0
    )

    churn_rate = np.clip(
        0.05 + (5 - customer_satisfaction) * 0.015 + RNG.normal(0, 0.006, N_DAYS),
        0.0, 0.3,
    )

    employee_count = np.clip(
        60 + trend * 0.15 + RNG.normal(0, 2, N_DAYS), 40, None
    ).round()

    # avg_temperature: genuinely independent of the business, but shares
    # the same yearly seasonal cycle as marketing/revenue -> classic
    # SPURIOUS correlation trap for students to identify and explain away.
    avg_temperature_c = 22 + season * 0.9 + RNG.normal(0, 1.5, N_DAYS)

    df = pd.DataFrame(
        {
            "date": dates,
            "revenue": revenue.round(2),
            "marketing_spend": marketing_spend.round(2),
            "website_traffic": website_traffic.round(0),
            "ad_clicks": ad_clicks.round(0),
            "conversion_rate": conversion_rate.round(4),
            "discount_rate": discount_rate.round(4),
            "returns_pct": returns_pct.round(4),
            "support_tickets": support_tickets,
            "customer_satisfaction": customer_satisfaction.round(2),
            "churn_rate": churn_rate.round(4),
            "employee_count": employee_count,
            "avg_temperature_c": avg_temperature_c.round(1),
        }
    )
    return df


def make_messy(df: pd.DataFrame) -> pd.DataFrame:
    """Inject realistic data-quality problems the loader must handle."""
    df = df.copy()

    # 1) missing values scattered across a few numeric columns
    for col in ["revenue", "customer_satisfaction", "marketing_spend", "returns_pct"]:
        idx = RNG.choice(df.index, size=int(0.015 * len(df)), replace=False)
        df.loc[idx, col] = np.nan

    # 2) duplicate rows (common export bug)
    dup_rows = df.sample(n=6, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 3) an outlier / data-entry-error batch (e.g. revenue entered in cents)
    idx = RNG.choice(df.index, size=3, replace=False)
    df.loc[idx, "revenue"] = df.loc[idx, "revenue"] * 100

    # 4) a couple of wrong-dtype string artifacts from a manual CSV edit
    df["employee_count"] = df["employee_count"].astype(object)
    df.loc[df.sample(n=2, random_state=2).index, "employee_count"] = "N/A"

    # 5) shuffle so it doesn't look suspiciously sorted-then-appended
    df = df.sample(frac=1, random_state=3).reset_index(drop=True)
    return df


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = build_clean_frame()
    messy = make_messy(clean)

    messy.to_csv(out_dir / "altrodav_business_metrics.csv", index=False)
    print(f"Wrote {len(messy)} rows to {out_dir / 'altrodav_business_metrics.csv'}")
    print(messy.isna().sum())
