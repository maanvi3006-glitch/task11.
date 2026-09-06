"""
run_analysis.py
------------------
Standalone, non-interactive run of the full Task 11 pipeline. Produces
static evidence files in outputs/ so the deliverable can be marked even
without launching the Streamlit dashboard:

  outputs/correlation_heatmap.png     - the core visual deliverable
  outputs/pearson_matrix.csv          - raw Pearson correlation matrix
  outputs/spearman_matrix.csv         - raw Spearman correlation matrix
  outputs/vif_table.csv               - multicollinearity check
  outputs/interpretation_report.md    - the written interpretation
  outputs/data_cleaning_report.md     - validation/cleaning evidence

Usage:
    python run_analysis.py [path/to/data.csv]
    (defaults to data/altrodav_business_metrics.csv)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_loader import load_and_validate, DataValidationError, EXPECTED_NUMERIC_COLUMNS
from correlation_analysis import (
    compute_matrices,
    cluster_order,
    strongest_pairs,
    compute_vif,
    build_summary,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"


def main():
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "altrodav_business_metrics.csv"
    OUT_DIR.mkdir(exist_ok=True)

    print(f"Loading and validating: {data_path}")
    try:
        df, report = load_and_validate(data_path)
    except DataValidationError as e:
        print(f"\n❌ Data validation failed: {e}")
        sys.exit(1)

    print(report.as_markdown())
    (OUT_DIR / "data_cleaning_report.md").write_text(report.as_markdown())

    numeric_cols = [c for c in EXPECTED_NUMERIC_COLUMNS if c in df.columns]
    numeric_df = df[numeric_cols].select_dtypes(include="number")

    print(f"\nComputing correlation matrices over {numeric_df.shape[1]} numeric variables...")
    pearson, spearman = compute_matrices(numeric_df)
    pearson.to_csv(OUT_DIR / "pearson_matrix.csv")
    spearman.to_csv(OUT_DIR / "spearman_matrix.csv")

    order = cluster_order(pearson)
    ordered = pearson.loc[order, order]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        ordered, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.5, cbar_kws={"label": "Pearson r"}, ax=ax,
    )
    ax.set_title("Altrodav Technologies — Business Metrics Correlation Heatmap\n"
                  "(hierarchically clustered, Pearson r)", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "correlation_heatmap.png", dpi=200)
    print(f"Saved heatmap -> {OUT_DIR / 'correlation_heatmap.png'}")

    pairs = strongest_pairs(pearson, spearman)
    vif_table = compute_vif(numeric_df)
    vif_table.to_csv(OUT_DIR / "vif_table.csv", index=False)

    summary_md = build_summary(pairs, vif_table)
    report_text = (
        "# Task 11 — Correlation & Heatmaps: Interpretation Report\n\n"
        f"Data source: `{data_path.name}` · {report.rows_out} clean rows · "
        f"{numeric_df.shape[1]} numeric variables\n\n"
        "![heatmap](correlation_heatmap.png)\n\n"
        + summary_md
    )
    (OUT_DIR / "interpretation_report.md").write_text(report_text)
    print(f"Saved interpretation -> {OUT_DIR / 'interpretation_report.md'}")

    print("\n✅ Done. All evidence files are in outputs/.")


if __name__ == "__main__":
    main()
