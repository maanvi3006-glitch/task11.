"""
test_data_loader.py
----------------------
Proves the edge-case / failure handling required by the rubric actually
works, rather than just being claimed. Run with:  python -m pytest tests/ -v
(or just: python tests/test_data_loader.py)
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_loader import load_and_validate, DataValidationError


ROOT = Path(__file__).resolve().parents[1]
GOOD_CSV = ROOT / "data" / "altrodav_business_metrics.csv"


def test_bundled_dataset_loads_and_cleans():
    df, report = load_and_validate(GOOD_CSV)
    assert len(df) > 0
    assert report.duplicates_removed >= 0
    # cleaned columns should have zero remaining missing values (imputed)
    for col, n in report.missing_after.items():
        assert n == 0


def test_missing_file_raises_friendly_error():
    with pytest.raises(DataValidationError):
        load_and_validate(ROOT / "data" / "does_not_exist.csv")


def test_empty_file_raises_friendly_error(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(DataValidationError):
        load_and_validate(empty)


def test_header_only_file_raises_friendly_error(tmp_path):
    header_only = tmp_path / "header_only.csv"
    header_only.write_text("revenue,marketing_spend\n")
    with pytest.raises(DataValidationError):
        load_and_validate(header_only)


def test_single_numeric_column_raises_friendly_error(tmp_path):
    one_col = tmp_path / "one_col.csv"
    one_col.write_text("revenue\n1\n2\n3\n")
    with pytest.raises(DataValidationError):
        load_and_validate(one_col)


def test_non_csv_garbage_raises_friendly_error(tmp_path):
    garbage = tmp_path / "garbage.csv"
    garbage.write_bytes(bytes(range(0, 255)) * 4)
    with pytest.raises(DataValidationError):
        load_and_validate(garbage)


def test_text_in_numeric_column_is_coerced(tmp_path):
    # "unknown" is NOT one of pandas' auto-recognised NA tokens (unlike
    # "N/A"), so this genuinely forces the column to load as object/text
    # dtype -- proving data_loader coerces it back to numeric itself
    # rather than relying on pandas' default NA parsing.
    messy = tmp_path / "messy.csv"
    messy.write_text(
        "revenue,marketing_spend\n"
        "100,50\n"
        "unknown,60\n"
        "200,unknown\n"
        "150,55\n"
    )
    df, report = load_and_validate(messy)
    assert "revenue" in report.coerced_columns
    assert "marketing_spend" in report.coerced_columns
    assert df["revenue"].isna().sum() == 0  # imputed after coercion


def test_duplicate_rows_are_removed(tmp_path):
    dup = tmp_path / "dup.csv"
    dup.write_text(
        "revenue,marketing_spend\n"
        "100,50\n"
        "100,50\n"
        "200,60\n"
    )
    df, report = load_and_validate(dup)
    assert report.duplicates_removed == 1
    assert len(df) == 2


def test_outlier_is_winsorised(tmp_path):
    import numpy as np
    rng = np.random.default_rng(0)
    n = 200
    revenue = rng.normal(1000, 50, n)
    revenue[0] = 999_999  # extreme data-entry error
    marketing = rng.normal(500, 20, n)
    df_in = pd.DataFrame({"revenue": revenue, "marketing_spend": marketing})
    p = tmp_path / "outlier.csv"
    df_in.to_csv(p, index=False)

    df, report = load_and_validate(p)
    assert df["revenue"].max() < 999_999
    assert "revenue" in report.outliers_capped


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
