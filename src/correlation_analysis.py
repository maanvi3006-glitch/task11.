"""
correlation_analysis.py
-------------------------
The analytical core of Task 11. Implements the build pipeline from the
task brief, step by step:

  1. Compute a correlation matrix for the key numeric variables.
  2. Render it as a heatmap, ordered to reveal clusters.
  3. Flag the strongest relationships and interrogate each.
  4. Distinguish plausible drivers from likely coincidences.
  5. Note multicollinearity that would hurt modelling.
  6. Summarise the few relationships worth acting on.

Design choices (see README for the full "alternative approaches" writeup):
  - Both Pearson (linear) AND Spearman (monotonic/rank) matrices are
    computed, directly addressing the brief's listed alternative
    ("Pearson vs Spearman") and the pitfall "Ignoring non-linear
    relationships Pearson misses" -- we flag pairs where the two
    coefficients disagree by a wide margin as likely non-linear.
  - Heatmap ordering uses hierarchical clustering (average linkage on
    1 - |correlation| distance) so related variables sit next to each
    other -- this is the "ordered to reveal clusters" requirement.
  - Multicollinearity is measured properly with Variance Inflation
    Factor (VIF), not just "any two vars with |r| > 0.8", because VIF
    also catches multivariable collinearity a pairwise view can miss.
"""

from __future__ import annotations

import dataclasses
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from statsmodels.stats.outliers_influence import variance_inflation_factor

STRONG_THRESHOLD = 0.60          # |r| above this = "strong", flagged for interpretation
VERY_STRONG_THRESHOLD = 0.85     # |r| above this = candidate multicollinearity pair
PEARSON_SPEARMAN_GAP = 0.15      # divergence flagged as "possibly non-linear"
VIF_CONCERN = 5.0                # standard rule-of-thumb VIF cutoff


@dataclasses.dataclass
class CorrelationPair:
    var_a: str
    var_b: str
    pearson_r: float
    spearman_r: float

    @property
    def abs_pearson(self) -> float:
        return abs(self.pearson_r)

    @property
    def likely_nonlinear(self) -> bool:
        return abs(self.pearson_r - self.spearman_r) >= PEARSON_SPEARMAN_GAP

    @property
    def direction(self) -> str:
        return "positive" if self.pearson_r > 0 else "negative"


def compute_matrices(numeric_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Step 1 + Pearson/Spearman alternative: return (pearson_matrix, spearman_matrix)."""
    pearson = numeric_df.corr(method="pearson")
    spearman = numeric_df.corr(method="spearman")
    return pearson, spearman


def cluster_order(pearson_matrix: pd.DataFrame) -> List[str]:
    """
    Step 2: hierarchical-clustering order so the heatmap groups related
    variables into visible blocks, instead of an arbitrary column order.
    """
    dist = (1 - pearson_matrix.abs()).to_numpy(copy=True)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)
    return [pearson_matrix.columns[i] for i in order]


def strongest_pairs(
    pearson_matrix: pd.DataFrame,
    spearman_matrix: pd.DataFrame,
    threshold: float = STRONG_THRESHOLD,
) -> List[CorrelationPair]:
    """Step 3: flag the strongest pairwise relationships for interrogation."""
    cols = pearson_matrix.columns
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r_p = pearson_matrix.loc[a, b]
            r_s = spearman_matrix.loc[a, b]
            if abs(r_p) >= threshold:
                pairs.append(CorrelationPair(a, b, float(r_p), float(r_s)))
    pairs.sort(key=lambda p: p.abs_pearson, reverse=True)
    return pairs


# Variable pairs we treat as having a believable, mechanistic causal story
# in this business context (used only to help auto-draft the "plausible
# driver vs coincidence" narrative -- analysts should still sanity-check
# this, per the brief's core pitfall about causation).
PLAUSIBLE_MECHANISMS = {
    frozenset({"marketing_spend", "website_traffic"}): (
        "Marketing spend buys ads/campaigns that directly drive site visits — "
        "a plausible, mechanistic driver relationship."
    ),
    frozenset({"website_traffic", "ad_clicks"}): (
        "Ad clicks are largely a subset of total traffic-generating activity, "
        "so this pair is expected to move together almost by construction — "
        "a strong candidate for multicollinearity rather than two independent signals."
    ),
    frozenset({"website_traffic", "revenue"}): (
        "More visitors, at a roughly stable conversion rate, mechanically "
        "produces more revenue — plausible driver relationship."
    ),
    frozenset({"marketing_spend", "revenue"}): (
        "Indirect but plausible: marketing spend drives traffic, which drives "
        "revenue. Likely to be partly explained by traffic rather than a "
        "wholly independent effect."
    ),
    frozenset({"discount_rate", "returns_pct"}): (
        "Deeper discounts are associated with more impulse/low-commitment "
        "purchases, which are more often returned — plausible driver."
    ),
    frozenset({"customer_satisfaction", "support_tickets"}): (
        "More support tickets typically reflect friction/problems, which "
        "plausibly depresses satisfaction — plausible driver (direction of "
        "causality could run either way)."
    ),
    frozenset({"customer_satisfaction", "churn_rate"}): (
        "Lower satisfaction plausibly drives customers to leave — "
        "plausible driver relationship."
    ),
    frozenset({"marketing_spend", "conversion_rate"}): (
        "Negative relationship: as spend scales up, campaigns likely reach "
        "further into lower-intent audiences, diluting conversion quality — "
        "a plausible 'diminishing returns' story, but worth confirming against "
        "campaign-level targeting data before treating spend cuts as a fix."
    ),
    frozenset({"revenue", "conversion_rate"}): (
        "Revenue is mechanically a function of traffic and conversion rate, "
        "so this positive link is expected by construction rather than a "
        "separate discovery — flagged here mainly as a modelling reminder, "
        "not a new lever."
    ),
    frozenset({"marketing_spend", "ad_clicks"}): (
        "Marketing spend plausibly buys some of these clicks directly, but "
        "this pair overlaps heavily with marketing_spend↔website_traffic and "
        "website_traffic↔ad_clicks — likely a secondary effect of the same "
        "underlying chain rather than an independent relationship."
    ),
}

SPURIOUS_CANDIDATES = {
    frozenset({"avg_temperature_c", "revenue"}),
    frozenset({"avg_temperature_c", "marketing_spend"}),
    frozenset({"avg_temperature_c", "website_traffic"}),
}


def classify_pair(pair: CorrelationPair) -> str:
    """Step 4: label a strong pair as plausible driver, likely spurious, or unclear."""
    key = frozenset({pair.var_a, pair.var_b})
    if key in SPURIOUS_CANDIDATES:
        return "likely_spurious"
    if key in PLAUSIBLE_MECHANISMS:
        return "plausible_driver"
    return "worth_investigating"


def explain_pair(pair: CorrelationPair) -> str:
    key = frozenset({pair.var_a, pair.var_b})
    if key in PLAUSIBLE_MECHANISMS:
        return PLAUSIBLE_MECHANISMS[key]
    if key in SPURIOUS_CANDIDATES:
        return (
            "No direct mechanism links these two — the shared movement is best "
            "explained by both variables riding the same underlying seasonal "
            "cycle (a classic spurious/confounded correlation, not a driver)."
        )
    return (
        "No pre-registered mechanism for this pair — flagged for manual "
        "domain review before treating it as a driver."
    )


def compute_vif(numeric_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5: proper multicollinearity check via Variance Inflation Factor.
    VIF > 5 (some use 10) signals a variable is well-predicted by the
    others and would distort coefficients in a downstream regression model.
    """
    X = numeric_df.dropna().copy()
    # VIF is undefined/unstable on zero-variance columns; guard against that.
    X = X.loc[:, X.std(ddof=0) > 1e-9]
    # statsmodels' VIF assumes an intercept is present in the design matrix;
    # without one, VIF is wildly inflated for every column (a well-known
    # gotcha). We add a constant, compute VIF for the real columns only,
    # then drop the constant's own row.
    X_with_const = X.copy()
    X_with_const.insert(0, "_const", 1.0)
    vif_data = []
    for i, col in enumerate(X_with_const.columns):
        if col == "_const":
            continue
        try:
            vif = variance_inflation_factor(X_with_const.values, i)
        except Exception:
            vif = np.nan
        vif_data.append({"variable": col, "VIF": vif})
    out = pd.DataFrame(vif_data).sort_values("VIF", ascending=False).reset_index(drop=True)
    out["concern"] = out["VIF"] > VIF_CONCERN
    return out


def build_summary(
    pairs: List[CorrelationPair],
    vif_table: pd.DataFrame,
    top_n: int = 6,
) -> str:
    """Step 6: the few relationships worth acting on, in plain language."""
    lines = ["### Key relationships worth acting on\n"]
    plausible = [p for p in pairs if classify_pair(p) == "plausible_driver"]
    spurious = [p for p in pairs if classify_pair(p) == "likely_spurious"]
    other = [p for p in pairs if classify_pair(p) not in ("plausible_driver", "likely_spurious")]

    if plausible:
        lines.append("**Plausible drivers to act on:**")
        for p in plausible[:top_n]:
            lines.append(
                f"- `{p.var_a}` ↔ `{p.var_b}`: r = {p.pearson_r:+.2f} ({p.direction}). "
                f"{explain_pair(p)}"
            )
        lines.append("")

    if spurious:
        lines.append("**Likely spurious — do not act on directly:**")
        for p in spurious[:top_n]:
            lines.append(
                f"- `{p.var_a}` ↔ `{p.var_b}`: r = {p.pearson_r:+.2f}. {explain_pair(p)}"
            )
        lines.append("")

    if other:
        lines.append("**Strong but unexplained — needs domain review:**")
        for p in other[:top_n]:
            lines.append(f"- `{p.var_a}` ↔ `{p.var_b}`: r = {p.pearson_r:+.2f}.")
        lines.append("")

    concerning_vif = vif_table[vif_table["concern"]]
    if not concerning_vif.empty:
        lines.append("**Multicollinearity to prune before modelling (VIF > 5):**")
        for _, row in concerning_vif.iterrows():
            lines.append(f"- `{row['variable']}`: VIF = {row['VIF']:.1f}")
    else:
        lines.append("**Multicollinearity:** no variable exceeds the VIF > 5 concern threshold.")

    nonlinear = [p for p in pairs if p.likely_nonlinear]
    if nonlinear:
        lines.append("\n**Possible non-linear relationships** (Pearson vs Spearman disagree "
                      "by ≥ 0.15 — a straight-line Pearson read alone would understate these):")
        for p in nonlinear[:top_n]:
            lines.append(
                f"- `{p.var_a}` ↔ `{p.var_b}`: Pearson {p.pearson_r:+.2f} vs "
                f"Spearman {p.spearman_r:+.2f}"
            )

    return "\n".join(lines)
