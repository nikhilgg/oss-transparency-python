"""
Data Quality & Validation Checks

Runs missingness analysis, outlier detection, and correlation checks
on the built datasets. Outputs a summary report to stdout and saves
correlation matrices to outputs/tables/.

Reference: Claude.ai conversation Step 4 (Data Quality & Validation)
"""

import os
import sys
import pandas as pd
import numpy as np
from src.common import load_settings


def missingness_report(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Report missing values per column."""
    print(f"\n{'='*60}")
    print(f"MISSINGNESS ANALYSIS: {name}")
    print(f"{'='*60}")
    print(f"Total rows: {len(df)}")

    rows = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct = n_missing / len(df) * 100
        rows.append({"column": col, "missing": n_missing, "pct_missing": round(pct, 1),
                      "dtype": str(df[col].dtype)})
    report = pd.DataFrame(rows).sort_values("pct_missing", ascending=False)
    print(report[report["pct_missing"] > 0].to_string(index=False))
    if report["pct_missing"].max() == 0:
        print("  No missing values!")
    return report


def outlier_detection(df: pd.DataFrame, name: str, numeric_cols: list = None) -> pd.DataFrame:
    """Detect outliers using IQR method and flag extreme values."""
    print(f"\n{'='*60}")
    print(f"OUTLIER DETECTION: {name}")
    print(f"{'='*60}")

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_low = (s < lower).sum()
        n_high = (s > upper).sum()
        n_outliers = n_low + n_high
        rows.append({
            "column": col,
            "mean": round(s.mean(), 3),
            "median": round(s.median(), 3),
            "std": round(s.std(), 3),
            "min": round(s.min(), 3),
            "max": round(s.max(), 3),
            "q1": round(q1, 3),
            "q3": round(q3, 3),
            "n_outliers": n_outliers,
            "pct_outliers": round(n_outliers / len(s) * 100, 1),
            "skew": round(s.skew(), 3),
        })

    report = pd.DataFrame(rows)
    if len(report):
        print(report.to_string(index=False))
        skewed = report[report["skew"].abs() > 2]
        if len(skewed):
            print(f"\n  WARNING: {len(skewed)} highly skewed variables (|skew| > 2):")
            for _, r in skewed.iterrows():
                print(f"    {r['column']}: skew={r['skew']}")
    return report


def correlation_analysis(df: pd.DataFrame, name: str, key_cols: list = None,
                         outdir: str = None, fmt: str = "parquet") -> pd.DataFrame:
    """Compute correlation matrix for key variables and flag high collinearity."""
    print(f"\n{'='*60}")
    print(f"CORRELATION ANALYSIS: {name}")
    print(f"{'='*60}")

    if key_cols is None:
        key_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter to columns that exist and have variance
    valid_cols = [c for c in key_cols if c in df.columns and df[c].nunique() > 1]
    corr = df[valid_cols].corr()

    # Flag highly correlated pairs (|r| > 0.7)
    print("\nHighly correlated pairs (|r| > 0.7):")
    flagged = []
    for i in range(len(valid_cols)):
        for j in range(i + 1, len(valid_cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                flagged.append((valid_cols[i], valid_cols[j], round(r, 3)))
                print(f"  {valid_cols[i]} <-> {valid_cols[j]}: r = {r:.3f}")

    if not flagged:
        print("  None found (all |r| <= 0.7)")

    # Save correlation matrix
    if outdir:
        fname = f"{outdir}/correlation_matrix_{name.lower().replace(' ', '_')}"
        corr.to_csv(f"{fname}.csv")
        print(f"\n  Correlation matrix saved to {fname}.csv")

    return corr


def main():
    settings = load_settings()
    outdir = settings["outputs"]["outdir"]
    fmt = settings["outputs"]["format"]

    # Load datasets
    panel = pd.read_parquet(f"{outdir}/dataset_repo_month_panel.parquet") if fmt == "parquet" else pd.read_csv(f"{outdir}/dataset_repo_month_panel.csv")
    repo = pd.read_parquet(f"{outdir}/dataset_repo_level_busfactor.parquet") if fmt == "parquet" else pd.read_csv(f"{outdir}/dataset_repo_level_busfactor.csv")

    vuln_q_path = f"{outdir}/dataset_vuln_quarterly.{'parquet' if fmt == 'parquet' else 'csv'}"
    vuln_q = None
    if os.path.exists(vuln_q_path):
        vuln_q = pd.read_parquet(vuln_q_path) if fmt == "parquet" else pd.read_csv(vuln_q_path)

    # ===== MISSINGNESS =====
    missingness_report(repo, "Repo-Level Dataset")
    missingness_report(panel, "Monthly Panel Dataset")
    if vuln_q is not None:
        missingness_report(vuln_q, "Quarterly Vulnerability Dataset")

    # ===== OUTLIER DETECTION =====
    repo_numeric = ["stars", "forks", "top1_share", "gini_contrib", "bus_factor_proxy_k50",
                     "scorecard_score", "governance_index", "external_contributor_ratio",
                     "transparency_index", "contributor_count"]
    repo_numeric = [c for c in repo_numeric if c in repo.columns]
    outlier_detection(repo, "Repo-Level Dataset", repo_numeric)

    panel_numeric = ["pr_count", "review_latency_p50", "review_latency_p90",
                      "bug_issue_rate", "bug_mttr_p50_days", "many_eyes_proxy"]
    panel_numeric = [c for c in panel_numeric if c in panel.columns]
    # Use only active months for panel outlier detection
    panel_active = panel[(panel["pr_count"] > 0) | (panel["bug_issue_rate"] > 0)]
    if len(panel_active) > 0:
        outlier_detection(panel_active, "Monthly Panel (active months only)", panel_numeric)

    # ===== CORRELATION ANALYSIS =====
    # Key transparency proxies — check for collinearity
    transparency_vars = ["transparency_index", "many_eyes_proxy", "contributor_count",
                          "external_contributor_ratio", "pr_count"]
    transparency_vars = [c for c in transparency_vars if c in panel.columns]

    # Repo-level correlations
    repo_key_vars = ["stars", "forks", "top1_share", "gini_contrib", "bus_factor_proxy_k50",
                      "scorecard_score", "governance_artifact_score", "governance_index",
                      "external_contributor_ratio", "transparency_index", "contributor_count"]
    if "vuln_total" in repo.columns:
        repo_key_vars += ["vuln_total", "vuln_severe", "has_severe_vuln"]
    repo_key_vars = [c for c in repo_key_vars if c in repo.columns]

    correlation_analysis(repo, "repo_level", repo_key_vars, outdir=outdir, fmt=fmt)

    # Panel-level correlations (on active months)
    panel_key_vars = ["pr_count", "review_latency_p50", "bug_issue_rate", "bug_mttr_p50_days",
                       "scorecard_score", "governance_index", "external_contributor_ratio",
                       "transparency_index", "many_eyes_proxy", "repo_age_days"]
    panel_key_vars = [c for c in panel_key_vars if c in panel.columns]
    if len(panel_active) > 0:
        correlation_analysis(panel_active, "panel_active", panel_key_vars, outdir=outdir, fmt=fmt)

    # ===== SUMMARY =====
    print(f"\n{'='*60}")
    print("DATA QUALITY SUMMARY")
    print(f"{'='*60}")
    print(f"Repo-level: {len(repo)} repos, {len(repo.columns)} columns")
    print(f"Panel: {len(panel)} rows, {panel['repo_full_name'].nunique()} repos, {panel['month'].nunique()} months")
    if vuln_q is not None:
        print(f"Vulnerability quarterly: {len(vuln_q)} rows, {vuln_q['repo_full_name'].nunique()} repos")

    # Key warnings
    print("\nKey observations:")
    if "transparency_index" in repo.columns and "external_contributor_ratio" in repo.columns:
        r = repo[["transparency_index", "external_contributor_ratio"]].corr().iloc[0, 1]
        print(f"  transparency_index <-> external_contributor_ratio: r={r:.3f}")
    if "top1_share" in repo.columns and "gini_contrib" in repo.columns:
        r = repo[["top1_share", "gini_contrib"]].corr().iloc[0, 1]
        print(f"  top1_share <-> gini_contrib: r={r:.3f} (expected high — both measure concentration)")
    if "stars" in repo.columns and "forks" in repo.columns:
        r = repo[["stars", "forks"]].corr().iloc[0, 1]
        print(f"  stars <-> forks: r={r:.3f} (control variable collinearity check)")


if __name__ == "__main__":
    main()
