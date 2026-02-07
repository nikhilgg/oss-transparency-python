import os
import pandas as pd
import numpy as np
from dateutil import parser
from src.common import load_settings, write_df

def month_bucket(ts: str) -> str:
    dt = parser.isoparse(ts)
    return f"{dt.year:04d}-{dt.month:02d}"

def gini(x: np.ndarray) -> float:
    x = x[x >= 0]
    if len(x) == 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    if cumx[-1] == 0:
        return 0.0
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def main():
    settings = load_settings()
    outdir = settings["outputs"]["outdir"]
    fmt = settings["outputs"]["format"]

    # Load extracted tables
    repo_meta = pd.read_parquet(f"{outdir}/github_repo_meta.parquet") if fmt=="parquet" else pd.read_csv(f"{outdir}/github_repo_meta.csv")
    prs = pd.read_parquet(f"{outdir}/github_prs.parquet") if fmt=="parquet" else pd.read_csv(f"{outdir}/github_prs.csv")
    bugs = pd.read_parquet(f"{outdir}/github_bug_issues.parquet") if fmt=="parquet" else pd.read_csv(f"{outdir}/github_bug_issues.csv")
    contrib = pd.read_parquet(f"{outdir}/github_contributors.parquet") if fmt=="parquet" else pd.read_csv(f"{outdir}/github_contributors.csv")

    # Scorecard optional
    score_path = f"{outdir}/scorecard_results.{ 'parquet' if fmt=='parquet' else 'csv'}"
    if os.path.exists(score_path):
        score = pd.read_parquet(score_path) if fmt=="parquet" else pd.read_csv(score_path)
    else:
        score = pd.DataFrame(columns=["repo_full_name", "scorecard_score"])

    # --- Review latency monthly ---
    prs = prs.dropna(subset=["pr_created_at"]).copy()
    prs["month"] = prs["pr_created_at"].apply(month_bucket)
    latency = (prs.groupby(["repo_full_name", "month"])["latency_first_review_hours"]
                 .agg(review_latency_p50="median", review_latency_p90=lambda s: s.quantile(0.90),
                      pr_count="count")
                 .reset_index())

    # --- Bug issues monthly ---
    bugs = bugs.dropna(subset=["created_at"]).copy()
    bugs["month"] = bugs["created_at"].apply(month_bucket)
    bug_month = (bugs.groupby(["repo_full_name", "month"])
                   .agg(bug_issue_rate=("issue_number", "count"),
                        bug_mttr_p50_days=("mttr_days", "median"))
                   .reset_index())

    # --- Bus factor proxy ---
    contrib = contrib.dropna(subset=["contributions"]).copy()
    bf_rows = []
    for repo, g in contrib.groupby("repo_full_name"):
        x = g["contributions"].astype(float).values
        x_sorted = np.sort(x)[::-1]
        top1 = x_sorted[0] if len(x_sorted) else 0
        total = x_sorted.sum() if len(x_sorted) else 0
        top1_share = (top1 / total) if total > 0 else np.nan
        # bus factor proxy: smallest k s.t. top k contribute >= 50% of commits
        cum = np.cumsum(x_sorted)
        k50 = int(np.argmax(cum >= 0.5 * total) + 1) if total > 0 else np.nan
        bf_rows.append({
            "repo_full_name": repo,
            "top1_share": top1_share,
            "gini_contrib": gini(x),
            "bus_factor_proxy_k50": k50
        })
    bus = pd.DataFrame(bf_rows)

    # --- Build repo_month panel (Dataset 1 & 5 base) ---
    panel = pd.merge(latency, bug_month, on=["repo_full_name", "month"], how="outer")
    panel = pd.merge(panel, repo_meta[["repo_full_name","stars","forks","language","created_at"]], on="repo_full_name", how="left")
    panel = pd.merge(panel, score[["repo_full_name","scorecard_score"]], on="repo_full_name", how="left")

    # project age in days at month start (approx)
    panel["repo_created_at"] = pd.to_datetime(panel["created_at"], errors="coerce", utc=True)
    panel["month_start"] = pd.to_datetime(panel["month"] + "-01", utc=True)
    panel["repo_age_days"] = (panel["month_start"] - panel["repo_created_at"]).dt.days

    # governance index: start simple (scorecard_score); can enrich later with community profile
    panel["governance_index"] = panel["scorecard_score"]

    # transparency score: start simple; can improve later with contributor/PR diversity measures
    # Here: many eyes proxy = pr_count (monthly) (normalize later)
    panel["many_eyes_proxy"] = panel["pr_count"].fillna(0)

    # Interaction
    panel["many_eyes_x_governance"] = panel["many_eyes_proxy"] * panel["governance_index"].fillna(0)

    # Write datasets
    write_df(panel, f"{outdir}/dataset_repo_month_panel.{ 'csv' if fmt=='csv' else 'parquet'}", fmt=fmt)

    # Dataset 4: repo-level bus factor + risk target placeholder (built after OSV processing)
    repo_level = repo_meta[["repo_full_name","stars","forks","language","created_at"]].merge(bus, on="repo_full_name", how="left")
    repo_level = repo_level.merge(score[["repo_full_name","scorecard_score"]], on="repo_full_name", how="left")
    write_df(repo_level, f"{outdir}/dataset_repo_level_busfactor.{ 'csv' if fmt=='csv' else 'parquet'}", fmt=fmt)

    print("Datasets built: dataset_repo_month_panel, dataset_repo_level_busfactor")

if __name__ == "__main__":
    main()
