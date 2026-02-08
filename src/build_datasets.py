import os
import re
import pandas as pd
import numpy as np
from dateutil import parser
from src.common import load_settings, write_df, ensure_dir

def month_bucket(ts: str) -> str:
    dt = parser.isoparse(ts)
    return f"{dt.year:04d}-{dt.month:02d}"

def quarter_bucket(ts: str) -> str:
    dt = parser.isoparse(ts)
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year:04d}-Q{q}"

def parse_cvss_severity(cvss_str: str) -> str:
    """Classify CVSS vector string into critical/high/medium/low."""
    if not cvss_str or pd.isna(cvss_str):
        return "unknown"
    # Try to extract base score from CVSS 3.x or 4.0 vector
    # For CVSS 3.x, we approximate from the vector components
    # AV:N = network (higher), AC:L = low complexity (higher), PR:N = no privs (higher)
    s = str(cvss_str).upper()
    # Simple heuristic based on attack vector + privileges + scope
    score = 0
    if "AV:N" in s: score += 3
    elif "AV:A" in s: score += 2
    elif "AV:L" in s: score += 1
    if "AC:L" in s: score += 1.5
    if "PR:N" in s: score += 2
    elif "PR:L" in s: score += 1
    if "UI:N" in s: score += 1
    if "S:C" in s or "SC:H" in s or "SC:L" in s: score += 1
    # Confidentiality/Integrity/Availability impact
    for impact in ["C:H", "I:H", "A:H", "VC:H", "VI:H", "VA:H"]:
        if impact in s: score += 1
    if score >= 8: return "critical"
    elif score >= 6: return "high"
    elif score >= 3: return "medium"
    else: return "low"

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
    # Use all repos as the base: cross-join repo_meta with all observed months
    activity = pd.merge(latency, bug_month, on=["repo_full_name", "month"], how="outer")
    all_months = sorted(activity["month"].dropna().unique())
    if not all_months:
        all_months = [month_bucket(repo_meta["created_at"].dropna().iloc[0])] if len(repo_meta) else []

    # Create base grid: every repo x every month
    valid_repos = repo_meta[repo_meta["stars"].notna()]["repo_full_name"].unique()
    base = pd.DataFrame(
        [(r, m) for r in valid_repos for m in all_months],
        columns=["repo_full_name", "month"]
    )
    # Merge activity onto the base grid
    panel = base.merge(activity, on=["repo_full_name", "month"], how="left")
    panel = panel.merge(repo_meta[["repo_full_name","stars","forks","language","created_at"]], on="repo_full_name", how="left")
    panel = panel.merge(score[["repo_full_name","scorecard_score"]], on="repo_full_name", how="left")

    # Fill missing activity with 0 (no PRs/bugs that month = 0 count)
    panel["pr_count"] = panel["pr_count"].fillna(0).astype(int)
    panel["bug_issue_rate"] = panel["bug_issue_rate"].fillna(0).astype(int)

    # project age in days at month start (approx)
    panel["repo_created_at"] = pd.to_datetime(panel["created_at"], errors="coerce", utc=True)
    panel["month_start"] = pd.to_datetime(panel["month"] + "-01", utc=True)
    panel["repo_age_days"] = (panel["month_start"] - panel["repo_created_at"]).dt.days

    # --- External contributor ratio (per repo) ---
    ext_rows = []
    if "author_association" in prs.columns:
        internal_types = {"OWNER", "MEMBER", "COLLABORATOR"}
        for repo, g in prs.groupby("repo_full_name"):
            total_prs = len(g)
            external_prs = len(g[~g["author_association"].isin(internal_types)])
            ext_rows.append({
                "repo_full_name": repo,
                "external_contributor_ratio": external_prs / total_prs if total_prs > 0 else np.nan,
                "total_prs": total_prs,
                "external_prs": external_prs,
            })
    ext_df = pd.DataFrame(ext_rows) if ext_rows else pd.DataFrame(columns=["repo_full_name", "external_contributor_ratio"])

    # --- Governance artifact score (load if available) ---
    gov_path = f"{outdir}/governance_artifacts.{ 'parquet' if fmt=='parquet' else 'csv'}"
    if os.path.exists(gov_path):
        gov = pd.read_parquet(gov_path) if fmt == "parquet" else pd.read_csv(gov_path)
    else:
        gov = pd.DataFrame(columns=["repo_full_name", "governance_artifact_score"])

    # --- Composite governance index ---
    # 0.6 * scorecard_score (normalized to 0-1) + 0.4 * governance_artifact_score
    gov_merge = score[["repo_full_name", "scorecard_score"]].merge(
        gov[["repo_full_name", "governance_artifact_score"]], on="repo_full_name", how="outer"
    )
    gov_merge["scorecard_norm"] = gov_merge["scorecard_score"].fillna(0) / 10.0  # scorecard is 0-10
    gov_merge["governance_index"] = (
        0.6 * gov_merge["scorecard_norm"] +
        0.4 * gov_merge["governance_artifact_score"].fillna(0)
    )

    # --- Enrich panel ---
    panel = panel.merge(gov_merge[["repo_full_name", "governance_index"]], on="repo_full_name", how="left")
    panel = panel.merge(ext_df[["repo_full_name", "external_contributor_ratio"]], on="repo_full_name", how="left")

    # --- Composite transparency index (normalized) ---
    # Compute per-repo totals for normalization
    repo_pr_total = prs.groupby("repo_full_name")["pr_number"].count().rename("total_prs_repo")
    repo_contrib_count = contrib.groupby("repo_full_name")["contributor_login"].nunique().rename("contributor_count")
    transparency_df = pd.DataFrame({"repo_full_name": valid_repos}).merge(
        repo_pr_total, on="repo_full_name", how="left"
    ).merge(
        repo_contrib_count, on="repo_full_name", how="left"
    ).merge(
        ext_df[["repo_full_name", "external_contributor_ratio"]], on="repo_full_name", how="left"
    )
    transparency_df["total_prs_repo"] = transparency_df["total_prs_repo"].fillna(0)
    transparency_df["contributor_count"] = transparency_df["contributor_count"].fillna(0)
    transparency_df["external_contributor_ratio"] = transparency_df["external_contributor_ratio"].fillna(0)

    # Min-max normalize each component to [0,1] then average
    for col in ["total_prs_repo", "contributor_count", "external_contributor_ratio"]:
        cmin, cmax = transparency_df[col].min(), transparency_df[col].max()
        if cmax > cmin:
            transparency_df[f"{col}_norm"] = (transparency_df[col] - cmin) / (cmax - cmin)
        else:
            transparency_df[f"{col}_norm"] = 0.0

    transparency_df["transparency_index"] = (
        transparency_df["total_prs_repo_norm"] +
        transparency_df["contributor_count_norm"] +
        transparency_df["external_contributor_ratio_norm"]
    ) / 3.0

    # many_eyes_proxy: monthly PR count (for panel); transparency_index is repo-level
    panel["many_eyes_proxy"] = panel["pr_count"].fillna(0)
    panel = panel.merge(transparency_df[["repo_full_name", "transparency_index", "contributor_count"]],
                        on="repo_full_name", how="left")

    # Interaction terms
    panel["many_eyes_x_governance"] = panel["many_eyes_proxy"] * panel["governance_index"].fillna(0)
    panel["transparency_x_governance"] = panel["transparency_index"].fillna(0) * panel["governance_index"].fillna(0)

    # --- Load OSV and link to repos via pypi_repo_master ---
    osv_path = f"{outdir}/osv_vulns_raw.{ 'parquet' if fmt=='parquet' else 'csv'}"
    pypi_path = f"{outdir}/pypi_repo_master.{ 'parquet' if fmt=='parquet' else 'csv'}"
    vuln_quarter = pd.DataFrame()
    vuln_repo_level = pd.DataFrame()

    if os.path.exists(osv_path) and os.path.exists(pypi_path):
        osv = pd.read_parquet(osv_path) if fmt == "parquet" else pd.read_csv(osv_path)
        pypi = pd.read_parquet(pypi_path) if fmt == "parquet" else pd.read_csv(pypi_path)

        # Map package_name -> repo_full_name
        pypi["repo_full_name"] = pypi["github_url"].str.replace("https://github.com/", "", regex=False)
        pkg_to_repo = pypi[["package_name", "repo_full_name"]].drop_duplicates("package_name")
        osv = osv.merge(pkg_to_repo, on="package_name", how="left")
        osv = osv.dropna(subset=["repo_full_name"])

        # Classify severity
        osv["severity_class"] = osv["severity_raw"].apply(parse_cvss_severity)

        # Quarterly vulnerability counts per repo
        osv["quarter"] = osv["published"].apply(lambda x: quarter_bucket(x) if pd.notna(x) else None)
        osv = osv.dropna(subset=["quarter"])

        vuln_quarter = osv.groupby(["repo_full_name", "quarter", "severity_class"]).size().reset_index(name="vuln_count")
        # Pivot: one row per repo-quarter, columns for each severity level
        vuln_quarter_wide = vuln_quarter.pivot_table(
            index=["repo_full_name", "quarter"],
            columns="severity_class",
            values="vuln_count",
            fill_value=0
        ).reset_index()
        vuln_quarter_wide.columns.name = None
        for sev in ["critical", "high", "medium", "low", "unknown"]:
            if sev not in vuln_quarter_wide.columns:
                vuln_quarter_wide[sev] = 0
        vuln_quarter_wide["vuln_total"] = (
            vuln_quarter_wide["critical"] + vuln_quarter_wide["high"] +
            vuln_quarter_wide["medium"] + vuln_quarter_wide["low"] +
            vuln_quarter_wide["unknown"]
        )
        vuln_quarter_wide["vuln_severe"] = vuln_quarter_wide["critical"] + vuln_quarter_wide["high"]

        write_df(vuln_quarter_wide, f"{outdir}/dataset_vuln_quarterly.{ 'csv' if fmt=='csv' else 'parquet'}", fmt=fmt)
        print(f"  Quarterly vulnerability dataset: {len(vuln_quarter_wide)} rows")

        # Repo-level vulnerability summary
        vuln_repo_level = osv.groupby("repo_full_name").agg(
            vuln_total=("osv_id", "count"),
            vuln_critical=("severity_class", lambda x: (x == "critical").sum()),
            vuln_high=("severity_class", lambda x: (x == "high").sum()),
            vuln_medium=("severity_class", lambda x: (x == "medium").sum()),
            vuln_low=("severity_class", lambda x: (x == "low").sum()),
        ).reset_index()
        vuln_repo_level["vuln_severe"] = vuln_repo_level["vuln_critical"] + vuln_repo_level["vuln_high"]
        vuln_repo_level["has_severe_vuln"] = (vuln_repo_level["vuln_severe"] > 0).astype(int)

    # --- Write panel dataset ---
    write_df(panel, f"{outdir}/dataset_repo_month_panel.{ 'csv' if fmt=='csv' else 'parquet'}", fmt=fmt)

    # --- Dataset 4: repo-level bus factor + vulnerability risk ---
    repo_level = repo_meta[["repo_full_name","stars","forks","language","created_at"]].merge(bus, on="repo_full_name", how="left")
    repo_level = repo_level.merge(score[["repo_full_name","scorecard_score"]], on="repo_full_name", how="left")
    repo_level = repo_level.merge(gov[["repo_full_name", "governance_artifact_score"]], on="repo_full_name", how="left")
    repo_level = repo_level.merge(gov_merge[["repo_full_name", "governance_index"]], on="repo_full_name", how="left")
    repo_level = repo_level.merge(ext_df[["repo_full_name", "external_contributor_ratio"]], on="repo_full_name", how="left")
    repo_level = repo_level.merge(transparency_df[["repo_full_name", "transparency_index", "contributor_count"]],
                                   on="repo_full_name", how="left")

    # Merge vulnerability counts onto repo-level
    if len(vuln_repo_level) > 0:
        repo_level = repo_level.merge(vuln_repo_level, on="repo_full_name", how="left")
        for col in ["vuln_total", "vuln_critical", "vuln_high", "vuln_medium", "vuln_low", "vuln_severe", "has_severe_vuln"]:
            repo_level[col] = repo_level[col].fillna(0).astype(int)

    # Log-transformed stars for controls
    repo_level["log_stars"] = np.log1p(repo_level["stars"].fillna(0))

    write_df(repo_level, f"{outdir}/dataset_repo_level_busfactor.{ 'csv' if fmt=='csv' else 'parquet'}", fmt=fmt)

    print("Datasets built: dataset_repo_month_panel, dataset_repo_level_busfactor")

if __name__ == "__main__":
    main()
