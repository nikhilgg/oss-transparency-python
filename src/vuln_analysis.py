"""
Vulnerability Analysis Data Builder

Re-queries OSV to extract fix versions from affected.ranges.events,
then queries PyPI release history to get fix release dates.
Produces two derived datasets for Hypothesis (ii):

  (ii-a) dataset_vuln_discovery_rate: repo-quarter vulnerability discovery counts
         with transparency and governance metrics joined.
  (ii-b) dataset_vuln_survival: event-level vulnerability records with
         time-to-fix, censoring flag, and severity for survival analysis.
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dateutil import parser as dtparser
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Tuple, List

from src.common import load_settings, http_get, http_post, write_df


# ---------- OSV: extract fix version per vulnerability ----------

def osv_query_with_affected(package_name: str) -> List[Dict[str, Any]]:
    """Query OSV and extract fix version + introduced version from affected ranges."""
    url = "https://api.osv.dev/v1/query"
    body = {"package": {"name": package_name, "ecosystem": "PyPI"}}
    try:
        resp = http_post(url, json_body=body)
        data = resp.json()
    except Exception:
        return []

    rows = []
    for v in data.get("vulns") or []:
        osv_id = v.get("id")
        published = v.get("published")
        modified = v.get("modified")

        # Severity
        severity_raw = None
        sev = v.get("severity") or []
        if sev:
            severity_raw = sev[0].get("score") or sev[0].get("type")

        # Extract fix version from affected ranges
        fix_version = None
        introduced_version = None
        for aff in v.get("affected") or []:
            pkg = aff.get("package") or {}
            if pkg.get("ecosystem", "").lower() != "pypi":
                continue
            if pkg.get("name", "").lower() != package_name.lower():
                continue
            for rng in aff.get("ranges") or []:
                for event in rng.get("events") or []:
                    if "fixed" in event and not fix_version:
                        fix_version = event["fixed"]
                    if "introduced" in event and not introduced_version:
                        introduced_version = event["introduced"]

        rows.append({
            "package_name": package_name,
            "osv_id": osv_id,
            "published": published,
            "modified": modified,
            "summary": v.get("summary"),
            "severity_raw": severity_raw,
            "aliases": ";".join(v.get("aliases") or []),
            "fix_version": fix_version,
            "introduced_version": introduced_version,
        })

    return rows


# ---------- PyPI: get release dates ----------

def fetch_pypi_release_dates(package_name: str) -> Dict[str, str]:
    """Fetch all version release dates from PyPI JSON API. Returns {version: iso_date}."""
    try:
        url = f"https://pypi.org/pypi/{package_name}/json"
        resp = http_get(url)
        data = resp.json()
        releases = data.get("releases") or {}
        version_dates = {}
        for ver, files in releases.items():
            if files:
                upload_time = files[0].get("upload_time_iso_8601") or files[0].get("upload_time")
                if upload_time:
                    version_dates[ver] = upload_time
        return version_dates
    except Exception:
        return {}


# ---------- Severity classification ----------

def classify_severity(cvss_str: str) -> str:
    """Classify CVSS vector string into critical/high/medium/low."""
    if not cvss_str or pd.isna(cvss_str):
        return "unknown"
    s = str(cvss_str).upper()
    score = 0
    if "AV:N" in s: score += 3
    elif "AV:A" in s: score += 2
    elif "AV:L" in s: score += 1
    if "AC:L" in s: score += 1.5
    if "PR:N" in s: score += 2
    elif "PR:L" in s: score += 1
    if "UI:N" in s: score += 1
    if "S:C" in s or "SC:H" in s or "SC:L" in s: score += 1
    for impact in ["C:H", "I:H", "A:H", "VC:H", "VI:H", "VA:H"]:
        if impact in s: score += 1
    if score >= 8: return "critical"
    elif score >= 6: return "high"
    elif score >= 3: return "medium"
    else: return "low"


def quarter_bucket(ts: str) -> Optional[str]:
    if not ts or pd.isna(ts):
        return None
    dt = dtparser.isoparse(str(ts))
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year:04d}-Q{q}"


# ---------- Main ----------

def main():
    settings = load_settings()
    outdir = settings["outputs"]["outdir"]
    fmt = settings["outputs"]["format"]
    ext = "csv" if fmt == "csv" else "parquet"

    # Load pypi_repo_master for package→repo mapping
    pypi_path = f"{outdir}/pypi_repo_master.{ext}"
    pypi = pd.read_csv(pypi_path) if fmt == "csv" else pd.read_parquet(pypi_path)
    pypi["repo_full_name"] = pypi["github_url"].str.replace("https://github.com/", "", regex=False)
    pkg_to_repo = pypi[["package_name", "repo_full_name"]].drop_duplicates("package_name")

    packages = pypi["package_name"].dropna().unique().tolist()

    # ===== Step 1: Re-query OSV with affected ranges =====
    print(f"Step 1: Querying OSV for {len(packages)} packages (with fix versions)...")
    all_vulns = []
    for pkg in tqdm(packages, desc="OSV (fix versions)"):
        rows = osv_query_with_affected(pkg)
        all_vulns.extend(rows)

    vulns = pd.DataFrame(all_vulns)
    if vulns.empty:
        print("No vulnerabilities found. Exiting.")
        return

    print(f"  Total vulnerability records: {len(vulns)}")
    print(f"  Records with fix_version: {vulns['fix_version'].notna().sum()}")

    # ===== Step 2: Fetch PyPI release dates for packages with fix versions =====
    pkgs_with_fixes = vulns[vulns["fix_version"].notna()]["package_name"].unique().tolist()
    print(f"\nStep 2: Fetching PyPI release dates for {len(pkgs_with_fixes)} packages...")

    release_dates = {}  # {package_name: {version: date_str}}

    def fetch_one(pkg):
        return pkg, fetch_pypi_release_dates(pkg)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch_one, pkg): pkg for pkg in pkgs_with_fixes}
        for f in tqdm(as_completed(futures), total=len(futures), desc="PyPI releases"):
            pkg, dates = f.result()
            release_dates[pkg] = dates

    # ===== Step 3: Compute fix_date and time_to_fix =====
    print("\nStep 3: Computing time-to-fix...")

    fix_dates = []
    for _, row in vulns.iterrows():
        fix_date = None
        pkg = row["package_name"]
        fv = row["fix_version"]
        if pd.notna(fv) and pkg in release_dates:
            fix_date = release_dates[pkg].get(fv)
        fix_dates.append(fix_date)

    vulns["fix_date"] = fix_dates
    vulns["fix_date"] = pd.to_datetime(vulns["fix_date"], errors="coerce", utc=True)
    vulns["published_dt"] = pd.to_datetime(vulns["published"], errors="coerce", utc=True)

    vulns["time_to_fix_days"] = (vulns["fix_date"] - vulns["published_dt"]).dt.total_seconds() / 86400.0
    # Negative values mean fix was released before advisory was published (pre-disclosure)
    # Keep them — they're valid (responsible disclosure pattern)

    # Censoring flag: is_fixed = 1 if fix exists, 0 if still open
    vulns["is_fixed"] = vulns["fix_date"].notna().astype(int)

    # Severity classification
    vulns["severity_class"] = vulns["severity_raw"].apply(classify_severity)

    # Map to repos
    vulns = vulns.merge(pkg_to_repo, on="package_name", how="left")
    vulns = vulns.dropna(subset=["repo_full_name"])

    print(f"  Vulns with fix_date: {vulns['fix_date'].notna().sum()} / {len(vulns)}")
    print(f"  Vulns still open (censored): {(vulns['is_fixed'] == 0).sum()}")
    print(f"  Median time-to-fix (fixed only): {vulns.loc[vulns['is_fixed']==1, 'time_to_fix_days'].median():.1f} days")

    # ===== Step 4: Build dataset_vuln_survival (event-level) =====
    survival = vulns[[
        "repo_full_name", "package_name", "osv_id", "severity_class", "severity_raw",
        "published", "fix_version", "fix_date", "time_to_fix_days", "is_fixed", "aliases"
    ]].copy()

    # For censored (unfixed) vulns, duration = now - published
    now = datetime.now(timezone.utc)
    survival["duration_days"] = survival["time_to_fix_days"].copy()
    mask_censored = survival["is_fixed"] == 0
    survival.loc[mask_censored, "duration_days"] = (
        (now - vulns.loc[mask_censored, "published_dt"]).dt.total_seconds() / 86400.0
    )

    write_df(survival, f"{outdir}/dataset_vuln_survival.{ext}", fmt=fmt)
    print(f"\n  dataset_vuln_survival: {len(survival)} rows")

    # ===== Step 5: Build dataset_vuln_discovery_rate (repo-quarter) =====
    # Load repo-level transparency/governance metrics
    repo_path = f"{outdir}/dataset_repo_level_busfactor.{ext}"
    repo = pd.read_csv(repo_path) if fmt == "csv" else pd.read_parquet(repo_path)

    vulns["quarter"] = vulns["published"].apply(quarter_bucket)
    vulns_with_q = vulns.dropna(subset=["quarter"])

    # Count vulns per repo-quarter, by severity
    discovery = vulns_with_q.groupby(["repo_full_name", "quarter"]).agg(
        vuln_count=("osv_id", "count"),
        vuln_critical=("severity_class", lambda x: (x == "critical").sum()),
        vuln_high=("severity_class", lambda x: (x == "high").sum()),
        vuln_medium=("severity_class", lambda x: (x == "medium").sum()),
        vuln_low=("severity_class", lambda x: (x == "low").sum()),
    ).reset_index()
    discovery["vuln_severe"] = discovery["vuln_critical"] + discovery["vuln_high"]

    # Join repo-level transparency and governance metrics
    repo_cols = ["repo_full_name", "stars", "forks", "log_stars",
                 "transparency_index", "governance_index", "governance_artifact_score",
                 "scorecard_score", "external_contributor_ratio",
                 "bus_factor_proxy_k50", "gini_contrib", "top1_share", "contributor_count"]
    repo_cols = [c for c in repo_cols if c in repo.columns]
    discovery = discovery.merge(repo[repo_cols], on="repo_full_name", how="left")

    write_df(discovery, f"{outdir}/dataset_vuln_discovery_rate.{ext}", fmt=fmt)
    print(f"  dataset_vuln_discovery_rate: {len(discovery)} rows, {discovery['repo_full_name'].nunique()} repos")

    # ===== Summary =====
    print(f"\n{'='*60}")
    print("VULNERABILITY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total vulnerability events: {len(survival)}")
    print(f"  Fixed: {survival['is_fixed'].sum()} ({survival['is_fixed'].mean()*100:.1f}%)")
    print(f"  Censored (still open): {(survival['is_fixed']==0).sum()} ({(survival['is_fixed']==0).mean()*100:.1f}%)")
    fixed_only = survival[survival["is_fixed"] == 1]["time_to_fix_days"]
    if len(fixed_only) > 0:
        print(f"  Time-to-fix (fixed vulns): median={fixed_only.median():.1f}d, mean={fixed_only.mean():.1f}d, p90={fixed_only.quantile(0.9):.1f}d")
    print(f"\nSeverity breakdown:")
    print(survival["severity_class"].value_counts().to_string())
    print(f"\nDiscovery rate: {len(discovery)} repo-quarters across {discovery['repo_full_name'].nunique()} repos")


if __name__ == "__main__":
    main()
