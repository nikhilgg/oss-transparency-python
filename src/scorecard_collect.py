import os
import json
import glob
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.common import load_settings, http_get, write_df


def fetch_scorecard_api(repo_full_name: str) -> dict:
    """Fetch scorecard from api.securityscorecards.dev for a single repo."""
    url = f"https://api.securityscorecards.dev/projects/github.com/{repo_full_name}"
    r = http_get(url)
    return r.json()


def parse_scorecard_response(j: dict, repo_full_name: str) -> dict:
    """Parse scorecard API response into a flat dict."""
    score = j.get("score")
    checks = j.get("checks") or []
    d = {
        "repo_full_name": repo_full_name,
        "scorecard_score": score,
    }
    for c in checks:
        name = c.get("name")
        if name:
            d[f"sc_{name}"] = c.get("score")
    return d


def main():
    settings = load_settings()
    outdir = settings["outputs"]["outdir"]
    fmt = settings["outputs"]["format"]

    # Load repo list from github_repo_meta
    ext = "parquet" if fmt == "parquet" else "csv"
    meta_path = f"{outdir}/github_repo_meta.{ext}"
    meta = pd.read_parquet(meta_path) if fmt == "parquet" else pd.read_csv(meta_path)
    repos = [r for r in meta["repo_full_name"].dropna().unique() if "/" in r and "error" not in str(r)]

    print(f"Fetching scorecards for {len(repos)} repos from API...")

    rows = []
    errors = 0

    def fetch_one(repo):
        try:
            j = fetch_scorecard_api(repo)
            return parse_scorecard_response(j, repo)
        except Exception as e:
            return {"repo_full_name": repo, "scorecard_score": None, "error": str(e)[:100]}

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch_one, r): r for r in repos}
        for future in tqdm(as_completed(futures), total=len(repos), desc="Scorecard API"):
            result = future.result()
            if "error" in result:
                errors += 1
            rows.append(result)

    df = pd.DataFrame(rows)
    # Drop error column before writing (keep only rows with scores)
    if "error" in df.columns:
        scored = df[df["scorecard_score"].notna()].drop(columns=["error"], errors="ignore")
    else:
        scored = df[df["scorecard_score"].notna()]

    write_df(scored, f"{outdir}/scorecard_results.{ext}", fmt=fmt)
    print(f"Wrote {len(scored)} scored rows ({errors} repos had no scorecard)")


if __name__ == "__main__":
    main()
