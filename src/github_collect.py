import os
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from typing import List, Dict, Any
from dateutil import parser
from src.common import load_settings, http_get, gh_headers, months_ago, iso, write_df

def parse_repo_full_name(github_url: str) -> str:
    # https://github.com/owner/repo
    parts = github_url.rstrip("/").split("/")
    return f"{parts[-2]}/{parts[-1]}"

def gh_paginate(url: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    page = 1
    per_page = int(params.get("per_page", 100))
    while True:
        p = dict(params)
        p["page"] = page
        r = http_get(url, headers=gh_headers(), params=p)
        batch = r.json()
        if not isinstance(batch, list) or len(batch) == 0:
            break
        items.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
    return items

def collect_repo_meta(repo: str) -> Dict[str, Any]:
    url = f"https://api.github.com/repos/{repo}"
    j = http_get(url, headers=gh_headers()).json()
    return {
        "repo_full_name": repo,
        "repo_id": j.get("id"),
        "default_branch": j.get("default_branch"),
        "created_at": j.get("created_at"),
        "updated_at": j.get("updated_at"),
        "pushed_at": j.get("pushed_at"),
        "stars": j.get("stargazers_count"),
        "forks": j.get("forks_count"),
        "open_issues": j.get("open_issues_count"),
        "language": j.get("language"),
        "archived": j.get("archived"),
        "fork": j.get("fork"),
        "license": (j.get("license") or {}).get("spdx_id"),
    }

def collect_pull_requests(repo: str, since_iso: str) -> pd.DataFrame:
    # PR list endpoint includes PRs; we then fetch reviews for latency
    url = f"https://api.github.com/repos/{repo}/pulls"
    pulls = gh_paginate(url, {"state": "all", "sort": "updated", "direction": "desc", "per_page": 100})
    rows = []
    for pr in pulls:
        created = pr.get("created_at")
        if created and created < since_iso:
            # because sorted by updated, can't break safely; just skip older
            pass
        pr_number = pr["number"]
        pr_created = pr.get("created_at")
        pr_merged = pr.get("merged_at")
        pr_closed = pr.get("closed_at")
        # reviews
        rev_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews"
        reviews = gh_paginate(rev_url, {"per_page": 100})
        first_review_at = None
        if reviews:
            # earliest submitted_at
            times = [rv.get("submitted_at") for rv in reviews if rv.get("submitted_at")]
            if times:
                first_review_at = sorted(times)[0]
        rows.append({
            "repo_full_name": repo,
            "pr_number": pr_number,
            "pr_created_at": pr_created,
            "pr_closed_at": pr_closed,
            "pr_merged_at": pr_merged,
            "first_review_at": first_review_at,
            "review_count": len(reviews),
            "author_association": pr.get("author_association"),
        })
    df = pd.DataFrame(rows)
    # compute latency hours
    def hours(a, b):
        if not a or not b:
            return None
        return (parser.isoparse(b) - parser.isoparse(a)).total_seconds() / 3600.0
    if len(df):
        df["latency_first_review_hours"] = df.apply(lambda r: hours(r["pr_created_at"], r["first_review_at"]), axis=1)
        df["latency_merge_hours"] = df.apply(lambda r: hours(r["pr_created_at"], r["pr_merged_at"]), axis=1)
    return df

def collect_bug_issues(repo: str, since_iso: str) -> pd.DataFrame:
    # Use search API to find issues with label:bug is heavy; use issues endpoint + filter labels
    url = f"https://api.github.com/repos/{repo}/issues"
    issues = gh_paginate(url, {"state": "all", "since": since_iso, "per_page": 100})
    rows = []
    for it in issues:
        if "pull_request" in it:
            continue
        labels = [lb.get("name", "").lower() for lb in (it.get("labels") or [])]
        is_bug = any("bug" == l or "type: bug" in l or "kind/bug" in l for l in labels)
        if not is_bug:
            continue
        created = it.get("created_at")
        closed = it.get("closed_at")
        mttr_days = None
        if created and closed:
            mttr_days = (parser.isoparse(closed) - parser.isoparse(created)).total_seconds() / 86400.0
        rows.append({
            "repo_full_name": repo,
            "issue_number": it.get("number"),
            "created_at": created,
            "closed_at": closed,
            "mttr_days": mttr_days,
            "state": it.get("state"),
            "comments": it.get("comments"),
        })
    return pd.DataFrame(rows)

def collect_commit_contributors(repo: str) -> pd.DataFrame:
    # contributor stats endpoint is cached + sometimes 202; fall back to /contributors
    url = f"https://api.github.com/repos/{repo}/contributors"
    contributors = gh_paginate(url, {"anon": "true", "per_page": 100})
    rows = []
    for c in contributors:
        rows.append({
            "repo_full_name": repo,
            "contributor_login": c.get("login") or c.get("name") or "unknown",
            "contributions": c.get("contributions"),
            "type": c.get("type"),
        })
    return pd.DataFrame(rows)

def main():
    settings = load_settings()
    outdir = settings["outputs"]["outdir"]
    fmt = settings["outputs"]["format"]
    window_months = int(os.getenv("WINDOW_MONTHS", settings["window_months"]))
    since = months_ago(window_months)
    since_iso = iso(since)

    # Load master
    master_path = f"{outdir}/pypi_repo_master.{ 'csv' if fmt=='csv' else 'parquet'}"
    master = pd.read_parquet(master_path) if fmt == "parquet" else pd.read_csv(master_path)
    master = master.dropna(subset=["github_url"]).copy()
    master["repo_full_name"] = master["github_url"].apply(parse_repo_full_name)

    repo_meta_rows = []
    all_prs = []
    all_bug_issues = []
    all_contribs = []

    for repo in tqdm(master["repo_full_name"].unique(), desc="GitHub repos"):
        try:
            meta = collect_repo_meta(repo)
            if meta["archived"] or meta["fork"]:
                continue
            repo_meta_rows.append(meta)

            prs = collect_pull_requests(repo, since_iso)
            if len(prs):
                all_prs.append(prs)

            bugs = collect_bug_issues(repo, since_iso)
            if len(bugs):
                all_bug_issues.append(bugs)

            contrib = collect_commit_contributors(repo)
            if len(contrib):
                all_contribs.append(contrib)

        except Exception as e:
            repo_meta_rows.append({"repo_full_name": repo, "error": str(e)})

    repo_meta = pd.DataFrame(repo_meta_rows)
    write_df(repo_meta, f"{outdir}/github_repo_meta.{ 'csv' if fmt=='csv' else 'parquet'}", fmt=fmt)

    if all_prs:
        prs_df = pd.concat(all_prs, ignore_index=True)
        write_df(prs_df, f"{outdir}/github_prs.{ 'csv' if fmt=='csv' else 'parquet'}", fmt=fmt)

    if all_bug_issues:
        bug_df = pd.concat(all_bug_issues, ignore_index=True)
        write_df(bug_df, f"{outdir}/github_bug_issues.{ 'csv' if fmt=='csv' else 'parquet'}", fmt=fmt)

    if all_contribs:
        cdf = pd.concat(all_contribs, ignore_index=True)
        write_df(cdf, f"{outdir}/github_contributors.{ 'csv' if fmt=='csv' else 'parquet'}", fmt=fmt)

    print("Done.")

if __name__ == "__main__":
    main()
