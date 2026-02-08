import os
import json
import time
import threading
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from dateutil import parser
from typing import List, Dict, Any, Optional, Set

from src.common import (
    load_settings, http_get, http_graphql, gh_headers,
    get_token_rotator, months_ago, iso, write_df, ensure_dir,
)

# ---------------------------------------------------------------------------
# GraphQL query — fetches repo meta, PRs (with first review), and bug issues
# in a single API call.  Contributors stays REST (no GraphQL equivalent).
# ---------------------------------------------------------------------------

REPO_GRAPHQL = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    databaseId
    name
    nameWithOwner
    defaultBranchRef { name }
    createdAt
    updatedAt
    pushedAt
    stargazerCount
    forkCount
    primaryLanguage { name }
    isArchived
    isFork
    licenseInfo { spdxId }
    openIssues: issues(states: OPEN) { totalCount }

    pullRequests(last: 100, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number
        createdAt
        closedAt
        mergedAt
        authorAssociation
        reviews(first: 1) {
          nodes { createdAt }
        }
      }
    }

    bugIssues: issues(last: 100, labels: ["bug"], orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number
        createdAt
        closedAt
        state
        comments { totalCount }
      }
    }
  }
}
"""


def parse_repo_full_name(github_url: str) -> str:
    parts = github_url.rstrip("/").split("/")
    return f"{parts[-2]}/{parts[-1]}"


# ---------------------------------------------------------------------------
# Checkpoint helpers — JSONL append for crash-safe resume
# ---------------------------------------------------------------------------

_checkpoint_lock = threading.Lock()


def load_checkpoint(path: str) -> Set[str]:
    """Load set of repo_full_names already collected."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                repo = obj.get("repo_full_name")
                if repo:
                    done.add(repo)
            except json.JSONDecodeError:
                continue
    return done


def append_checkpoint(path: str, record: Dict[str, Any]) -> None:
    """Thread-safe append of a single JSON record to checkpoint file."""
    with _checkpoint_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


def load_checkpoint_data(path: str) -> List[Dict[str, Any]]:
    """Load all checkpoint records."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ---------------------------------------------------------------------------
# Data extraction from GraphQL response
# ---------------------------------------------------------------------------

def extract_repo_meta(repo_data: Dict[str, Any], repo_full_name: str) -> Dict[str, Any]:
    return {
        "repo_full_name": repo_full_name,
        "repo_id": repo_data.get("databaseId"),
        "default_branch": (repo_data.get("defaultBranchRef") or {}).get("name"),
        "created_at": repo_data.get("createdAt"),
        "updated_at": repo_data.get("updatedAt"),
        "pushed_at": repo_data.get("pushedAt"),
        "stars": repo_data.get("stargazerCount"),
        "forks": repo_data.get("forkCount"),
        "open_issues": (repo_data.get("openIssues") or {}).get("totalCount"),
        "language": (repo_data.get("primaryLanguage") or {}).get("name"),
        "archived": repo_data.get("isArchived"),
        "fork": repo_data.get("isFork"),
        "license": (repo_data.get("licenseInfo") or {}).get("spdxId"),
    }


def extract_pull_requests(repo_data: Dict[str, Any], repo_full_name: str) -> List[Dict[str, Any]]:
    """Extract PRs from GraphQL response. No date filter — GraphQL already returns last 100."""
    pr_nodes = (repo_data.get("pullRequests") or {}).get("nodes") or []
    rows = []
    for pr in pr_nodes:
        created = pr.get("createdAt")
        reviews = (pr.get("reviews") or {}).get("nodes") or []
        first_review_at = reviews[0].get("createdAt") if reviews else None
        review_count = len(reviews)
        rows.append({
            "repo_full_name": repo_full_name,
            "pr_number": pr.get("number"),
            "pr_created_at": created,
            "pr_closed_at": pr.get("closedAt"),
            "pr_merged_at": pr.get("mergedAt"),
            "first_review_at": first_review_at,
            "review_count": review_count,
            "author_association": pr.get("authorAssociation"),
        })
    return rows


def extract_bug_issues(repo_data: Dict[str, Any], repo_full_name: str) -> List[Dict[str, Any]]:
    """Extract bug issues from GraphQL response. No date filter — GraphQL already returns last 100."""
    issue_nodes = (repo_data.get("bugIssues") or {}).get("nodes") or []
    rows = []
    for it in issue_nodes:
        created = it.get("createdAt")
        closed = it.get("closedAt")
        mttr_days = None
        if created and closed:
            mttr_days = (parser.isoparse(closed) - parser.isoparse(created)).total_seconds() / 86400.0
        rows.append({
            "repo_full_name": repo_full_name,
            "issue_number": it.get("number"),
            "created_at": created,
            "closed_at": closed,
            "mttr_days": mttr_days,
            "state": it.get("state"),
            "comments": (it.get("comments") or {}).get("totalCount"),
        })
    return rows


def collect_contributors_rest(repo: str) -> List[Dict[str, Any]]:
    """Contributors via REST API (no GraphQL equivalent). Single page only."""
    rotator = get_token_rotator()
    token = rotator.get_token()
    url = f"https://api.github.com/repos/{repo}/contributors"
    r = http_get(url, headers=gh_headers(token=token), params={"anon": "true", "per_page": 100})
    rotator.update_limits(token, r)
    contributors = r.json() if isinstance(r.json(), list) else []
    rows = []
    for c in contributors:
        rows.append({
            "repo_full_name": repo,
            "contributor_login": c.get("login") or c.get("name") or "unknown",
            "contributions": c.get("contributions"),
            "type": c.get("type"),
        })
    return rows


# ---------------------------------------------------------------------------
# Per-repo collection (runs in thread pool)
# ---------------------------------------------------------------------------

def collect_one_repo(repo: str, since_iso: str, checkpoint_path: str,
                     rotator) -> Optional[Dict[str, Any]]:
    """Collect all data for a single repo via GraphQL + REST contributors.
    Returns a summary dict, or None if repo should be skipped."""
    owner, name = repo.split("/", 1)

    # GraphQL call for meta + PRs + bug issues
    result = http_graphql(REPO_GRAPHQL, variables={
        "owner": owner, "name": name,
    }, rotator=rotator)

    # Handle errors
    if "errors" in result and not result.get("data"):
        append_checkpoint(checkpoint_path, {
            "repo_full_name": repo, "error": str(result["errors"])
        })
        return {"repo_full_name": repo, "error": str(result["errors"])}

    repo_data = (result.get("data") or {}).get("repository")
    if not repo_data:
        append_checkpoint(checkpoint_path, {
            "repo_full_name": repo, "error": "repository not found"
        })
        return {"repo_full_name": repo, "error": "repository not found"}

    # Skip archived/forked repos
    if repo_data.get("isArchived") or repo_data.get("isFork"):
        append_checkpoint(checkpoint_path, {
            "repo_full_name": repo, "skipped": "archived_or_fork"
        })
        return None

    meta = extract_repo_meta(repo_data, repo)
    prs = extract_pull_requests(repo_data, repo)
    bugs = extract_bug_issues(repo_data, repo)

    # REST: contributors
    try:
        contribs = collect_contributors_rest(repo)
    except Exception as e:
        contribs = []

    # Compute PR latency fields
    def hours(a, b):
        if not a or not b:
            return None
        return (parser.isoparse(b) - parser.isoparse(a)).total_seconds() / 3600.0

    for pr in prs:
        pr["latency_first_review_hours"] = hours(pr["pr_created_at"], pr["first_review_at"])
        pr["latency_merge_hours"] = hours(pr["pr_created_at"], pr["pr_merged_at"])

    # Write checkpoint
    record = {
        "repo_full_name": repo,
        "meta": meta,
        "prs": prs,
        "bugs": bugs,
        "contribs": contribs,
    }
    append_checkpoint(checkpoint_path, record)
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    settings = load_settings()
    outdir = settings["outputs"]["outdir"]
    fmt = settings["outputs"]["format"]
    window_months = int(os.getenv("WINDOW_MONTHS", settings["window_months"]))
    since = months_ago(window_months)
    since_iso = iso(since)
    max_workers = int(settings.get("github", {}).get("max_workers", 3))
    checkpoint_path = settings.get("github", {}).get(
        "checkpoint_path", f"{outdir}/github_checkpoint.jsonl"
    )

    ensure_dir(os.path.dirname(checkpoint_path))

    # Load master
    master_path = f"{outdir}/pypi_repo_master.{'csv' if fmt == 'csv' else 'parquet'}"
    master = pd.read_parquet(master_path) if fmt == "parquet" else pd.read_csv(master_path)
    master = master.dropna(subset=["github_url"]).copy()
    master["repo_full_name"] = master["github_url"].apply(parse_repo_full_name)
    all_repos = list(master["repo_full_name"].unique())

    # Resume from checkpoint
    done = load_checkpoint(checkpoint_path)
    remaining = [r for r in all_repos if r not in done]
    print(f"Total repos: {len(all_repos)} | Already done: {len(done)} | Remaining: {len(remaining)}")

    # Initialize token rotator
    rotator = get_token_rotator()
    print(f"[TokenRotator] {rotator.count} token(s) loaded")

    # Parallel collection
    repo_meta_rows = []
    pbar = tqdm(total=len(remaining), desc="GitHub repos (GraphQL)")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(collect_one_repo, repo, since_iso, checkpoint_path, rotator): repo
            for repo in remaining
        }
        for future in as_completed(futures):
            repo = futures[future]
            try:
                result = future.result()
                if result:
                    repo_meta_rows.append(result)
            except Exception as e:
                repo_meta_rows.append({"repo_full_name": repo, "error": str(e)})
            pbar.update(1)
            # Log rate limit status every 20 repos
            if pbar.n % 20 == 0:
                print(f"\n  [Rate limits] {rotator.status()}")

    pbar.close()

    # Rebuild full datasets from checkpoint (includes both old + new runs)
    print("Rebuilding datasets from checkpoint...")
    all_records = load_checkpoint_data(checkpoint_path)

    all_meta = []
    all_prs = []
    all_bugs = []
    all_contribs = []

    for rec in all_records:
        if "error" in rec and "meta" not in rec:
            all_meta.append({"repo_full_name": rec["repo_full_name"], "error": rec["error"]})
            continue
        if "skipped" in rec:
            continue
        if "meta" in rec:
            all_meta.append(rec["meta"])
            all_prs.extend(rec.get("prs", []))
            all_bugs.extend(rec.get("bugs", []))
            all_contribs.extend(rec.get("contribs", []))

    # Write output tables
    ext = "csv" if fmt == "csv" else "parquet"

    repo_meta_df = pd.DataFrame(all_meta)
    write_df(repo_meta_df, f"{outdir}/github_repo_meta.{ext}", fmt=fmt)
    print(f"  github_repo_meta: {len(repo_meta_df)} rows")

    if all_prs:
        prs_df = pd.DataFrame(all_prs)
        write_df(prs_df, f"{outdir}/github_prs.{ext}", fmt=fmt)
        print(f"  github_prs: {len(prs_df)} rows")

    if all_bugs:
        bug_df = pd.DataFrame(all_bugs)
        write_df(bug_df, f"{outdir}/github_bug_issues.{ext}", fmt=fmt)
        print(f"  github_bug_issues: {len(bug_df)} rows")

    if all_contribs:
        cdf = pd.DataFrame(all_contribs)
        write_df(cdf, f"{outdir}/github_contributors.{ext}", fmt=fmt)
        print(f"  github_contributors: {len(cdf)} rows")

    print("Done.")


if __name__ == "__main__":
    main()
