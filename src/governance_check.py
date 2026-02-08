"""
Governance artifact detection for OSS repos via GitHub GraphQL API.

Checks for the existence of key governance files in a single query per repo:
  - SECURITY.md / .github/SECURITY.md
  - CODE_OF_CONDUCT.md / .github/CODE_OF_CONDUCT.md
  - CONTRIBUTING.md / .github/CONTRIBUTING.md
  - .github/CODEOWNERS
  - .github/FUNDING.yml

Returns a governance_artifact_score in [0, 1] = fraction of artifact groups present.
"""

import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.common import (
    load_settings, http_graphql, get_token_rotator, write_df,
)

# GraphQL query checks all file paths in a single call per repo
GOVERNANCE_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    security_root: object(expression: "HEAD:SECURITY.md") { __typename }
    security_gh: object(expression: "HEAD:.github/SECURITY.md") { __typename }
    coc_root: object(expression: "HEAD:CODE_OF_CONDUCT.md") { __typename }
    coc_gh: object(expression: "HEAD:.github/CODE_OF_CONDUCT.md") { __typename }
    contributing_root: object(expression: "HEAD:CONTRIBUTING.md") { __typename }
    contributing_gh: object(expression: "HEAD:.github/CONTRIBUTING.md") { __typename }
    codeowners_gh: object(expression: "HEAD:.github/CODEOWNERS") { __typename }
    codeowners_root: object(expression: "HEAD:CODEOWNERS") { __typename }
    funding: object(expression: "HEAD:.github/FUNDING.yml") { __typename }
  }
}
"""

# Each group = one "governance artifact". Having any path in the group counts.
ARTIFACT_GROUPS = [
    ("security", ["security_root", "security_gh"]),
    ("coc", ["coc_root", "coc_gh"]),
    ("contributing", ["contributing_root", "contributing_gh"]),
    ("codeowners", ["codeowners_gh", "codeowners_root"]),
    ("funding", ["funding"]),
]


def check_one_repo(repo: str, rotator) -> dict:
    owner, name = repo.split("/", 1)
    result = http_graphql(GOVERNANCE_QUERY,
                          variables={"owner": owner, "name": name},
                          rotator=rotator)

    repo_data = (result.get("data") or {}).get("repository")
    if not repo_data:
        return {"repo_full_name": repo, "governance_artifact_score": None}

    present = 0
    row = {"repo_full_name": repo}
    for group_name, keys in ARTIFACT_GROUPS:
        has_any = any(repo_data.get(k) is not None for k in keys)
        row[f"has_{group_name}"] = has_any
        present += int(has_any)

    row["governance_artifact_score"] = present / len(ARTIFACT_GROUPS)
    return row


def main():
    settings = load_settings()
    outdir = settings["outputs"]["outdir"]
    fmt = settings["outputs"]["format"]
    ext = "parquet" if fmt == "parquet" else "csv"

    meta_path = f"{outdir}/github_repo_meta.{ext}"
    meta = pd.read_parquet(meta_path) if fmt == "parquet" else pd.read_csv(meta_path)
    repos = [r for r in meta["repo_full_name"].dropna().unique() if "/" in r]

    rotator = get_token_rotator()
    print(f"Checking governance artifacts for {len(repos)} repos via GraphQL...")

    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(check_one_repo, r, rotator): r for r in repos}
        for future in tqdm(as_completed(futures), total=len(repos), desc="Governance check"):
            try:
                rows.append(future.result())
            except Exception as e:
                repo = futures[future]
                rows.append({"repo_full_name": repo, "governance_artifact_score": None})

    df = pd.DataFrame(rows)
    write_df(df, f"{outdir}/governance_artifacts.{ext}", fmt=fmt)
    print(f"Wrote {len(df)} rows")
    scored = df["governance_artifact_score"].dropna()
    if len(scored):
        print(f"  Mean governance artifact score: {scored.mean():.2f}")


if __name__ == "__main__":
    main()
