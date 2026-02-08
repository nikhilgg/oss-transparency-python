# OSS Transparency Research Tool

A data pipeline for studying transparency, governance, and maintainer risk across top open-source Python packages. Collects metadata from PyPI, GitHub, OSV, and OpenSSF Scorecard APIs, then builds curated research datasets for analysis.

## Pipeline Overview

```
data/samples/pypi_packages.txt  (curated list of ~260 top PyPI packages)
        |
        v
   pypi_collect         -- PyPI metadata + GitHub URL extraction
        |
        v
   github_collect       -- Repo meta, PRs, bug issues, contributors (GraphQL)
        |
        +--> osv_collect           -- Vulnerability records from OSV API
        +--> scorecard_collect     -- OpenSSF Scorecard scores
        +--> governance_check      -- Governance artifact detection (GraphQL)
        |
        v
   build_datasets       -- Aggregates into two research datasets
        |
        +--> dataset_repo_month_panel.parquet
        +--> dataset_repo_level_busfactor.parquet
```

## Quick Start

### Prerequisites

- Python 3.11+
- One or more GitHub Personal Access Tokens

### Setup

```bash
pip install -r requirements.txt

# Create .env with your GitHub token(s)
echo "GITHUB_TOKEN_PAT=ghp_your_token_here" > .env
# Optional: add more tokens for higher throughput
echo "GITHUB_TOKEN_PAT_2=ghp_second_token" >> .env
echo "GITHUB_TOKEN_PAT_3=ghp_third_token" >> .env
```

### Run the Full Pipeline

```bash
mkdir -p outputs/tables

python -m src.pypi_collect
python -m src.github_collect
python -m src.osv_collect
python -m src.scorecard_collect
python -m src.governance_check
python -m src.build_datasets
```

Each step is idempotent. The GitHub collector supports checkpoint-based resumption -- if interrupted, re-run and it will skip already-collected repos.

## Source Modules

| Module | Description |
|--------|-------------|
| `src/common.py` | Shared utilities: settings loader, HTTP retry logic, GitHub token rotation, GraphQL executor, file I/O |
| `src/pypi_collect.py` | Fetches PyPI package metadata and extracts GitHub repository URLs |
| `src/github_collect.py` | Collects repo metadata, PRs (with review latency), bug issues (with MTTR), and contributors via GraphQL + REST |
| `src/osv_collect.py` | Queries the OSV API for vulnerability records per package |
| `src/scorecard_collect.py` | Fetches OpenSSF Security Scorecard results from the public API |
| `src/governance_check.py` | Detects governance artifacts (SECURITY.md, CODE_OF_CONDUCT.md, CONTRIBUTING.md, CODEOWNERS, FUNDING.yml) via GraphQL |
| `src/build_datasets.py` | Aggregates raw tables into two curated research datasets with derived metrics |

## Output Datasets

### Raw Tables (in `outputs/tables/`)

| Table | Rows | Description |
|-------|------|-------------|
| `pypi_repo_master` | ~241 | PyPI metadata linked to GitHub URLs |
| `github_repo_meta` | ~231 | Repository metadata (stars, forks, language, license) |
| `github_prs` | ~22K | Pull requests with review latency and author association |
| `github_bug_issues` | ~9.5K | Bug-labeled issues with mean time to resolution |
| `github_contributors` | ~19K | Contributor commit counts per repo |
| `osv_vulns_raw` | ~1.6K | Vulnerability records per package |
| `scorecard_results` | ~213 | OpenSSF Scorecard scores and individual checks |
| `governance_artifacts` | ~231 | Governance file presence and artifact score |

### Curated Research Datasets

**`dataset_repo_month_panel`** -- Monthly panel (repo x month granularity)

Key columns: `pr_count`, `review_latency_p50`, `review_latency_p90`, `bug_issue_rate`, `bug_mttr_p50_days`, `scorecard_score`, `governance_index`, `external_contributor_ratio`, `repo_age_days`, `many_eyes_proxy`

**`dataset_repo_level_busfactor`** -- Repo-level risk assessment (one row per repo)

Key columns: `top1_share`, `gini_contrib`, `bus_factor_proxy_k50`, `scorecard_score`, `governance_artifact_score`, `governance_index`, `external_contributor_ratio`

## Derived Metrics

| Metric | Definition |
|--------|------------|
| `governance_index` | `0.6 * scorecard_score_normalized + 0.4 * governance_artifact_score` |
| `external_contributor_ratio` | Fraction of PR authors not in (OWNER, MEMBER, COLLABORATOR) |
| `bus_factor_proxy_k50` | Minimum contributors accounting for 50% of commits |
| `gini_contrib` | Gini coefficient of contributor commit distribution (0 = equal, 1 = concentrated) |
| `top1_share` | Fraction of total commits by the top contributor |
| `bug_mttr_p50_days` | Median time-to-resolution for bug issues (days) |
| `review_latency_p50` | Median time from PR creation to first review (hours) |
| `many_eyes_proxy` | Monthly PR count as a proxy for "many eyes" transparency |

## Configuration

Edit `config/settings.yaml`:

```yaml
ecosystem: pypi
window_months: 24
sampling:
  pypi_top_n: 300          # max packages from curated list
github:
  max_workers: 3           # parallel threads for GitHub API
outputs:
  format: parquet          # parquet or csv
  outdir: outputs/tables
```

## Token Rotation

The pipeline supports multiple GitHub PATs to maximize API throughput. Set environment variables `GITHUB_TOKEN_PAT`, `GITHUB_TOKEN_PAT_2`, through `GITHUB_TOKEN_PAT_5`. The `TokenRotator` automatically selects the token with the most remaining quota and sleeps when all tokens are rate-limited.

## CI/CD

The GitHub Actions workflow at `src/.github/workflows/nightly_extract.yml` runs the full pipeline on a nightly schedule (00:00 IST). It requires `GITHUB_TOKEN_PAT`, `GITHUB_TOKEN_PAT_2`, and `GITHUB_TOKEN_PAT_3` as repository secrets.
