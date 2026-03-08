# OSS Transparency Research Tool

A data pipeline for studying transparency, governance, and maintainer risk across top open-source Python packages. It collects metadata from PyPI, GitHub, OSV, and OpenSSF Scorecard APIs, then builds curated research datasets for hypothesis-driven analysis.

## Pipeline Overview

```
data/samples/pypi_packages.txt  (curated list of PyPI packages)
        |
        v
   pypi_collect         -- PyPI metadata + GitHub URL extraction
        |
        +--> github_collect        -- Repo meta, PRs, bug issues, contributors
        +--> osv_collect           -- Vulnerability records from OSV API
        |
        +--> scorecard_collect     -- OpenSSF Scorecard scores
        +--> governance_check      -- Governance artifact detection
        |
        v
   build_datasets       -- Builds panel + repo-level datasets
        |
        +--> dataset_repo_month_panel.(csv|parquet)
        +--> dataset_repo_level_busfactor.(csv|parquet)
        +--> dataset_vuln_quarterly.(csv|parquet)
        |
        v
   vuln_analysis        -- Builds vulnerability modeling datasets
        |
        +--> dataset_vuln_discovery_rate.(csv|parquet)
        +--> dataset_vuln_survival.(csv|parquet)
```

## Quick Start

### Prerequisites

- Python 3.9+
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
python -m src.vuln_analysis
python -m src.data_quality
```

Each step is idempotent. The GitHub collector supports checkpoint-based resumption (`outputs/tables/github_checkpoint.jsonl`) if interrupted.

## Source Modules

| Module | Description |
|--------|-------------|
| `src/common.py` | Shared utilities: settings loader, HTTP retry logic, GitHub token rotation, GraphQL executor, file I/O |
| `src/pypi_collect.py` | Fetches PyPI package metadata and extracts GitHub repository URLs |
| `src/github_collect.py` | Collects repo metadata, PRs (with review latency), bug issues (with MTTR), and contributors via GraphQL + REST |
| `src/osv_collect.py` | Queries the OSV API for vulnerability records per package |
| `src/scorecard_collect.py` | Fetches OpenSSF Security Scorecard results from the public API |
| `src/governance_check.py` | Detects governance artifacts (SECURITY.md, CODE_OF_CONDUCT.md, CONTRIBUTING.md, CODEOWNERS, FUNDING.yml) via GraphQL |
| `src/build_datasets.py` | Aggregates extracted tables into panel and repo-level research datasets with derived metrics |
| `src/vuln_analysis.py` | Builds vulnerability discovery-rate and survival datasets using OSV + PyPI release timelines |
| `src/data_quality.py` | Runs missingness, outlier, and correlation diagnostics; writes correlation matrices |

## Output Datasets

### Raw Tables (in `outputs/tables/`)

Latest observed snapshot from current workspace files (March 8, 2026):

| Table | Rows | Description |
|-------|------|-------------|
| `pypi_repo_master` | 191 | PyPI metadata linked to GitHub URLs |
| `github_repo_meta` | 231 | Repository metadata (stars, forks, language, license) |
| `github_prs` | 22,342 | Pull requests with review latency and author association |
| `github_bug_issues` | 9,565 | Bug-labeled issues with resolution timing |
| `github_contributors` | 19,007 | Contributor commit counts per repo |
| `osv_vulns_raw` | 1,576 | OSV vulnerability records per package |
| `scorecard_results` | 215 | OpenSSF Scorecard scores and check-level fields |
| `governance_artifacts` | 231 | Governance file presence and artifact score |

### Curated Research Datasets

| Dataset | Rows | Purpose |
|---------|------|---------|
| `dataset_repo_month_panel` | 54,740 | Monthly panel for RQ1 and RQ4 |
| `dataset_repo_level_busfactor` | 231 | Repo-level risk dataset for RQ3 |
| `dataset_vuln_quarterly` | 372 | Repo-quarter vulnerability summary |
| `dataset_vuln_discovery_rate` | 372 | Discovery-rate modeling dataset for RQ2a |
| `dataset_vuln_survival` | 1,576 | Event-level survival dataset for RQ2b |
| `clean_repo_month_panel` | 54,740 | Cleaned panel dataset |
| `clean_repo_level_busfactor` | 231 | Cleaned repo-level dataset |
| `clean_vuln_discovery_rate` | 364 | Cleaned discovery-rate dataset |
| `clean_vuln_survival` | 1,576 | Cleaned survival dataset |

## Derived Metrics

| Metric | Definition |
|--------|------------|
| `governance_index` | `0.6 * scorecard_score_normalized + 0.4 * governance_artifact_score` |
| `external_contributor_ratio` | Fraction of PR authors not in (`OWNER`, `MEMBER`, `COLLABORATOR`) |
| `bus_factor_proxy_k50` | Minimum contributors accounting for 50% of commits |
| `gini_contrib` | Gini coefficient of contributor commit distribution (0 = equal, 1 = concentrated) |
| `top1_share` | Fraction of total commits by the top contributor |
| `bug_mttr_p50_days` | Median time-to-resolution for bug issues (days) |
| `review_latency_p50` | Median time from PR creation to first review (hours) |
| `many_eyes_proxy` | Monthly PR count as a proxy for "many eyes" transparency |
| `transparency_index` | Composite of normalized PR volume, contributor count, and external contributor ratio |

## Configuration

Edit `config/settings.yaml`:

```yaml
ecosystem: pypi
window_months: 24
sampling:
  pypi_top_n: 300
  min_stars: 100
  min_prs_last_12m: 20
github:
  max_workers: 3
outputs:
  format: csv          # csv or parquet
  outdir: outputs/tables
```

## Token Rotation

The pipeline supports multiple GitHub PATs to maximize API throughput. Set environment variables `GITHUB_TOKEN_PAT`, `GITHUB_TOKEN_PAT_2`, through `GITHUB_TOKEN_PAT_5`. `TokenRotator` selects the token with the most remaining quota and sleeps when all tokens are rate-limited.

## CI/CD

The GitHub Actions workflow at `src/.github/workflows/nightly_extract.yml` runs the full pipeline on a nightly schedule (00:00 IST). It requires `GITHUB_TOKEN_PAT`, `GITHUB_TOKEN_PAT_2`, and `GITHUB_TOKEN_PAT_3` as repository secrets.
