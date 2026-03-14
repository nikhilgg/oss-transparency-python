# Data Dictionary

This document describes the data used and produced by the `oss-transparency-python` project.

## Scope

The project studies transparency, governance, contributor concentration, and vulnerability outcomes for a curated set of PyPI packages mapped to GitHub repositories.

Primary upstream sources:

- PyPI JSON API: package metadata and release history
- GitHub GraphQL API: repository metadata, pull requests, governance files, bug issues
- GitHub REST API: contributors
- OSV API: vulnerability advisories
- OpenSSF Scorecard API: repository security posture metrics

## Project Structure And Data Flow

| Layer | Path | Purpose |
|---|---|---|
| Input sample | `data/samples/pypi_packages.txt` | Curated package list used as the starting sample |
| Configuration | `config/settings.yaml` | Extraction window, output format, API behavior |
| Raw extraction code | `src/pypi_collect.py`, `src/github_collect.py`, `src/osv_collect.py`, `src/scorecard_collect.py`, `src/governance_check.py` | Builds raw tables in `outputs/tables/` |
| Dataset builders | `src/build_datasets.py`, `src/vuln_analysis.py` | Builds panel, repo-level, quarterly, and survival datasets |
| Quality checks | `src/data_quality.py` | Writes correlation matrices and validates outputs |
| Data outputs | `outputs/tables/*.csv` | Materialized analysis datasets used by notebooks |
| Analysis notebooks | `notebooks/*.ipynb` | Modeling and exploratory analysis on curated datasets |

## Core Entities

| Entity | Grain | Natural key | Notes |
|---|---|---|---|
| Package | One PyPI package | `package_name` | Starting unit of collection |
| Repository | One GitHub repository | `repo_full_name` | Usually mapped from a PyPI package |
| Pull request | One PR in one repo | `repo_full_name` + `pr_number` | Latest 100 PRs per repo from GraphQL |
| Bug issue | One bug-labeled issue in one repo | `repo_full_name` + `issue_number` | Latest 100 bug issues per repo |
| Contributor | One contributor in one repo | `repo_full_name` + `contributor_login` | Single REST page, up to 100 contributors |
| Vulnerability | One OSV advisory affecting one package | `osv_id` + `package_name` | Linked to repo through PyPI mapping |
| Repo-month | One repo in one calendar month | `repo_full_name` + `month` | Main panel dataset |
| Repo-quarter | One repo in one calendar quarter | `repo_full_name` + `quarter` | Vulnerability aggregation grain |

## Inputs

### `data/samples/pypi_packages.txt`

| Field | Type | Description |
|---|---|---|
| line value | string | PyPI package name. One package per non-comment line. |

### `config/settings.yaml`

| Field | Type | Description |
|---|---|---|
| `ecosystem` | string | Package ecosystem. Current project uses `pypi`. |
| `window_months` | integer | Lookback window parameter for GitHub collection logic. |
| `sampling.pypi_top_n` | integer | Maximum number of packages read from the curated package file. |
| `sampling.min_stars` | integer | Stated sampling threshold for repository popularity. Not enforced in current Python modules. |
| `sampling.min_prs_last_12m` | integer | Stated PR activity threshold. Not enforced in current Python modules. |
| `outputs.format` | string | Output format: `csv` or `parquet`. |
| `outputs.outdir` | string | Output directory for materialized datasets. |
| `github.api_base` | string | GitHub REST API base URL. |
| `github.graphql_url` | string | GitHub GraphQL endpoint. |
| `github.per_page` | integer | REST page size setting. |
| `github.max_workers` | integer | Thread pool size for GitHub collection. |
| `github.checkpoint_path` | string | JSONL checkpoint file used for resume behavior. |
| `osv.api_base` | string | OSV API base URL. |
| `scorecard.enabled` | boolean | Feature flag for scorecard collection. |

## Raw Extracted Tables

### `outputs/tables/pypi_repo_master.csv`

Grain: one row per sampled package with a resolvable GitHub repository URL.

| Field | Type | Description |
|---|---|---|
| `package_name` | string | Package name from the curated sample list. |
| `pypi_name` | string | Canonical name returned by PyPI metadata. |
| `version_latest` | string | Latest version reported by PyPI. |
| `summary` | string | Short package description from PyPI. |
| `github_url` | string | Extracted GitHub repository URL. |
| `license` | string | License string from PyPI metadata. |
| `requires_python` | string | Python version requirement declared on PyPI. |

### `outputs/tables/github_repo_meta.csv`

Grain: one row per GitHub repository.

| Field | Type | Description |
|---|---|---|
| `repo_full_name` | string | Repository in `owner/name` form. |
| `repo_id` | integer | GitHub database ID. |
| `default_branch` | string | Default branch name. |
| `created_at` | datetime string | Repository creation timestamp. |
| `updated_at` | datetime string | Repository update timestamp. |
| `pushed_at` | datetime string | Timestamp of latest push. |
| `stars` | integer | Stargazer count. |
| `forks` | integer | Fork count. |
| `open_issues` | integer | Number of open issues. |
| `language` | string | Primary language name. |
| `archived` | boolean | Whether the repo is archived. |
| `fork` | boolean | Whether the repo is itself a fork. |
| `license` | string | SPDX license ID from GitHub. |
| `error` | string | Error message when collection failed for that repo. Present only for failed rows. |

### `outputs/tables/github_prs.csv`

Grain: one row per pull request.

| Field | Type | Description |
|---|---|---|
| `repo_full_name` | string | Repository in `owner/name` form. |
| `pr_number` | integer | Pull request number. |
| `pr_created_at` | datetime string | PR creation timestamp. |
| `pr_closed_at` | datetime string | PR close timestamp. |
| `pr_merged_at` | datetime string | PR merge timestamp. |
| `first_review_at` | datetime string | Timestamp of the first review returned by GraphQL. |
| `review_count` | integer | Review count captured in the query response. In current code this is effectively `0` or `1` because only the first review node is requested. |
| `author_association` | string | GitHub author association such as `OWNER`, `MEMBER`, `COLLABORATOR`, or external contributor types. |
| `latency_first_review_hours` | float | Hours from PR creation to first review. |
| `latency_merge_hours` | float | Hours from PR creation to merge. |

### `outputs/tables/github_bug_issues.csv`

Grain: one row per bug-labeled issue.

| Field | Type | Description |
|---|---|---|
| `repo_full_name` | string | Repository in `owner/name` form. |
| `issue_number` | integer | Issue number. |
| `created_at` | datetime string | Issue creation timestamp. |
| `closed_at` | datetime string | Issue close timestamp. |
| `mttr_days` | float | Mean time to resolution in days for that issue, computed as `closed_at - created_at`. |
| `state` | string | GitHub issue state. |
| `comments` | integer | Comment count. |

### `outputs/tables/github_contributors.csv`

Grain: one row per contributor-repo pair.

| Field | Type | Description |
|---|---|---|
| `repo_full_name` | string | Repository in `owner/name` form. |
| `contributor_login` | string | Contributor login, name, or `unknown` fallback. |
| `contributions` | integer | Contribution count returned by GitHub contributors API. |
| `type` | string | Contributor type from GitHub, for example `User`. |

### `outputs/tables/osv_vulns_raw.csv`

Grain: one row per OSV vulnerability affecting a package.

| Field | Type | Description |
|---|---|---|
| `package_name` | string | Affected PyPI package. |
| `osv_id` | string | OSV or GHSA advisory identifier. |
| `published` | datetime string | Advisory publication timestamp. |
| `modified` | datetime string | Advisory last modification timestamp. |
| `summary` | string | Short advisory summary. |
| `details` | string | Truncated advisory description, capped at 5000 characters in code. |
| `severity_raw` | string | Raw severity score or severity type from OSV. Often a CVSS vector string. |
| `references` | string | Semicolon-delimited advisory reference URLs. |
| `aliases` | string | Semicolon-delimited alias IDs such as CVEs or PYSECs. |

### `outputs/tables/scorecard_results.csv`

Grain: one row per repository with a successful OpenSSF Scorecard response.

| Field | Type | Description |
|---|---|---|
| `repo_full_name` | string | Repository in `owner/name` form. |
| `scorecard_score` | float | Overall OpenSSF Scorecard score on a 0 to 10 scale. |
| `sc_Code-Review` | float | Scorecard check score for code review controls. |
| `sc_Dangerous-Workflow` | float | Scorecard check score for dangerous GitHub workflow usage. |
| `sc_Security-Policy` | float | Scorecard check score for presence of a security policy. |
| `sc_Binary-Artifacts` | float | Scorecard check score for binary artifact risks. |
| `sc_Maintained` | float | Scorecard check score for maintenance signals. |
| `sc_CII-Best-Practices` | float | Scorecard check score for CII best practices. |
| `sc_Token-Permissions` | float | Scorecard check score for token permissions hardening. |
| `sc_Vulnerabilities` | float | Scorecard check score for vulnerability handling signals. |
| `sc_License` | float | Scorecard check score for license metadata. |
| `sc_Fuzzing` | float | Scorecard check score for fuzzing adoption. |
| `sc_Packaging` | float | Scorecard check score for packaging quality. |
| `sc_Pinned-Dependencies` | float | Scorecard check score for dependency pinning. |
| `sc_Signed-Releases` | float | Scorecard check score for signed releases. |
| `sc_Branch-Protection` | float | Scorecard check score for branch protection. |
| `sc_SAST` | float | Scorecard check score for static analysis. |
| `sc_Dependency-Update-Tool` | float | Scorecard check score for dependency update tooling. |
| `sc_CI-Tests` | float | Scorecard check score for continuous integration tests. |
| `sc_Contributors` | float | Scorecard check score for contributor trust/process controls. |

Notes:

- Check columns are dynamic and depend on what the Scorecard API returns.
- Negative values can occur in the source output and should be interpreted as API-defined sentinel values rather than ordinary scores.

### `outputs/tables/governance_artifacts.csv`

Grain: one row per repository.

| Field | Type | Description |
|---|---|---|
| `repo_full_name` | string | Repository in `owner/name` form. |
| `has_security` | boolean | Whether `SECURITY.md` exists in root or `.github/`. |
| `has_coc` | boolean | Whether `CODE_OF_CONDUCT.md` exists in root or `.github/`. |
| `has_contributing` | boolean | Whether `CONTRIBUTING.md` exists in root or `.github/`. |
| `has_codeowners` | boolean | Whether `CODEOWNERS` exists in root or `.github/`. |
| `has_funding` | boolean | Whether `.github/FUNDING.yml` exists. |
| `governance_artifact_score` | float | Fraction of artifact groups present, from `0.0` to `1.0`. |

## Derived Research Datasets

### `outputs/tables/dataset_repo_month_panel.csv`

Grain: one row per repository-month.

| Field | Type | Description |
|---|---|---|
| `repo_full_name` | string | Repository in `owner/name` form. |
| `month` | string | Calendar month bucket in `YYYY-MM` form. |
| `review_latency_p50` | float | Median first-review latency in hours for PRs in that repo-month. |
| `review_latency_p90` | float | 90th percentile first-review latency in hours for PRs in that repo-month. |
| `pr_count` | integer | Number of PRs in that repo-month. |
| `bug_issue_rate` | integer | Number of bug issues created in that repo-month. |
| `bug_mttr_p50_days` | float | Median bug issue resolution time in days for that repo-month. |
| `stars` | integer | Repository star count, copied from repo metadata. |
| `forks` | integer | Repository fork count, copied from repo metadata. |
| `language` | string | Primary language. |
| `created_at` | datetime string | Original repository creation timestamp from GitHub metadata. |
| `scorecard_score` | float | Overall OpenSSF Scorecard score. |
| `repo_created_at` | datetime | Parsed `created_at` timestamp. |
| `month_start` | datetime | First day of the `month` bucket. |
| `repo_age_days` | integer | Approximate repo age in days at month start. Can be negative because the panel uses all observed months across all repos, including months before some repos existed. |
| `governance_index` | float | Composite governance metric: `0.6 * (scorecard_score / 10) + 0.4 * governance_artifact_score`. |
| `external_contributor_ratio` | float | Share of PRs whose `author_association` is not `OWNER`, `MEMBER`, or `COLLABORATOR`. |
| `many_eyes_proxy` | integer | Monthly PR count, equal to `pr_count`. |
| `transparency_index` | float | Mean of min-max normalized repo-level PR volume, contributor count, and external contributor ratio. |
| `contributor_count` | integer | Unique contributor count per repo from `github_contributors`. |
| `many_eyes_x_governance` | float | Interaction term: `many_eyes_proxy * governance_index`. |
| `transparency_x_governance` | float | Interaction term: `transparency_index * governance_index`. |

### `outputs/tables/dataset_repo_level_busfactor.csv`

Grain: one row per repository.

| Field | Type | Description |
|---|---|---|
| `repo_full_name` | string | Repository in `owner/name` form. |
| `stars` | integer | Repository star count. |
| `forks` | integer | Repository fork count. |
| `language` | string | Primary language. |
| `created_at` | datetime string | Repository creation timestamp. |
| `top1_share` | float | Share of all contributor commits attributable to the top contributor. |
| `gini_contrib` | float | Gini coefficient of contributor concentration. Higher values mean more concentrated contribution distribution. |
| `bus_factor_proxy_k50` | integer | Smallest number of contributors needed to account for at least 50% of contributions. |
| `scorecard_score` | float | Overall OpenSSF Scorecard score. |
| `governance_artifact_score` | float | Governance file presence score from 0 to 1. |
| `governance_index` | float | Composite governance metric. |
| `external_contributor_ratio` | float | Share of external PR authors. |
| `transparency_index` | float | Composite transparency metric. |
| `contributor_count` | integer | Unique contributor count. |
| `vuln_total` | integer | Total vulnerability count linked to the repo through package mapping. |
| `vuln_critical` | integer | Count of vulnerabilities classified as critical. |
| `vuln_high` | integer | Count of vulnerabilities classified as high. |
| `vuln_medium` | integer | Count of vulnerabilities classified as medium. |
| `vuln_low` | integer | Count of vulnerabilities classified as low. |
| `vuln_severe` | integer | `vuln_critical + vuln_high`. |
| `has_severe_vuln` | integer | Binary flag equal to 1 when `vuln_severe > 0`. |
| `log_stars` | float | `log(1 + stars)` control variable. |

### `outputs/tables/dataset_vuln_quarterly.csv`

Grain: one row per repository-quarter.

| Field | Type | Description |
|---|---|---|
| `repo_full_name` | string | Repository in `owner/name` form. |
| `quarter` | string | Calendar quarter bucket in `YYYY-Qn` form. |
| `critical` | integer | Count of critical vulnerabilities published that quarter. |
| `high` | integer | Count of high vulnerabilities published that quarter. |
| `medium` | integer | Count of medium vulnerabilities published that quarter. |
| `unknown` | integer | Count of vulnerabilities with unknown severity. |
| `low` | integer | Count of low vulnerabilities published that quarter. |
| `vuln_total` | integer | Total vulnerabilities in that repo-quarter. |
| `vuln_severe` | integer | `critical + high` in that repo-quarter. |

### `outputs/tables/dataset_vuln_discovery_rate.csv`

Grain: one row per repository-quarter.

| Field | Type | Description |
|---|---|---|
| `repo_full_name` | string | Repository in `owner/name` form. |
| `quarter` | string | Calendar quarter bucket in `YYYY-Qn` form. |
| `vuln_count` | integer | Total OSV advisories published for the repo in that quarter. |
| `vuln_critical` | integer | Critical vulnerability count. |
| `vuln_high` | integer | High vulnerability count. |
| `vuln_medium` | integer | Medium vulnerability count. |
| `vuln_low` | integer | Low vulnerability count. |
| `vuln_severe` | integer | `vuln_critical + vuln_high`. |
| `stars` | integer | Repository star count. |
| `forks` | integer | Repository fork count. |
| `log_stars` | float | Log-transformed star count. |
| `transparency_index` | float | Repo-level transparency metric. |
| `governance_index` | float | Repo-level governance metric. |
| `governance_artifact_score` | float | Governance file presence score. |
| `scorecard_score` | float | Overall Scorecard score. |
| `external_contributor_ratio` | float | Share of external PR authors. |
| `bus_factor_proxy_k50` | integer | Contributor concentration proxy. |
| `gini_contrib` | float | Contributor concentration coefficient. |
| `top1_share` | float | Share of commits by the top contributor. |
| `contributor_count` | integer | Unique contributor count. |

### `outputs/tables/dataset_vuln_survival.csv`

Grain: one row per vulnerability event.

| Field | Type | Description |
|---|---|---|
| `repo_full_name` | string | Repository linked from the affected package. |
| `package_name` | string | Affected PyPI package. |
| `osv_id` | string | Advisory identifier. |
| `severity_class` | string | Derived severity bucket: `critical`, `high`, `medium`, `low`, or `unknown`. |
| `severity_raw` | string | Raw OSV severity string. |
| `published` | datetime string | Advisory publication timestamp. |
| `fix_version` | string | Package version identified as fixing the vulnerability from OSV affected ranges. |
| `fix_date` | datetime | Release timestamp of the fix version from PyPI release history. |
| `time_to_fix_days` | float | `fix_date - published` in days. Can be negative when the fix predates public disclosure. |
| `is_fixed` | integer | `1` when a fix date is known, else `0`. |
| `aliases` | string | Semicolon-delimited alias IDs such as CVEs. |
| `duration_days` | float | Survival duration. Equals `time_to_fix_days` for fixed vulnerabilities, otherwise `now - published`. |

## Cleaned Datasets Present In Outputs

These tables exist in `outputs/tables/`, but their cleaning logic is not implemented in the Python modules under `src/`. Their definitions below are inferred from the materialized CSVs.

### `outputs/tables/clean_repo_month_panel.csv`

Same grain and columns as `dataset_repo_month_panel`, plus:

| Field | Type | Description |
|---|---|---|
| `has_pr` | integer | Binary indicator equal to 1 when `pr_count > 0`. |
| `has_bug` | integer | Binary indicator equal to 1 when `bug_issue_rate > 0`. |

Observed cleaning behavior from sample rows:

- Missing numeric activity measures are filled with `0.0`.

### `outputs/tables/clean_repo_level_busfactor.csv`

Schema matches `dataset_repo_level_busfactor.csv` in the current workspace.

### `outputs/tables/clean_vuln_discovery_rate.csv`

Schema matches `dataset_vuln_discovery_rate.csv` in the current workspace.

### `outputs/tables/clean_vuln_survival.csv`

Same grain and columns as `dataset_vuln_survival`, plus:

| Field | Type | Description |
|---|---|---|
| `severity_missing` | integer | Binary indicator equal to 1 when raw severity information was missing or unclassified in the cleaning workflow. |

## Diagnostics Outputs

### `outputs/tables/correlation_matrix_repo_level.csv`

Square correlation matrix written by `src/data_quality.py` for selected numeric repo-level variables.

### `outputs/tables/correlation_matrix_panel_active.csv`

Square correlation matrix written by `src/data_quality.py` for selected numeric panel variables using active months only.

## Derived Metric Definitions

| Metric | Definition |
|---|---|
| `governance_artifact_score` | Fraction of governance artifact groups present across security policy, code of conduct, contributing guide, codeowners, and funding file |
| `governance_index` | `0.6 * (scorecard_score / 10) + 0.4 * governance_artifact_score` |
| `external_contributor_ratio` | External PRs divided by total PRs, where internal authors are `OWNER`, `MEMBER`, or `COLLABORATOR` |
| `top1_share` | Top contributor contributions divided by total contributions |
| `gini_contrib` | Gini coefficient of contributor contribution counts |
| `bus_factor_proxy_k50` | Smallest `k` such that top `k` contributors account for at least 50% of contributions |
| `transparency_index` | Average of min-max normalized `total_prs_repo`, `contributor_count`, and `external_contributor_ratio` |
| `many_eyes_proxy` | Monthly PR count |
| `review_latency_p50` | Median hours from PR creation to first review within a repo-month |
| `review_latency_p90` | 90th percentile hours from PR creation to first review within a repo-month |
| `bug_mttr_p50_days` | Median bug issue time to resolution in days within a repo-month |
| `vuln_severe` | Sum of critical and high vulnerability counts |
| `log_stars` | `log(1 + stars)` |

## Known Data Caveats

| Area | Caveat |
|---|---|
| PR coverage | GitHub GraphQL query collects only the latest 100 PRs per repo. |
| Bug issue coverage | GitHub GraphQL query collects only the latest 100 bug-labeled issues per repo. |
| Contributor coverage | GitHub REST collection uses one page only, so contributor lists are effectively capped at 100 rows per repo. |
| Repo-month panel base | Panel uses all observed months across all repos, which can create months that predate a repo's creation and therefore negative `repo_age_days`. |
| Review count | `review_count` is not total reviews; current query requests only the first review node. |
| Severity classification | Severity buckets are derived heuristically from raw CVSS strings when available. |
| Time-to-fix | Negative `time_to_fix_days` values are possible and valid when a fix release predates advisory publication. |
| Clean datasets | Cleaning transformations are not defined in `src/`; only output schemas can be verified from current CSVs. |

