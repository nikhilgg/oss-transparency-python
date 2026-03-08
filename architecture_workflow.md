# OSS Transparency Pipeline: Architecture and Workflow

## 1) End-to-End Architecture

```mermaid
flowchart LR
  A["Curated package list<br/>data/samples/pypi_packages.txt"] --> B["pypi_collect.py<br/>PyPI metadata + GitHub URL mapping"]

  B --> C["github_collect.py<br/>Repo meta + PRs + bug issues + contributors"]
  B --> D["osv_collect.py<br/>OSV vulnerability ingestion"]

  C --> E["scorecard_collect.py<br/>OpenSSF Scorecard API"]
  C --> F["governance_check.py<br/>Governance artifact detection"]

  C --> G["build_datasets.py"]
  D --> G
  E --> G
  F --> G
  B --> G

  G --> H["dataset_repo_month_panel.csv<br/>RQ1 + RQ4 modeling base"]
  G --> I["dataset_repo_level_busfactor.csv<br/>RQ3 modeling base"]
  G --> J["dataset_vuln_quarterly.csv<br/>Quarterly vulnerability counts"]

  I --> K["vuln_analysis.py"]
  B --> K

  K --> L["dataset_vuln_survival.csv<br/>RQ2b survival modeling"]
  K --> M["dataset_vuln_discovery_rate.csv<br/>RQ2a discovery-rate modeling"]

  H --> N["notebooks/02_h1_model_governance_manyeyes.ipynb"]
  H --> O["notebooks/06_h4_review_latency_reliability.ipynb"]
  I --> P["notebooks/05_h3_bus_factor_model.ipynb"]
  M --> Q["notebooks/03_h2a_vuln_discovery_model.ipynb"]
  L --> R["notebooks/04_h2b_survival_analysis.ipynb"]

  H --> S["data_quality.py"]
  I --> S
  J --> S
  S --> T["correlation_matrix_*.csv + QA summaries"]
```

## 2) Workflow (Slide-Friendly)

1. **Sampling and Mapping**  
   `pypi_collect.py` reads curated PyPI packages and maps each package to a GitHub repository.

2. **Repository Process Collection**  
   `github_collect.py` ingests repository metadata, pull-request activity, bug issues, and contributor distributions.

3. **Security Signal Collection**  
   `osv_collect.py` captures vulnerability events; `scorecard_collect.py` captures security posture; `governance_check.py` captures policy/process artifacts.

4. **Feature Engineering and Dataset Construction**  
   `build_datasets.py` creates governance and transparency indices plus modeling-ready datasets:
   - `dataset_repo_month_panel` (RQ1, RQ4)
   - `dataset_repo_level_busfactor` (RQ3)
   - `dataset_vuln_quarterly` (supporting vulnerability trend view)

5. **Vulnerability Event Modeling Data**  
   `vuln_analysis.py` enriches OSV findings with fix-version and release-date logic to produce:
   - `dataset_vuln_discovery_rate` (RQ2a)
   - `dataset_vuln_survival` (RQ2b)

6. **Modeling and Validation**  
   Notebooks execute RQ-specific models; `data_quality.py` runs missingness, outlier, and collinearity checks.

## 3) Recommended Slide 6 Narration (30-45 sec)

"The pipeline starts with a curated PyPI package universe, then links packages to GitHub repositories. We collect process and collaboration metrics from GitHub, vulnerability events from OSV, governance maturity from artifact checks, and security posture from OpenSSF Scorecard. These are integrated into panel-level and repo-level analytical datasets. A specialized vulnerability pipeline then derives discovery-rate and time-to-fix datasets for survival analysis. Finally, hypothesis-specific notebooks and quality checks produce statistically validated evidence for governance, transparency, and maintainer-risk effects on software reliability."
