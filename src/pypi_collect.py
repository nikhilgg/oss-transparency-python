import re
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, Optional, List
from src.common import load_settings, http_get, write_df

GITHUB_RE = re.compile(r"(https?://github\.com/[^/\s]+/[^/\s#]+)")

def extract_github_url(info: Dict[str, Any]) -> Optional[str]:
    # Try multiple places PyPI stores URLs
    urls = []
    project_urls = (info.get("project_urls") or {})
    if isinstance(project_urls, dict):
        urls.extend(list(project_urls.values()))
    for k in ("home_page", "project_url", "download_url"):
        v = info.get(k)
        if v:
            urls.append(v)
    for u in urls:
        if not u:
            continue
        m = GITHUB_RE.search(str(u))
        if m:
            url = m.group(1).rstrip("/")
            # Skip non-repo URLs like github.com/sponsors/username
            parts = url.split("/")
            if len(parts) >= 5 and parts[3] in ("sponsors", "orgs", "settings"):
                continue
            return url
    return None

def pypi_json(package: str) -> Dict[str, Any]:
    url = f"https://pypi.org/pypi/{package}/json"
    return http_get(url).json()

def load_top_packages_fallback(n: int) -> List[str]:
    """
    PyPI doesn't provide a simple official 'top downloads' endpoint.
    For research reproducibility, you can supply your own list (recommended).
    Here we provide a fallback: read from a curated list file if present.
    """
    import os
    path = "data/samples/pypi_packages.txt"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            pkgs = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.startswith("#")]
        return pkgs[:n]
    raise FileNotFoundError(
        "Provide data/samples/pypi_packages.txt (curated list of PyPI package names). "
        "This keeps the sampling reproducible."
    )

def main():
    settings = load_settings()
    outdir = settings["outputs"]["outdir"]
    fmt = settings["outputs"]["format"]
    top_n = int(__import__("os").getenv("PYPI_TOP_N", settings["sampling"]["pypi_top_n"]))

    packages = load_top_packages_fallback(top_n)
    rows = []
    for pkg in tqdm(packages, desc="PyPI metadata"):
        try:
            j = pypi_json(pkg)
            info = j.get("info", {})
            gh = extract_github_url(info)
            if not gh:
                continue
            rows.append({
                "package_name": pkg,
                "pypi_name": info.get("name"),
                "version_latest": info.get("version"),
                "summary": info.get("summary"),
                "github_url": gh,
                "license": info.get("license"),
                "requires_python": info.get("requires_python"),
            })
        except Exception as e:
            rows.append({"package_name": pkg, "error": str(e)})

    df = pd.DataFrame(rows)
    write_df(df, f"{outdir}/pypi_repo_master.{ 'csv' if fmt=='csv' else 'parquet'}", fmt=fmt)
    print(df.head(10).to_string(index=False))
    print(f"Wrote {len(df)} rows")

if __name__ == "__main__":
    main()
