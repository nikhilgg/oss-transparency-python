import os
import pandas as pd
from tqdm import tqdm
from dateutil import parser
from typing import Dict, Any, List
from src.common import load_settings, http_post, write_df

def osv_query(package_name: str) -> Dict[str, Any]:
    url = "https://api.osv.dev/v1/query"
    body = {"package": {"name": package_name, "ecosystem": "PyPI"}}
    return http_post(url, json_body=body).json()

def main():
    settings = load_settings()
    outdir = settings["outputs"]["outdir"]
    fmt = settings["outputs"]["format"]
    master_path = f"{outdir}/pypi_repo_master.{ 'csv' if fmt=='csv' else 'parquet'}"
    master = pd.read_parquet(master_path) if fmt == "parquet" else pd.read_csv(master_path)

    rows = []
    for pkg in tqdm(master["package_name"].dropna().unique(), desc="OSV PyPI"):
        try:
            res = osv_query(pkg)
            vulns = res.get("vulns") or []
            for v in vulns:
                # OSV fields: id, modified, published, summary, details, severity, affected, references
                vid = v.get("id")
                published = v.get("published")
                modified = v.get("modified")
                severity = None
                sev = v.get("severity") or []
                # Some OSV entries include CVSS vector; keep raw
                if sev:
                    severity = sev[0].get("score") or sev[0].get("type")
                # Determine fix date from affected ranges or versions; can be messy → store raw for later parsing
                rows.append({
                    "package_name": pkg,
                    "osv_id": vid,
                    "published": published,
                    "modified": modified,
                    "summary": v.get("summary"),
                    "details": (v.get("details") or "")[:5000],
                    "severity_raw": severity,
                    "references": ";".join([r.get("url","") for r in (v.get("references") or [])]),
                    "aliases": ";".join(v.get("aliases") or []),
                })
        except Exception as e:
            rows.append({"package_name": pkg, "error": str(e)})

    df = pd.DataFrame(rows)
    write_df(df, f"{outdir}/osv_vulns_raw.{ 'csv' if fmt=='csv' else 'parquet'}", fmt=fmt)
    print(f"Wrote {len(df)} rows")

if __name__ == "__main__":
    main()
