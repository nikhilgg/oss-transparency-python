import os
import json
import glob
import pandas as pd
from src.common import load_settings, write_df

def main():
    settings = load_settings()
    outdir = settings["outputs"]["outdir"]
    fmt = settings["outputs"]["format"]

    files = glob.glob("outputs/scorecard/*.json")
    rows = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            j = json.load(f)
        repo = (j.get("repo") or {}).get("name") or j.get("repo", "")
        score = j.get("score")
        checks = j.get("checks") or []
        d = {
            "repo_full_name": repo.replace("https://github.com/", "").replace("github.com/", ""),
            "scorecard_score": score,
        }
        # store a few key checks as columns
        for c in checks:
            name = c.get("name")
            d[f"sc_{name}"] = c.get("score")
        rows.append(d)

    df = pd.DataFrame(rows)
    write_df(df, f"{outdir}/scorecard_results.{ 'csv' if fmt=='csv' else 'parquet'}", fmt=fmt)
    print(f"Wrote {len(df)} rows")

if __name__ == "__main__":
    main()
