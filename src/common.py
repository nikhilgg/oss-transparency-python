import os
import time
import json
import math
import yaml
import requests
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential

def load_settings(path: str = "config/settings.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def months_ago(n: int) -> datetime:
    # approximate month window
    return utc_now() - timedelta(days=int(n * 30.4375))

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_df(df, outpath: str, fmt: str = "parquet") -> None:
    ensure_dir(os.path.dirname(outpath))
    if fmt == "csv":
        df.to_csv(outpath, index=False)
    elif fmt == "parquet":
        df.to_parquet(outpath, index=False)
    else:
        raise ValueError(f"Unknown format: {fmt}")

@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=1, max=60))
def http_get(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> requests.Response:
    r = requests.get(url, headers=headers, params=params, timeout=60)
    if r.status_code == 403:
        retry_after = int(r.headers.get("Retry-After", 60))
        time.sleep(retry_after)
        raise RuntimeError(f"Retryable HTTP 403 (secondary rate limit): {url}")
    if r.status_code in (429, 500, 502, 503, 504):
        raise RuntimeError(f"Retryable HTTP {r.status_code}: {url}")
    r.raise_for_status()
    return r

@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=1, max=60))
def http_post(url: str, json_body: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> requests.Response:
    r = requests.post(url, json=json_body, headers=headers, timeout=60)
    if r.status_code in (429, 500, 502, 503, 504):
        raise RuntimeError(f"Retryable HTTP {r.status_code}: {url}")
    r.raise_for_status()
    return r

def gh_headers() -> Dict[str, str]:
    token = os.getenv("GITHUB_TOKEN_PAT", "").strip()
    hdr = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "oss-transparency-python-research"
    }
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    return hdr

def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
