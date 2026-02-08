import os
import time
import json
import math
import yaml
import threading
import requests
from dataclasses import dataclass, field
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

def gh_headers(token: Optional[str] = None) -> Dict[str, str]:
    if token is None:
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


# ---------------------------------------------------------------------------
# Token rotation for multiple GitHub PATs
# ---------------------------------------------------------------------------

@dataclass
class _TokenState:
    token: str
    remaining: int = 5000
    reset_at: float = 0.0  # unix timestamp


class TokenRotator:
    """Round-robins across multiple GitHub PATs, preferring the one with most remaining quota."""

    def __init__(self):
        self._tokens: List[_TokenState] = []
        self._lock = threading.Lock()
        # Load tokens from env: GITHUB_TOKEN_PAT, GITHUB_TOKEN_PAT_2, GITHUB_TOKEN_PAT_3, ...
        for key in ("GITHUB_TOKEN_PAT", "GITHUB_TOKEN_PAT_2", "GITHUB_TOKEN_PAT_3",
                     "GITHUB_TOKEN_PAT_4", "GITHUB_TOKEN_PAT_5"):
            val = os.getenv(key, "").strip()
            if val:
                self._tokens.append(_TokenState(token=val))
        if not self._tokens:
            raise EnvironmentError("No GITHUB_TOKEN_PAT* environment variables set")

    @property
    def count(self) -> int:
        return len(self._tokens)

    def get_token(self) -> str:
        """Return the token with most remaining quota. If all exhausted, sleep until earliest reset."""
        with self._lock:
            now = time.time()
            # Refresh any token whose reset window has passed
            for ts in self._tokens:
                if ts.remaining <= 0 and now >= ts.reset_at:
                    ts.remaining = 5000  # assume reset

            # Sort by remaining (descending)
            best = max(self._tokens, key=lambda t: t.remaining)

            if best.remaining <= 0:
                # All exhausted — sleep until earliest reset
                earliest = min(ts.reset_at for ts in self._tokens)
                wait_sec = max(0, earliest - now) + 1
                print(f"[TokenRotator] All tokens exhausted, sleeping {wait_sec:.0f}s until reset")
                time.sleep(wait_sec)
                for ts in self._tokens:
                    ts.remaining = 5000
                best = self._tokens[0]

            return best.token

    def update_limits(self, token: str, response: requests.Response) -> None:
        """Update rate limit tracking from response headers."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset_at = response.headers.get("X-RateLimit-Reset")
        if remaining is None:
            return
        with self._lock:
            for ts in self._tokens:
                if ts.token == token:
                    ts.remaining = int(remaining)
                    if reset_at:
                        ts.reset_at = float(reset_at)
                    break

    def status(self) -> str:
        """Return a summary of token rate limit status."""
        parts = []
        for i, ts in enumerate(self._tokens):
            parts.append(f"T{i+1}:{ts.remaining}")
        return " | ".join(parts)


# Global singleton — lazy-initialized
_rotator: Optional[TokenRotator] = None
_rotator_lock = threading.Lock()


def get_token_rotator() -> TokenRotator:
    global _rotator
    if _rotator is None:
        with _rotator_lock:
            if _rotator is None:
                _rotator = TokenRotator()
    return _rotator


# ---------------------------------------------------------------------------
# GraphQL helper
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=2, max=120))
def http_graphql(query: str, variables: Optional[Dict[str, Any]] = None,
                 rotator: Optional[TokenRotator] = None) -> Dict[str, Any]:
    """Execute a GitHub GraphQL query with token rotation and retry."""
    if rotator is None:
        rotator = get_token_rotator()

    token = rotator.get_token()
    headers = gh_headers(token=token)

    body = {"query": query}
    if variables:
        body["variables"] = variables

    r = requests.post("https://api.github.com/graphql", json=body, headers=headers, timeout=60)
    rotator.update_limits(token, r)

    if r.status_code == 403:
        retry_after = int(r.headers.get("Retry-After", 60))
        time.sleep(retry_after)
        raise RuntimeError(f"GraphQL 403 (rate limit): {r.text[:200]}")
    if r.status_code in (429, 500, 502, 503, 504):
        raise RuntimeError(f"GraphQL HTTP {r.status_code}: {r.text[:200]}")
    r.raise_for_status()

    data = r.json()
    if "errors" in data:
        err_msgs = [e.get("message", "") for e in data["errors"]]
        # "rate limit" errors are retryable
        if any("rate limit" in m.lower() for m in err_msgs):
            time.sleep(30)
            raise RuntimeError(f"GraphQL rate limit error: {err_msgs}")
        # NOT_FOUND is not retryable — return the error data as-is
        if any("Could not resolve" in m for m in err_msgs):
            return data
        # Other errors — return as-is (caller decides)
        return data

    return data
