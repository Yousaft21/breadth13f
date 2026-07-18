"""OpenFIGI client for mapping 13F CUSIPs to tickers / FIGIs.

OpenFIGI (https://www.openfigi.com/api) is a free API that maps third-party
identifiers (here CUSIP) to Bloomberg FIGIs and their trading symbols. We use it
as the primary CUSIP -> ticker crosswalk.

Rate limits (POST /v3/mapping):

* **No API key:** 25 requests/minute, 10 jobs/request.
* **With API key** (``OPENFIGI_API_KEY`` env var): 25 requests / 6 seconds,
  100 jobs/request.

The client:

* batches CUSIPs into jobs of the allowed size,
* rate-limits and retries (429/5xx) with backoff,
* **caches every CUSIP's result to disk** (one JSON per CUSIP, including an
  explicit empty list for "no match") so a resumed run only queries CUSIPs it
  has never seen.

A CUSIP can resolve to several instruments (share classes / venues); selecting
the tradable US-equity representative is left to :mod:`security_master`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import requests

logger = logging.getLogger("breadth13f.openfigi")

_MAPPING_URL = "https://api.openfigi.com/v3/mapping"


class OpenFigiClient:
    def __init__(
        self,
        cache_dir: Path,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get("OPENFIGI_API_KEY")
        self.timeout = timeout_seconds
        self.max_retries = max_retries

        # Batch/rate parameters depend on whether we are authenticated.
        if self.api_key:
            self.batch_size = 100
            self._min_interval = 6.0 / 25.0  # 25 requests / 6 seconds
        else:
            self.batch_size = 10
            self._min_interval = 60.0 / 25.0  # 25 requests / minute
        self._last_request = 0.0

        self._session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-OPENFIGI-APIKEY"] = self.api_key
        self._session.headers.update(headers)

    # --- cache helpers --------------------------------------------------------
    def _cache_path(self, cusip: str) -> Path:
        shard = cusip[:2].upper()
        return self.cache_dir / shard / f"{cusip.upper()}.json"

    def _read_cache(self, cusip: str) -> list[dict] | None:
        path = self._cache_path(cusip)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _write_cache(self, cusip: str, instruments: list[dict]) -> None:
        path = self._cache_path(cusip)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(instruments), encoding="utf-8")

    # --- rate limiting --------------------------------------------------------
    def _throttle(self) -> None:
        now = time.monotonic()
        sleep_for = self._min_interval - (now - self._last_request)
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last_request = time.monotonic()

    # --- public API -----------------------------------------------------------
    def map_cusips(self, cusips: list[str], force: bool = False) -> dict[str, list[dict]]:
        """Map many CUSIPs to their OpenFIGI instrument lists (cached).

        Returns a dict ``{cusip: [instrument, ...]}``. A CUSIP with no match maps
        to an empty list. Only CUSIPs not already cached are sent to the API.
        """
        results: dict[str, list[dict]] = {}
        to_query: list[str] = []
        for c in {c.upper() for c in cusips if c}:
            if not force:
                cached = self._read_cache(c)
                if cached is not None:
                    results[c] = cached
                    continue
            to_query.append(c)

        if to_query:
            logger.info(
                "OpenFIGI: %d CUSIPs cached, querying %d new (batch=%d, key=%s)",
                len(results), len(to_query), self.batch_size, bool(self.api_key),
            )
        for i in range(0, len(to_query), self.batch_size):
            batch = to_query[i : i + self.batch_size]
            batch_results = self._map_batch(batch)
            for c, instruments in batch_results.items():
                self._write_cache(c, instruments)
                results[c] = instruments
        return results

    def _map_batch(self, cusips: list[str]) -> dict[str, list[dict]]:
        payload = [{"idType": "ID_CUSIP", "idValue": c} for c in cusips]
        data = self._post_with_retry(payload)
        # Response preserves job order; align back to the requested CUSIPs.
        out: dict[str, list[dict]] = {}
        for cusip, job_result in zip(cusips, data):
            out[cusip] = job_result.get("data", []) if isinstance(job_result, dict) else []
        return out

    def _post_with_retry(self, payload: list[dict]) -> list[dict]:
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self._session.post(_MAPPING_URL, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                wait = self._min_interval * attempt
                logger.warning("OpenFIGI request error (%d/%d): %s — retry in %.1fs",
                               attempt, self.max_retries, exc, wait)
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                # Respect the rate window generously before retrying.
                wait = max(self._min_interval * 2, 6.0) * attempt
                logger.warning("OpenFIGI 429 rate-limited — backing off %.1fs", wait)
                time.sleep(wait)
                continue
            if resp.status_code in (500, 502, 503, 504):
                time.sleep(self._min_interval * attempt)
                continue
            resp.raise_for_status()
        raise RuntimeError("OpenFIGI mapping failed after retries")
