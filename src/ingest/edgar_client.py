"""Rate-limited, disk-cached HTTP client for SEC EDGAR.

Design goals
------------
* **Fair-access compliant** — a global token-bucket keeps us under the SEC's
  10 requests/second cap (configurable, defaults to 7/s) and every request
  carries the descriptive ``User-Agent`` the SEC requires.
* **Cache everything** — every downloaded byte is written to disk keyed by the
  request path, so a resumed run never re-hits EDGAR for something it already
  has. Large text submissions are gzip-compressed on disk.
* **Resilient** — transient errors (429/5xx/connection) are retried with
  exponential backoff.

The cache key mirrors the EDGAR archive path, e.g.::

    https://www.sec.gov/Archives/edgar/full-index/2020/QTR1/form.idx
      -> data/raw/edgar/full-index/2020/QTR1/form.idx(.gz)
"""

from __future__ import annotations

import gzip
import logging
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

import requests

logger = logging.getLogger("breadth13f.edgar")


class RateLimiter:
    """Simple thread-safe minimum-interval limiter (requests/second cap)."""

    def __init__(self, requests_per_second: float) -> None:
        self._min_interval = 1.0 / float(requests_per_second)
        self._last = 0.0
        self._lock = Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = self._min_interval - (now - self._last)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last = time.monotonic()


class EdgarClient:
    """Fetch EDGAR resources with rate limiting, retries and a disk cache."""

    def __init__(
        self,
        cache_dir: Path,
        user_agent: str,
        requests_per_second: float = 7.0,
        max_retries: int = 5,
        backoff_seconds: float = 2.0,
        timeout_seconds: float = 30.0,
        connect_timeout_seconds: float = 10.0,
        conn_retries: int = 2,
        conn_backoff_seconds: float = 1.5,
        max_backoff_seconds: float = 30.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds
        self.connect_timeout = connect_timeout_seconds
        self.max_retries = max_retries
        self.backoff = backoff_seconds
        self.max_backoff = max_backoff_seconds
        self.conn_retries = conn_retries
        self.conn_backoff = conn_backoff_seconds
        self._limiter = RateLimiter(requests_per_second)
        self._session = requests.Session()
        # Host is set automatically per-request by requests from the URL, so it
        # is intentionally not pinned here (we hit both www.sec.gov and
        # data.sec.gov). The User-Agent is what the SEC fair-access policy needs.
        self._session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )

    # --- cache helpers --------------------------------------------------------
    def _cache_path(self, url: str, compress: bool) -> Path:
        """Map an EDGAR URL to a local path under the cache directory."""
        parsed = urlparse(url)
        # Strip the leading "/Archives/" so the cache mirrors the archive tree.
        # (Different SEC hosts — www vs data.sec.gov — are separated by giving
        # each client its own cache_dir, so host is not part of the key here.)
        rel = parsed.path.lstrip("/")
        if rel.startswith("Archives/"):
            rel = rel[len("Archives/") :]
        local = self.cache_dir / rel
        if compress:
            local = local.with_name(local.name + ".gz")
        return local

    @staticmethod
    def _read_cached(path: Path) -> bytes:
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as fh:
                return fh.read()
        return path.read_bytes()

    @staticmethod
    def _write_cache(path: Path, content: bytes, compress: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        if compress:
            with gzip.open(tmp, "wb") as fh:
                fh.write(content)
        else:
            tmp.write_bytes(content)
        tmp.replace(path)  # atomic on the same filesystem

    # --- public API -----------------------------------------------------------
    def get(self, url: str, *, compress: bool = False, force: bool = False) -> bytes:
        """Return the bytes at ``url``, using the disk cache when possible.

        Parameters
        ----------
        url:
            Absolute EDGAR URL.
        compress:
            Store the cached copy gzip-compressed (use for large text
            submissions to save disk).
        force:
            Ignore any cached copy and re-download.
        """
        cache_path = self._cache_path(url, compress)
        if cache_path.exists() and not force:
            return self._read_cached(cache_path)

        content = self._download(url)
        self._write_cache(cache_path, content, compress)
        return content

    def get_text(self, url: str, *, compress: bool = False, force: bool = False) -> str:
        return self.get(url, compress=compress, force=force).decode(
            "utf-8", errors="replace"
        )

    def get_optional(self, url: str, *, compress: bool = False, force: bool = False) -> bytes | None:
        """Like :meth:`get` but returns ``None`` on 404 and remembers the miss.

        Many SEC XBRL concepts do not exist for a given issuer (HTTP 404). We
        cache a small ``.absent`` marker so a resumed run does not re-request a
        concept known to be missing.
        """
        cache_path = self._cache_path(url, compress)
        absent = cache_path.with_name(cache_path.name + ".absent")
        if not force:
            if cache_path.exists():
                return self._read_cached(cache_path)
            if absent.exists():
                return None
        content = self._download(url, allow_missing=True)
        if content is None:
            absent.parent.mkdir(parents=True, exist_ok=True)
            absent.write_bytes(b"")
            return None
        self._write_cache(cache_path, content, compress)
        return content

    def is_cached(self, url: str, *, compress: bool = False) -> bool:
        return self._cache_path(url, compress).exists()

    def cached_at(self, url: str, *, compress: bool = False) -> datetime | None:
        """Local time the cached copy of ``url`` was written, or ``None``."""
        path = self._cache_path(url, compress)
        return datetime.fromtimestamp(path.stat().st_mtime) if path.exists() else None

    # --- download with retries ------------------------------------------------
    def _download(self, url: str, allow_missing: bool = False) -> bytes | None:
        last_exc: Exception | None = None
        # Separate budgets: throttle responses (429/5xx) are transient — wait
        # them out patiently with capped exponential backoff so a server-side
        # storm slows us rather than dropping filings. Connection/DNS errors are
        # usually a dead URL — cap those tightly so one bad filing can't stall.
        throttle_attempts = 0
        conn_errors = 0
        # (connect, read) timeouts: fail fast on connection/DNS stalls, but allow
        # a generous read window for large filings.
        timeout = (self.connect_timeout, self.timeout)
        while True:
            self._limiter.wait()
            try:
                resp = self._session.get(url, timeout=timeout)
                if resp.status_code == 200:
                    return resp.content
                if allow_missing and resp.status_code == 404:
                    return None
                if resp.status_code in (429, 500, 502, 503, 504):
                    throttle_attempts += 1
                    if throttle_attempts > self.max_retries:
                        logger.warning(
                            "Giving up on %s after %d throttle responses (last HTTP %s)",
                            url, throttle_attempts, resp.status_code,
                        )
                        break
                    wait = min(self.backoff * (2 ** (throttle_attempts - 1)), self.max_backoff)
                    logger.warning(
                        "HTTP %s for %s (throttle %d/%d) — backing off %.1fs",
                        resp.status_code, url, throttle_attempts, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
            except requests.RequestException as exc:  # connection/timeout
                last_exc = exc
                conn_errors += 1
                if conn_errors > self.conn_retries:
                    logger.warning(
                        "Giving up on %s after %d connection errors: %s",
                        url, conn_errors, exc,
                    )
                    break
                wait = self.conn_backoff * conn_errors
                logger.warning(
                    "Connection error for %s (%d/%d): %s — retrying in %.1fs",
                    url, conn_errors, self.conn_retries, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"Failed to download {url}") from last_exc
