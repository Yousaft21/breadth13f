"""Configuration loading and lightweight logging setup.

The whole pipeline is config-driven: a single YAML file (``config/config.yaml``)
holds the sample window, EDGAR access parameters and the point-in-time filing
lag. Loading returns a plain nested dict plus a couple of resolved absolute
paths so downstream modules never hard-code locations.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels up from this file (src/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@dataclass
class Config:
    """Parsed configuration with resolved absolute paths."""

    raw: dict[str, Any]
    repo_root: Path = REPO_ROOT
    paths: dict[str, Path] = field(default_factory=dict)

    # --- convenience accessors ------------------------------------------------
    @property
    def edgar(self) -> dict[str, Any]:
        return self.raw["edgar"]

    @property
    def window(self) -> dict[str, Any]:
        return self.raw["window"]

    @property
    def pit(self) -> dict[str, Any]:
        return self.raw["pit"]

    def path(self, key: str) -> Path:
        return self.paths[key]

    @property
    def user_agent(self) -> str:
        """User-Agent string, overridable via the EDGAR_USER_AGENT env var."""
        return os.environ.get("EDGAR_USER_AGENT", self.edgar["user_agent"])


def load_dotenv(path: str | Path = REPO_ROOT / ".env") -> None:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into the environment.

    Secrets (e.g. ``OPENFIGI_API_KEY``) live in a gitignored ``.env`` rather than
    in the tracked config. Existing environment variables take precedence, so an
    explicit shell export overrides the file. Missing ``.env`` is a no-op.
    """
    path = Path(path)
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load YAML config and resolve project directories to absolute paths."""
    load_dotenv()
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    proj = raw["project"]
    paths = {
        "data": REPO_ROOT / proj["data_dir"],
        "raw": REPO_ROOT / proj["raw_dir"],
        "interim": REPO_ROOT / proj["interim_dir"],
        "processed": REPO_ROOT / proj["processed_dir"],
        "results": REPO_ROOT / proj["results_dir"],
        "raw_edgar": REPO_ROOT / proj["raw_dir"] / "edgar",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    return Config(raw=raw, paths=paths)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logging once and return the package logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("breadth13f")
