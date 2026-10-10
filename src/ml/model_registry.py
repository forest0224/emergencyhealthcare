"""
src/ml/model_registry.py
=========================
Versioned model registry backed by a JSON manifest and file-system artifacts.

Public API
----------
class ModelRegistry
    register(model_path, metrics, tag) -> str
    load(version_id=None) -> (model, meta)
    promote(version_id, stage='production') -> None
    rollback(to_version_id) -> None
    list_versions() -> pd.DataFrame

Usage
-----
>>> reg = ModelRegistry("ml_runs")
>>> vid = reg.register("ml_runs/best_model.pkl", {"macro_f1": 0.82}, "v1")
>>> reg.promote(vid)
>>> model, meta = reg.load()
"""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

RANDOM_STATE = 42
_REGISTRY_FILE = "registry.json"
_VERSIONS_DIR  = "versions"


class ModelRegistry:
    """Versioned model registry for ED-Triage models.

    Persists a JSON manifest at {registry_dir}/registry.json and copies
    artifacts into {registry_dir}/versions/{version_id}/.

    Args:
        registry_dir: Root directory for registry data (default "ml_runs").

    Example:
        >>> reg = ModelRegistry("ml_runs")
        >>> versions = reg.list_versions()
    """

    def __init__(self, registry_dir: str = "ml_runs") -> None:
        """Initialize registry, creating directories and manifest if needed.

        Args:
            registry_dir: Path to the root ml_runs directory.

        Example:
            >>> reg = ModelRegistry("ml_runs")
        """
        self._root     = Path(registry_dir)
        self._ver_dir  = self._root / _VERSIONS_DIR
        self._manifest = self._root / _REGISTRY_FILE
        self._root.mkdir(parents=True, exist_ok=True)
        self._ver_dir.mkdir(parents=True, exist_ok=True)

        if not self._manifest.exists():
            self._write_manifest({"versions": [], "production": None})

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _read_manifest(self) -> dict:
        """Read and parse the registry JSON manifest.

        Returns:
            Parsed manifest dict.

        Example:
            >>> m = reg._read_manifest()
        """
        with open(self._manifest, "r") as fh:
            return json.load(fh)

    def _write_manifest(self, data: dict) -> None:
        """Write the registry JSON manifest atomically.

        Args:
            data: Full manifest dict to write.

        Example:
            >>> reg._write_manifest(manifest)
        """
        tmp = self._manifest.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=2)
        tmp.replace(self._manifest)

    @staticmethod
    def _version_id(tag: str, ts: str) -> str:
        """Generate a short version ID from tag + timestamp.

        Args:
            tag: Human-readable tag (e.g. "v1", "retrain-2024-01").
            ts:  ISO timestamp string.

        Returns:
            version_id string like "v1-20240101T120000".

        Example:
            >>> vid = ModelRegistry._version_id("v1", "2024-01-01T12:00:00")
        """
        slug = tag.replace(" ", "_").replace("/", "-")[:20]
        ts_short = ts[:16].replace(":", "").replace("-", "").replace("T", "T")
        return f"{slug}-{ts_short}"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def register(
        self,
        model_path: str,
        metrics: dict,
        tag: str = "untagged",
    ) -> str:
        """Register a new model version, copying artifacts into the registry.

        Copies model_path and an optional feature_meta.pkl (sibling file)
        into versions/{version_id}/ and records metrics in the manifest.

        Args:
            model_path: Path to the trained model pickle.
            metrics:    Dict of evaluation metrics (e.g. {"macro_f1": 0.82}).
            tag:        Human-readable label (e.g. "xgboost-v2").

        Returns:
            version_id string.

        Example:
            >>> vid = reg.register("ml_runs/best_model.pkl",
            ...                    {"macro_f1": 0.82}, tag="xgb-v1")
        """
        ts  = datetime.now(tz=timezone.utc).isoformat()
        vid = self._version_id(tag, ts)

        ver_path = self._ver_dir / vid
        ver_path.mkdir(parents=True, exist_ok=True)

        src = Path(model_path)
        if not src.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        shutil.copy2(src, ver_path / "model.pkl")

        # Copy feature_meta.pkl if it exists next to the model
        meta_src = src.parent / "feature_meta.pkl"
        if meta_src.exists():
            shutil.copy2(meta_src, ver_path / "feature_meta.pkl")

        # Compute SHA-256 of the model file
        sha256 = hashlib.sha256(src.read_bytes()).hexdigest()[:16]

        manifest = self._read_manifest()
        entry: dict[str, Any] = {
            "version_id": vid,
            "tag":        tag,
            "timestamp":  ts,
            "stage":      "staging",
            "metrics":    metrics,
            "model_sha":  sha256,
            "artifacts":  {
                "model":       str(ver_path / "model.pkl"),
                "feature_meta": str(ver_path / "feature_meta.pkl")
                               if (ver_path / "feature_meta.pkl").exists()
                               else None,
            },
        }
        manifest["versions"].append(entry)
        self._write_manifest(manifest)

        print(f"[registry] Registered version_id={vid}  tag={tag}  sha={sha256}")
        return vid

    # ------------------------------------------------------------------ #

    def load(
        self,
        version_id: str | None = None,
    ) -> tuple[Any, dict | None]:
        """Load a model (and optional meta) from the registry.

        Args:
            version_id: Version to load. None loads the current
                        production version.

        Returns:
            Tuple (model, meta) where meta is the feature_meta dict or None.

        Raises:
            ValueError: If no production version is set and version_id is None.
            KeyError:   If version_id is not found in the registry.

        Example:
            >>> model, meta = reg.load()           # production
            >>> model, meta = reg.load("v1-2024")  # specific version
        """
        manifest = self._read_manifest()

        if version_id is None:
            version_id = manifest.get("production")
            if version_id is None:
                raise ValueError(
                    "No production version set. Call promote() first or "
                    "supply an explicit version_id."
                )

        entry = next(
            (v for v in manifest["versions"] if v["version_id"] == version_id),
            None,
        )
        if entry is None:
            raise KeyError(f"version_id {version_id!r} not found in registry.")

        model = joblib.load(entry["artifacts"]["model"])

        meta: dict | None = None
        meta_path = entry["artifacts"].get("feature_meta")
        if meta_path and Path(meta_path).exists():
            meta = joblib.load(meta_path)

        print(f"[registry] Loaded version_id={version_id}  stage={entry['stage']}")
        return model, meta

    # ------------------------------------------------------------------ #

    def promote(
        self,
        version_id: str,
        stage: str = "production",
    ) -> None:
        """Promote a version to a given stage (default 'production').

        Only one version can be 'production' at a time.

        Args:
            version_id: Version to promote.
            stage:      Target stage string (e.g. 'production', 'staging').

        Example:
            >>> reg.promote("v1-20240101T120000")
        """
        manifest = self._read_manifest()
        entry = next(
            (v for v in manifest["versions"] if v["version_id"] == version_id),
            None,
        )
        if entry is None:
            raise KeyError(f"version_id {version_id!r} not found in registry.")

        if stage == "production":
            manifest["production"] = version_id

        entry["stage"] = stage
        self._write_manifest(manifest)
        print(f"[registry] Promoted version_id={version_id} → stage={stage}")

    # ------------------------------------------------------------------ #

    def rollback(self, to_version_id: str) -> None:
        """Roll back production to a previous version.

        Args:
            to_version_id: version_id to restore as production.

        Example:
            >>> reg.rollback("v1-20240101T120000")
        """
        self.promote(to_version_id, stage="production")
        print(f"[registry] Rolled back production to {to_version_id}")

    # ------------------------------------------------------------------ #

    def list_versions(self) -> pd.DataFrame:
        """List all registered model versions with their metrics.

        Returns:
            DataFrame sorted by timestamp descending with columns:
            version_id, tag, timestamp, stage, model_sha, and all
            metric keys flattened.

        Example:
            >>> df = reg.list_versions()
            >>> print(df[["version_id", "stage", "macro_f1"]])
        """
        manifest = self._read_manifest()
        rows = []
        for v in manifest["versions"]:
            row: dict[str, Any] = {
                "version_id": v["version_id"],
                "tag":        v["tag"],
                "timestamp":  v["timestamp"],
                "stage":      v["stage"],
                "model_sha":  v.get("model_sha", ""),
            }
            for k, val in v.get("metrics", {}).items():
                row[k] = val
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("timestamp", ascending=False).reset_index(drop=True)
        return df
