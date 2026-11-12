"""MLOps — model registry"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..contracts import *  # noqa

REGISTRY_FILE = Path("models/registry.json")


def register(component: str, path: str, metrics: dict) -> str:
    """Version a checkpoint + its metrics: records sha256 of the file and returns 'component:vN'."""
    entries = json.loads(REGISTRY_FILE.read_text()) if REGISTRY_FILE.exists() else []
    n = 1 + sum(e["component"] == component for e in entries)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    version = f"{component}:v{n}"
    entries.append({"version": version, "component": component, "path": str(path),
                    "sha256": digest, "metrics": metrics,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(entries, indent=1))
    return version
