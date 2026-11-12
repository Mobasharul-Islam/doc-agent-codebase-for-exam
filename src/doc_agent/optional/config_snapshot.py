"""OPTIONAL — snapshot exact run config for reproducibility+
Activate only if your data speciality or NFR requires it (repro+). Off by default; CI does not require impl."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path


def snapshot(cfg: dict, out: str) -> None:
    """Write cfg + git commit + library versions + corpus version to `out` (JSON)."""
    def sh(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""

    libs = {}
    for lib in ("torch", "numpy", "faiss", "sentence_transformers", "transformers"):
        try:
            libs[lib] = __import__(lib).__version__
        except Exception:
            libs[lib] = None
    ledger = Path("data/versions.jsonl")
    corpus = json.loads(ledger.read_text().splitlines()[-1])["version"] if ledger.exists() else None
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": sh("git", "rev-parse", "HEAD"),
           "python": platform.python_version(), "libs": libs, "corpus_version": corpus, "config": cfg}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rec, indent=1, default=str))
