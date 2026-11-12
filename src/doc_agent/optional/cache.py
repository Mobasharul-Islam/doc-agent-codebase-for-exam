"""OPTIONAL — embedding/retrieval/answer cache
Activate only if your data speciality or NFR requires it (e.g. a low-latency NFR). Off by default; CI does not require impl."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

_MEM: OrderedDict[str, object] = OrderedDict()
MAX_ITEMS = 512
DISK = Path("data/interim/cache.jsonl")   # JSON-serialisable values only; set DISK=None to disable


def get(key: str):
    """In-memory LRU first, then the on-disk log. Returns None on a miss."""
    if key in _MEM:
        _MEM.move_to_end(key)
        return _MEM[key]
    if DISK and DISK.exists():
        for line in DISK.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["k"] == key:
                _MEM[key] = rec["v"]
                return rec["v"]
    return None


def put(key: str, val) -> None:
    _MEM[key] = val
    _MEM.move_to_end(key)
    while len(_MEM) > MAX_ITEMS:
        _MEM.popitem(last=False)
    if DISK:
        DISK.parent.mkdir(parents=True, exist_ok=True)
        with DISK.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"k": key, "v": val}, default=str) + "\n")
