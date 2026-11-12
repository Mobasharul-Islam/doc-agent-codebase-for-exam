"""Shared fixture: a throw-away knowledge base built from the synthetic demo corpus."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def kb(tmp_path_factory):
    """Build the whole KB (ingest -> index) in a temp dir; returns the cfg that points at it."""
    from doc_agent import config, logging_conf, pipeline
    from doc_agent.agent import hitl_store

    base = tmp_path_factory.mktemp("kb")
    spec = importlib.util.spec_from_file_location("demo", ROOT / "scripts" / "make_demo_corpus.py")
    demo = importlib.util.module_from_spec(spec)
    sys.modules["demo"] = demo
    spec.loader.exec_module(demo)
    demo.main(str(base / "raw"))

    cfg = config.load(ROOT / "configs" / "config.yaml")
    cfg["data"] = {"raw_dir": str(base / "raw"), "interim_dir": str(base / "interim"),
                   "index_dir": str(base / "index"), "models_dir": str(base / "models"),
                   "reports_dir": str(base / "reports"), "trace_path": str(base / "run.jsonl")}
    cfg["retrieve"]["rerank"] = False            # keep tests offline/fast
    mp = pytest.MonkeyPatch()
    mp.setattr(logging_conf, "TRACE_PATH", base / "run.jsonl")
    mp.setattr(hitl_store, "DB_PATH", base / "hitl.db")
    pipeline.build_knowledge_base(cfg)
    yield cfg
    mp.undo()
