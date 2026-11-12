"""Extract exact source snippets (so the book never drifts from the code)."""
import ast, sys
from pathlib import Path

ROOT = Path(r"C:\Users\mobas\Desktop\Study-Materials\L4\4-1\DL_TF\doc-agent-template\doc-agent-starter")
OUT = Path(__file__).parent / "snips"
OUT.mkdir(exist_ok=True)
S = "src/doc_agent/"

M = {  # key: (file, symbol or None for whole file)
 "contracts": (S+"contracts.py", None), "hooks": (S+"hooks.py", None), "wiring": (S+"wiring.py", None),
 "pipeline": (S+"pipeline.py", None), "config": (S+"config.py", None), "settings": (S+"settings.py", None),
 "loader_paths": (S+"ingest/loader.py", "data_paths"), "loader_docid": (S+"ingest/loader.py", "_doc_id"),
 "loader_load": (S+"ingest/loader.py", "load_pages"),
 "val_validate": (S+"data/validate.py", "validate"), "val_split": (S+"data/validate.py", "split_by_doc"),
 "val_leak": (S+"data/validate.py", "assert_no_leakage"), "val_floors": (S+"data/validate.py", "check_floors"),
 "versioning": (S+"data/versioning.py", "snapshot"),
 "pre_skew": (S+"ingest/preprocess.py", "estimate_skew"), "pre_quality": (S+"ingest/preprocess.py", "quality_stats"),
 "pre_run": (S+"ingest/preprocess.py", "run"),
 "enh_net": (S+"ingest/enhance.py", "DenoiseNet"), "enh_degrade": (S+"ingest/enhance.py", "degrade"),
 "enh_train": (S+"ingest/enhance.py", "Enhancer.train"), "enh_img": (S+"ingest/enhance.py", "Enhancer.enhance_image"),
 "enh_run": (S+"ingest/enhance.py", "run"),
 "lay_lines": (S+"vision/layout.py", "_lines"), "lay_detect": (S+"vision/layout.py", "detect"),
 "ocr_reader": (S+"vision/ocr.py", "Reader"), "ocr_transcribe": (S+"vision/ocr.py", "transcribe"),
 "pii": (S+"governance/pii.py", None), "chunk": (S+"index/chunk.py", "split"), "embed": (S+"index/embed.py", None),
 "store": (S+"index/store.py", None), "retriever": (S+"retrieval/retriever.py", None), "rerank": (S+"retrieval/rerank.py", "rerank"),
 "llm_client": (S+"llm/client.py", "LLM"), "llm_words": (S+"llm/client.py", "content_words"),
 "prompts": (S+"llm/prompts.py", None), "post": (S+"llm/postprocess.py", None),
 "agent_init": (S+"agent/agent.py", "Agent.__init__"), "agent_run": (S+"agent/agent.py", "Agent.run"),
 "agent_decide": (S+"agent/agent.py", "Agent.decide"), "agent_reform": (S+"agent/agent.py", "Agent._reformulate"),
 "agent_act": (S+"agent/agent.py", "Agent.act"), "agent_synth": (S+"agent/agent.py", "Agent.synthesize"),
 "agent_span": (S+"agent/agent.py", "Agent._span"),
 "tools_ctx": (S+"agent/tools.py", "bind"), "tools_retrieve": (S+"agent/tools.py", "Retrieve"),
 "tools_extract": (S+"agent/tools.py", "Extract"), "tools_cite": (S+"agent/tools.py", "Cite"),
 "tools_calc": (S+"agent/tools.py", "Calculator"), "tools_eval": (S+"agent/tools.py", "_eval"),
 "tools_registry": (S+"agent/tools.py", "Tool"),
 "guard": (S+"agent/guardrails.py", None), "memory": (S+"agent/memory.py", None),
 "hitl": (S+"agent/hitl.py", None), "hitl_store": (S+"agent/hitl_store.py", None),
 "trace": (S+"logging_conf.py", "register"),
 "metrics": (S+"eval/metrics.py", None), "calib": (S+"eval/calibration.py", None),
 "judge": (S+"eval/judge.py", "judge"), "abl_eval": (S+"eval/ablation.py", "evaluate"),
 "abl_run": (S+"eval/ablation.py", "run"), "succ": ("grading_kit/success_check.py", None),
 "env": (S+"rl/env.py", None), "policy": (S+"rl/policy.py", None), "train_rl": (S+"rl/train_rl.py", None),
 "rlvr": (S+"rl/rlvr.py", None), "datamodule": (S+"training/datamodule.py", None),
 "lit": (S+"training/lit_modules.py", None), "train": (S+"training/train.py", None), "adapt": (S+"training/adapt.py", None),
 "tracking": (S+"mlops/tracking.py", None), "registry": (S+"mlops/registry.py", None), "monitor": (S+"mlops/monitor.py", "check_drift"),
 "api": (S+"serve/api.py", None), "ui_launch": (S+"serve/ui.py", "launch"),
 "opt_auth": (S+"optional/api_security.py", None), "opt_dp": (S+"optional/dp.py", None),
 "conftest": ("tests/conftest.py", None), "t_cross": ("tests/test_crosscutting.py", None),
 "t_agent": ("tests/test_agent.py", None), "demo": ("scripts/make_demo_corpus.py", "render"),
 "fair": (S+"eval/fairness.py", "audit"), "robust": (S+"eval/robustness.py", "stress"),
 "interp": (S+"eval/interpret.py", "explain"), "config_yaml": ("configs/config.yaml", None),
 "structure_test": ("tests/test_structure.py", None),
}
MAP = {"\u2014": "--", "\u2013": "-", "\u2192": "->", "\u2265": ">=", "\u2264": "<=", "\u2018": "'", "\u2019": "'",
       "\u201c": '"', "\u201d": '"', "\u00d7": "x", "\u2026": "...", "\u2248": "~", "\u2260": "!="}

def clean(t):
    for a, b in MAP.items(): t = t.replace(a, b)
    return t.encode("ascii", "replace").decode()

def find(tree, sym):
    parts = sym.split(".")
    nodes = tree.body
    node = None
    for p in parts:
        node = next(n for n in nodes if getattr(n, "name", None) == p)
        nodes = getattr(node, "body", [])
    return node

for key, (f, sym) in M.items():
    text = (ROOT / f).read_text(encoding="utf-8")
    if sym is None:
        out = text
    else:
        node = find(ast.parse(text), sym)
        lines = text.splitlines()
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
        out = "\n".join(lines[start:node.end_lineno]) + "\n"
        if "." in sym:   # method: dedent
            import textwrap; out = textwrap.dedent(out)
    (OUT / f"{key}.txt").write_text(clean(out), encoding="ascii")
print(len(M), "snippets")
