"""Stage 6 - FIXED loop - perceive -> decide -> act -> observe, with cross-cutting seams.
Implement decide() and synthesize() only. Security, grounding, PII, and tracing run via hooks at the
marked seams - do NOT inline them here."""
from __future__ import annotations
import re

from ..contracts import *  # noqa
from .. import hooks
from ..governance import pii
from ..llm import postprocess, prompts
from ..llm.client import LLM, content_words
from ..retrieval import retriever as rt
from . import guardrails, tools
from .memory import Memory

class Agent:
    """FIXED loop. Implement decide() (the policy) and synthesize() only."""
    def __init__(self, cfg: dict, retriever) -> None:
        self.cfg = cfg["agent"]; self.retriever = retriever; self.mem = Memory()
        self.full_cfg = cfg
        self.llm = LLM(cfg)
        tools.bind(retriever, cfg)    # tools get the retriever via module-level context

    def run(self, query_text: str) -> Answer:
        state = {"query": query_text, "obs": []}
        for _ in range(self.cfg["max_steps"]):
            hooks.run(hooks.ON_STEP, {"state": state})
            action = self.decide(state)                      # IMPLEMENT (policy)
            if action["tool"] == "stop":
                break
            hooks.run(hooks.ON_TOOL_CALL, {"action": action})   # guardrails/injection/trace
            result = self.act(action)                        # runs the tool via REGISTRY
            state["obs"].append(result); self.mem.add(result)
        hooks.run(hooks.BEFORE_ANSWER, {"state": state})     # grounding gate / PII redact
        ans = self.synthesize(state)                         # IMPLEMENT (grounded answer)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans})       # trace / metrics
        return ans

    def decide(self, state: dict) -> dict:
        """Evidence-gated re-search — the MANDATORY agentic behaviour (A3 gate, fail-closed).
        Read the last observation (top_score, k) and branch on the NUMBER, using retrieval.retriever:
          1. retrieve at k = cfg.retrieve.k
          2. if is_weak(chunks, cfg):  k2 = next_k(k, cfg)
               - k2 is not None -> retrieve AGAIN at the wider k2 (widen the net), then re-check
               - k2 is None (hit k_max) and still weak -> ABSTAIN ("insufficient evidence")
          3. else -> synthesize a grounded, cited answer
        Emit obs {"top_score": ..., "k": ...} on each step. A fixed retrieve->answer path is NOT agentic
        and caps the grade. Rule-based (baseline) or RL policy (Stage 7)."""
        obs = state["obs"]
        k0 = self.retriever.cfg["k"]
        if not obs:
            return {"tool": "retrieve", "args": {"query": state["query"], "k": k0}}
        last = obs[-1].payload
        top, k = last["top_score"], last["k"]
        state["last_obs"] = {"top_score": top, "k": k}
        if not last["weak"]:
            return {"tool": "stop"}                      # strong evidence -> answer
        k2 = rt.next_k(k, self.full_cfg)
        if k2 is None:                                   # hit k_max and still weak -> abstain
            state["abstain"] = True
            return {"tool": "stop"}
        query = self._reformulate(state["query"], attempt=len(obs))
        return {"tool": "retrieve", "args": {"query": query, "k": k2}}

    def _reformulate(self, query: str, attempt: int) -> str:
        """New wording for a re-search: LLM rewrite if a key is configured, else keyword query."""
        if self.llm.online:
            new = self.llm.complete(prompts.DECIDE.format(question=query), max_tokens=40).strip()
            if new:
                return new
        words = content_words(query)
        return " ".join(words) if words and " ".join(words) != query else query

    def act(self, action: dict) -> ToolResult:
        by_name = {t.name: t for t in tools.REGISTRY}
        cls = by_name.get(action["tool"])
        if cls is None:
            return ToolResult(ok=False, payload={"error": f"unknown tool {action['tool']}"})
        try:
            return cls()(**action.get("args", {}))
        except Exception as e:                           # a failing tool must not crash the loop
            return ToolResult(ok=False, payload={"error": f"{type(e).__name__}: {e}"})

    def synthesize(self, state: dict) -> Answer:
        """Grounded, cited answer; abstain if unsupported (no-hallucination)."""
        runs = [o.payload for o in state["obs"] if o.ok and "chunks" in o.payload]
        if state.get("abstain") or not runs:
            return postprocess.abstain_answer()
        best = max(runs, key=lambda p: p["top_score"])
        n_ctx = self.cfg.get("n_context", 4)
        # evidence is untrusted document text: neutralise injection phrases, collapse whitespace
        evid = [
            (c["id"], guardrails.sanitize(c["text"])) for c in best["chunks"][:n_ctx]
        ]
        prompt = prompts.SYNTHESIZE.format(
            evidence="\n".join(f"[{cid}] {txt}" for cid, txt in evid), question=state["query"]
        )
        raw = self.llm.complete(prompt)
        texts = dict(evid)
        cites = []
        for cid in dict.fromkeys(re.findall(r"\[([^\]]+)\]", raw)):
            if cid in texts:                             # drop hallucinated chunk ids
                cites.append(Citation(chunk_id=cid, span=self._span(texts[cid], raw)))
        ans = postprocess.format_answer(raw, cites)
        if ans.grounded:
            support = postprocess.support_score(ans.text, [texts[c.chunk_id] for c in ans.citations])
            if support < postprocess.MIN_SUPPORT:        # cited, but the text isn't in the evidence
                return postprocess.abstain_answer()
            ans.confidence = round(0.5 * min(1.0, best["top_score"]) + 0.5 * support, 3)
        ans.text = pii.redact(ans.text)
        return ans

    @staticmethod
    def _span(chunk_text: str, answer: str) -> tuple[int, int]:
        """Character span of the chunk sentence that best overlaps the answer."""
        aw = set(content_words(re.sub(r"\[[^\]]+\]", "", answer)))
        best, span = -1.0, (0, len(chunk_text))
        for m in re.finditer(r"[^.!?]+[.!?]?", chunk_text):
            s = m.group().strip()
            if not s:
                continue
            ov = len(aw & set(content_words(s))) / (len(aw) or 1)
            if ov > best:
                best, span = ov, (m.start() + len(m.group()) - len(m.group().lstrip()), m.end())
        return span
