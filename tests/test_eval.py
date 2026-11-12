"""Unit tests for eval metrics, calibration, RL pieces."""
import numpy as np
import pytest

from doc_agent.contracts import Answer, Citation
from doc_agent.eval import calibration, metrics
from doc_agent.rl import env, rlvr


def test_ocr_f1_and_recall():
    assert metrics.ocr_f1("a b c", "a b c") == 1.0
    assert metrics.ocr_f1("a b", "c d") == 0.0
    assert metrics.recall_at_k(["x", "y", "z"], ["y", "q"], 2) == 0.5


def test_groundedness_rules():
    good = Answer(text="t", citations=[Citation(chunk_id="c", span=(0, 1))], grounded=True, confidence=0.9)
    abst = Answer(text="n", citations=[], grounded=False, confidence=0.0)
    bad = Answer(text="t", citations=[], grounded=True, confidence=0.9)
    assert metrics.groundedness(good) == metrics.groundedness(abst) == 1.0
    assert metrics.groundedness(bad) == 0.0


def test_ece_perfect_and_overconfident():
    assert metrics.ece([0.5] * 4, [1, 0, 1, 0]) == pytest.approx(0.0)
    assert metrics.ece([0.95] * 4, [0, 0, 0, 0]) > 0.9


def test_temperature_scaling_softens_overconfident_logits():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 400)
    logits = rng.normal(0, 1, (400, 3)) * 6
    logits[np.arange(400), labels] += 4 * (rng.random(400) < 0.6)   # right only ~60% of the time
    assert calibration.temperature_scale(logits, labels).T > 1.0


def test_subgroup_gap():
    assert metrics.subgroup_gap({"a": [1.0, 1.0], "b": [0.5, 0.5]}) == pytest.approx(0.5)


def test_rlvr_reward_is_exact_match():
    assert rlvr.verifiable_reward("4.2 metres.", "4.2 Metres") == 1.0
    assert rlvr.verifiable_reward("about 4 metres", "4.2 metres") == 0.0


def test_env_step_contract():
    e = env.ToolSelectionEnv()
    obs, _ = e.reset(seed=0)
    assert obs.shape == (3,)
    _, r, done, _, _ = e.step(env.ABSTAIN)
    assert done and r in (0.3, -0.3)
