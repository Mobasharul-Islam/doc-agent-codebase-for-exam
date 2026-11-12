"""Unit tests for data validation / versioning."""
import pytest

from doc_agent.contracts import Page
from doc_agent.data import validate, versioning


def test_validate_rejects_bad_input(tmp_path):
    with pytest.raises(AssertionError):
        validate.validate([])
    with pytest.raises(AssertionError):
        validate.validate([Page(id="a", image_path=str(tmp_path / "missing.png"), doc_id="d")])


def test_split_is_by_document():
    pages = [Page(id=f"{d}_{i}", image_path="x.png", doc_id=d) for d in "abcdefghij" for i in range(3)]
    splits = validate.split_by_doc(pages, seed=1)
    validate.assert_no_leakage(splits)
    assert sum(len(v) for v in splits.values()) == 30 and splits["val"] and splits["test"]


def test_floors_report():
    task = {"corpus": {"min_pages": 300, "min_words": 60000}}
    assert not validate.check_floors(10, 1000, task)["ok"]
    assert validate.check_floors(300, 60000, task)["ok"]


def test_snapshot_changes_with_content(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    v1 = versioning.snapshot(str(tmp_path))
    assert versioning.snapshot(str(tmp_path)) == v1
    (tmp_path / "a.txt").write_text("hello!")
    assert versioning.snapshot(str(tmp_path)) != v1
