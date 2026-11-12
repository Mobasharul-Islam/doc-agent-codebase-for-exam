"""Unit tests for ingest."""
from pathlib import Path

from doc_agent.ingest import enhance, loader, preprocess


def test_load_pages(kb):
    pages = loader.load_pages(kb)
    assert len(pages) == 10 and {p.doc_id for p in pages} == {"almanac_1431", "tide_tables", "harvest_ledger"}


def test_preprocess_and_enhance(kb):
    pages = loader.load_pages(kb)
    pre = preprocess.run(pages, kb)
    assert all(Path(p.image_path).exists() for p in pre)
    enh = enhance.run(pre, kb)
    assert [p.id for p in enh] == [p.id for p in pre]
    assert (Path(kb["data"]["interim_dir"]) / "page_meta.json").exists()


def test_denoiser_forward_shape():
    import torch

    net = enhance.DenoiseNet()
    x = torch.rand(2, 1, 32, 48)
    assert net(x).shape == x.shape
