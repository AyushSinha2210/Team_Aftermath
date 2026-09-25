"""Offline contract tests against the installed MTEB interface."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("MTEB_CACHE", str(Path(__file__).resolve().parents[2] / ".cache" / "mteb"))
mteb = pytest.importorskip("mteb")

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.models_protocols import SearchProtocol

from ariadne.retrieval.encoder import PrePostPipelineEncoder
from ariadne.retrieval.mteb_search import HybridSearchModel
from ariadne.retrieval.run_mteb_eval import run_evaluation


def fake_encode(texts: list[str], **kwargs) -> np.ndarray:
    return np.asarray(
        [[float("binary" in text.lower()), float("sort" in text.lower())] for text in texts],
        dtype=np.float32,
    )


def test_abs_encoder_batches() -> None:
    encoder = PrePostPipelineEncoder(embed=fake_encode)
    assert isinstance(encoder, AbsEncoder)
    vectors = encoder.encode(
        [{"text": ["binary search", "sort"]}],
        task_metadata=SimpleNamespace(name="AppsRetrieval"),
        hf_split="test",
        hf_subset="default",
    )
    assert vectors.shape == (2, 2)
    assert vectors.dtype == np.float32


def test_hybrid_search_protocol_returns_mteb_scores() -> None:
    model = HybridSearchModel(encoder=fake_encode)
    assert isinstance(model, SearchProtocol)
    model.index(
        [{"id": "a", "body": "binary_search"}, {"id": "b", "body": "quick sort"}],
        task_metadata=None,
        hf_split="test",
        hf_subset="default",
        encode_kwargs={},
        num_proc=None,
    )
    scores = model.search(
        [{"id": "q", "text": "binary search"}],
        task_metadata=None,
        hf_split="test",
        hf_subset="default",
        top_k=2,
        encode_kwargs={},
        num_proc=None,
    )
    assert list(scores) == ["q"]
    assert list(scores["q"])[0] == "a"
    assert all(isinstance(score, float) for score in scores["q"].values())


def test_runner_writes_only_completed_test_result(monkeypatch) -> None:
    class FakeResult:
        task_name = "AppsRetrieval"
        scores = {"test": [{"ndcg_at_10": 0.5}]}

        def model_dump(self, *, mode):
            return {"task_name": self.task_name, "scores": self.scores}

    monkeypatch.setattr(mteb, "get_task", lambda **kwargs: object())
    monkeypatch.setattr(
        mteb, "evaluate", lambda *args, **kwargs: SimpleNamespace(task_results=[FakeResult()])
    )
    output = Path(__file__).resolve().parents[2] / ".cache" / "contract-test-results.json"
    output.parent.mkdir(exist_ok=True)
    try:
        run_evaluation("hybrid", output)
        assert json.loads(output.read_text())["scores"]["test"][0]["ndcg_at_10"] == 0.5

        monkeypatch.setattr(
            mteb, "evaluate", lambda *args, **kwargs: SimpleNamespace(task_results=[])
        )
        output.unlink()
        with pytest.raises(RuntimeError):
            run_evaluation("hybrid", output)
        assert not output.exists()
    finally:
        output.unlink(missing_ok=True)
