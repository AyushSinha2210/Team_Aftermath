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

from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode

from ariadne.retrieval.encoder import PrePostPipelineEncoder
from ariadne.retrieval.mteb_search import HybridSearchModel
from ariadne.retrieval.run_mteb_eval import run_evaluation


def fake_encode(texts: list[str], **kwargs) -> np.ndarray:
    return np.asarray(
        [[float("binary" in text.lower()), float("sort" in text.lower())] for text in texts],
        dtype=np.float32,
    )


def test_encoder_protocol_contract() -> None:
    encoder = PrePostPipelineEncoder(embed=fake_encode)
    assert isinstance(encoder, Encoder)
    assert isinstance(encoder, EncoderWithQueryCorpusEncode)

    # Encode sentences
    vectors = encoder.encode(["binary search", "sort"])
    assert vectors.shape == (2, 2)
    assert vectors.dtype == np.float32

    # Encode queries
    q_vectors = encoder.encode_queries(["binary search", "sort"])
    assert q_vectors.shape == (2, 2)
    assert q_vectors.dtype == np.float32

    # Encode corpus
    c_vectors = encoder.encode_corpus([
        {"title": "Search", "text": "binary search"},
        {"title": "Sort", "text": "quick sort"},
    ])
    assert c_vectors.shape == (2, 2)
    assert c_vectors.dtype == np.float32


def test_hybrid_search_returns_mteb_scores() -> None:
    calls: list[int] = []

    def tracked_encode(texts: list[str]) -> np.ndarray:
        calls.append(len(texts))
        return fake_encode(texts)

    model = HybridSearchModel(encoder=tracked_encode)
    model.index(
        [{"id": "a", "text": "binary_search"}, {"id": "b", "text": "quick sort"}],
        task_metadata=None,
        hf_split="test",
        hf_subset="default",
        encode_kwargs={},
        num_proc=None,
    )
    scores = model.search(
        [{"id": "q", "text": "binary search"}, {"id": "q2", "text": "sort"}],
        task_metadata=None,
        hf_split="test",
        hf_subset="default",
        top_k=2,
        encode_kwargs={},
        num_proc=None,
    )
    assert list(scores) == ["q", "q2"]
    assert list(scores["q"])[0] == "a"
    assert model.pipeline.corpus["a"] == "binary_search"
    assert all(isinstance(score, float) for score in scores["q"].values())
    assert calls == [2, 2]


def test_runner_writes_only_completed_test_result(monkeypatch) -> None:
    class FakeResult:
        task_name = "AppsRetrieval"
        scores = {"test": [{"ndcg_at_10": 0.5}]}

        def model_dump(self, *, mode):
            return {"task_name": self.task_name, "scores": self.scores}

    monkeypatch.setattr(mteb, "get_task", lambda **kwargs: object(), raising=False)
    monkeypatch.setattr(
        mteb, "evaluate", lambda *args, **kwargs: SimpleNamespace(task_results=[FakeResult()]), raising=False
    )
    output = Path(__file__).resolve().parents[2] / ".cache" / "contract-test-results.json"
    output.parent.mkdir(exist_ok=True)
    try:
        run_evaluation("dense", output)
        assert json.loads(output.read_text())["scores"]["test"][0]["ndcg_at_10"] == 0.5

        monkeypatch.setattr(
            mteb, "evaluate", lambda *args, **kwargs: SimpleNamespace(task_results=[]), raising=False
        )
        output.unlink()
        with pytest.raises(RuntimeError):
            run_evaluation("dense", output)
        assert not output.exists()
    finally:
        output.unlink(missing_ok=True)
