"""Unit tests for CascadeRouter."""

from __future__ import annotations

from ariadne.reranking.cascade_router import CascadeRouter


def test_cascade_router_empty_candidates():
    router = CascadeRouter(confidence_threshold=0.05)
    result = router.route("test query", [])
    assert result["candidates"] == []
    assert result["decision"] == "fast_path"


def test_cascade_router_single_candidate():
    router = CascadeRouter(confidence_threshold=0.05)
    candidate = {"id": "c1", "fusion_score": 0.9}
    result = router.route("test query", [candidate])
    assert result["candidates"] == [candidate]
    assert result["decision"] == "fast_path"


def test_cascade_router_high_confidence_bypasses_rerank():
    called = []

    def mock_rerank(q, cands, score_cache=None):
        called.append(True)
        return cands

    router = CascadeRouter(confidence_threshold=0.05, rerank_fn=mock_rerank)
    candidates = [
        {"id": "c1", "fusion_score": 0.85},
        {"id": "c2", "fusion_score": 0.70},  # margin = 0.15 >= 0.05
    ]
    result = router.route("query", candidates)
    assert result["decision"] == "fast_path"
    assert result["margin"] >= 0.14
    assert len(called) == 0  # reranker was NOT called


def test_cascade_router_low_confidence_escalates_to_rerank():
    called = []

    def mock_rerank(q, cands, score_cache=None):
        called.append(True)
        return [{"id": "c2"}, {"id": "c1"}]

    router = CascadeRouter(confidence_threshold=0.05, rerank_fn=mock_rerank)
    candidates = [
        {"id": "c1", "fusion_score": 0.80},
        {"id": "c2", "fusion_score": 0.78},  # margin = 0.02 < 0.05
    ]
    result = router.route("query", candidates)
    assert result["decision"] == "escalated"
    assert len(called) == 1  # reranker was escalated
    assert result["candidates"][0]["id"] == "c2"
