from __future__ import annotations

from typing import Any, Dict, List

from ariadne.reranking import cross_encoder


class FakeCrossEncoder:
    """Deterministic CPU-only stand-in for the sentence-transformers model."""

    def predict(self, pairs: List[tuple[str, str]]) -> List[float]:
        scores = []
        for _, text in pairs:
            if "binary_search" in text:
                scores.append(0.95)
            elif "quicksort" in text:
                scores.append(0.65)
            else:
                scores.append(0.10)
        return scores


def _candidate(candidate_id: str, text: str, fusion_score: float) -> Dict[str, Any]:
    return {"id": candidate_id, "text": text, "fusion_score": fusion_score}


def test_rerank_empty_candidates_returns_empty_list() -> None:
    assert cross_encoder.rerank("any query", []) == []


def test_rerank_preserves_fusion_score(monkeypatch) -> None:
    monkeypatch.setattr(cross_encoder, "load_cross_encoder", lambda: FakeCrossEncoder())
    candidates = [
        _candidate("quicksort", "def quicksort(arr): return sorted(arr)", 0.21),
        _candidate("binary", "def binary_search(arr, target): return -1", 0.42),
    ]

    reranked = cross_encoder.rerank("logarithmic search in sorted array", candidates)

    assert [candidate["fusion_score"] for candidate in reranked] == [0.42, 0.21]
    assert all("rerank_score" in candidate for candidate in reranked)


def test_rerank_does_not_reorder_candidates_beyond_cutoff(monkeypatch) -> None:
    monkeypatch.setattr(cross_encoder, "load_cross_encoder", lambda: FakeCrossEncoder())
    candidates = [
        _candidate(str(index), f"def fibonacci(n): return {index}", float(index))
        for index in range(22)
    ]
    candidates[0]["text"] = "def fibonacci(n): return 0"
    candidates[1]["text"] = "def binary_search(arr, target): return -1"

    reranked = cross_encoder.rerank("logarithmic search in sorted array", candidates)

    assert [candidate["id"] for candidate in reranked[20:]] == ["20", "21"]
    assert all("rerank_score" not in candidate for candidate in reranked[20:])


def test_rerank_prefers_binary_search_for_search_query(monkeypatch) -> None:
    monkeypatch.setattr(cross_encoder, "load_cross_encoder", lambda: FakeCrossEncoder())
    candidates = [
        _candidate(
            "quicksort",
            "def quicksort(arr): return arr if len(arr) <= 1 else quicksort(arr[1:])",
            0.31,
        ),
        _candidate(
            "binary_search",
            "def binary_search(arr, target): l, r = 0, len(arr)-1; return -1",
            0.29,
        ),
        _candidate(
            "fibonacci",
            "def fibonacci(n): a, b = 0, 1; return a",
            0.28,
        ),
    ]

    reranked = cross_encoder.rerank("logarithmic search in sorted array", candidates)

    assert reranked[0]["id"] == "binary_search"
    assert reranked[0]["rerank_score"] > reranked[2]["rerank_score"]