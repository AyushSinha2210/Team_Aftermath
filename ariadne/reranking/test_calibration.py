from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ariadne.reranking import calibration


def _candidates(scores: List[float]) -> List[Dict[str, Any]]:
    return [
        {"id": str(index), "text": f"snippet {index}", "rerank_score": score}
        for index, score in enumerate(scores)
    ]


def _config(threshold: float = 0.35) -> Dict[str, Any]:
    return {"reranking": {"rerank_top_k": 20, "calibration_threshold": threshold}}


def test_dominant_top_score_does_not_abstain(monkeypatch) -> None:
    monkeypatch.setattr(calibration, "load_config", lambda: _config())

    result = calibration.calibrate(_candidates([0.95, 0.2, 0.1, 0.1]))

    assert result["should_abstain"] is False
    assert result["top_candidate"]["id"] == "0"


def test_nearly_identical_scores_abstain(monkeypatch) -> None:
    monkeypatch.setattr(calibration, "load_config", lambda: _config())

    result = calibration.calibrate(_candidates([0.5, 0.51, 0.49, 0.5]))

    assert result["should_abstain"] is True
    assert result["top_candidate"] is None


def test_exact_threshold_does_not_abstain(monkeypatch) -> None:
    scores = [0.0, 0.5, 0.5, 1.0]
    normalized_variance = float(np.var(scores) / 0.25)
    monkeypatch.setattr(
        calibration,
        "load_config",
        lambda: _config(threshold=normalized_variance),
    )

    result = calibration.calibrate(_candidates(scores))

    assert result["confidence_variance"] == normalized_variance
    assert result["should_abstain"] is False


def test_empty_and_single_item_inputs_abstain(monkeypatch) -> None:
    monkeypatch.setattr(calibration, "load_config", lambda: _config())

    empty_result = calibration.calibrate([])
    single_result = calibration.calibrate(_candidates([0.9]))

    assert empty_result == {
        "should_abstain": True,
        "confidence_variance": 0.0,
        "top_candidate": None,
    }
    assert single_result["should_abstain"] is True
    assert single_result["top_candidate"] is None


def test_threshold_from_config_changes_decision(monkeypatch) -> None:
    scores = _candidates([0.9, 0.2, 0.1, 0.1])
    monkeypatch.setattr(calibration, "load_config", lambda: _config(threshold=0.1))
    low_threshold_result = calibration.calibrate(scores)

    monkeypatch.setattr(calibration, "load_config", lambda: _config(threshold=0.9))
    high_threshold_result = calibration.calibrate(scores)

    assert low_threshold_result["should_abstain"] is False
    assert high_threshold_result["should_abstain"] is True


def test_maximally_spread_scores_reach_normalization_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(calibration, "load_config", lambda: _config())

    result = calibration.calibrate(_candidates([0.0, 0.0, 1.0, 1.0]))

    assert np.isclose(result["confidence_variance"], 1.0)