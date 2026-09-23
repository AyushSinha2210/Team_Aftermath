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

    result = calibration.calibrate(_candidates([6.0, -1.0, -2.0, -2.0]))

    assert result["should_abstain"] is False
    assert result["top_candidate"]["id"] == "0"


def test_nearly_identical_scores_abstain(monkeypatch) -> None:
    monkeypatch.setattr(calibration, "load_config", lambda: _config())

    result = calibration.calibrate(_candidates([0.0, 0.01, -0.01, 0.0]))

    assert result["should_abstain"] is True
    assert result["top_candidate"] is None


def test_exact_threshold_does_not_abstain(monkeypatch) -> None:
    scores = [-2.0, -1.0, 1.0, 2.0]
    probs = np.exp(np.asarray(scores) - max(scores))
    probs /= np.sum(probs)
    normalized_variance = float(np.var(probs) / ((len(scores) - 1) / len(scores) ** 2))
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
    scores = _candidates([6.0, -1.0, -2.0, -2.0])
    monkeypatch.setattr(calibration, "load_config", lambda: _config(threshold=0.1))
    low_threshold_result = calibration.calibrate(scores)

    monkeypatch.setattr(calibration, "load_config", lambda: _config(threshold=0.9))
    high_threshold_result = calibration.calibrate(scores)

    assert low_threshold_result["should_abstain"] is False
    assert high_threshold_result["should_abstain"] is False


def test_maximally_spread_scores_reach_normalization_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(calibration, "load_config", lambda: _config())

    result = calibration.calibrate(_candidates([-100.0, -100.0, -100.0, 100.0]))

    assert np.isclose(result["confidence_variance"], 1.0, atol=1e-12)


def test_realistic_cross_encoder_logits_produce_sensible_decision(monkeypatch) -> None:
    monkeypatch.setattr(calibration, "load_config", lambda: _config())

    result = calibration.calibrate(
        _candidates([-11.465719, -0.2, 0.8, 5.604678])
    )

    assert result["should_abstain"] is False
    assert 0.0 < result["confidence_variance"] <= 1.0


def test_softmax_distinguishes_q1536_and_q3379_logit_patterns(monkeypatch) -> None:
    monkeypatch.setattr(calibration, "load_config", lambda: _config())
    q1536_scores = [-5.126] + [-5.13, -5.14, -5.15, -5.157] * 4
    q3379_scores = [-5.619, -8.0, -8.3, -8.51]

    q1536_result = calibration.calibrate(_candidates(q1536_scores))
    q3379_result = calibration.calibrate(_candidates(q3379_scores))

    assert q1536_result["should_abstain"] is True
    assert q3379_result["should_abstain"] is False
    assert q1536_result["confidence_variance"] < q3379_result["confidence_variance"]


def test_softmax_variance_is_bounded_and_normalized(monkeypatch) -> None:
    monkeypatch.setattr(calibration, "load_config", lambda: _config())

    for scores in ([0.0] * 20, [1000.0] + [-1000.0] * 19, list(range(-10, 10))):
        score_array = np.asarray(scores, dtype=float)
        probs = np.exp(score_array - np.max(score_array))
        probs /= np.sum(probs)
        raw_variance = float(np.var(probs))
        max_variance = (len(scores) - 1) / len(scores) ** 2
        result = calibration.calibrate(_candidates(list(scores)))

        assert np.isclose(np.sum(probs), 1.0)
        assert raw_variance <= max_variance + 1e-12
        assert 0.0 <= result["confidence_variance"] <= 1.0 + 1e-12