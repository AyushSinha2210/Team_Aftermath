"""Validate both retrieval adapter modes with a deterministic encoder."""
from pathlib import Path
import uuid

import numpy as np
import pytest

from ariadne.frontend_api.service import BusyError, ROOT, SearchService


@pytest.fixture
def services(monkeypatch):
    import ariadne.finetuning.embedder as embedder
    import ariadne.versioning.incremental_index as incremental

    def encode(texts):
        return np.array([[float(any(word in text.lower() for word in terms)) for terms in [('auth', 'session', 'verify'), ('volume', 'audio'), ('bluetooth', 'device')]] for text in texts], dtype=np.float32)

    monkeypatch.setattr(embedder, 'encode', encode)
    monkeypatch.setattr(incremental, 'encode', encode)
    cache = ROOT / '.cache' / f'test-service-{uuid.uuid4().hex}.pkl'
    dense = SearchService(cache=cache)
    hybrid = SearchService(mode='hybrid')
    yield dense, hybrid
    cache.unlink(missing_ok=True)


@pytest.mark.parametrize('mode', [0, 1])
def test_modes_return_real_function_metadata(services, mode):
    service = services[mode]
    response = service.search('verify session authentication', 5)
    assert response['status'] == 'success'
    assert response['mode'] == ['dense', 'hybrid'][mode]
    for result in response['results']:
        assert result['id'] in service.documents
        assert 'dense' in result['sources']
        assert np.isfinite(result['score'])
        if mode == 1:
            assert 'sparse' in result['sources']
            assert result['sparse_score'] is not None


@pytest.mark.parametrize('mode', [0, 1])
def test_nonsense_abstains_without_fabricated_results(services, mode):
    response = services[mode].search('zxqvv zzz', 5)
    assert response['status'] == 'empty' and response['results'] == []


def test_busy_guard_releases_after_inference(services):
    service = services[0]
    service.lock.acquire()
    try:
        with pytest.raises(BusyError):
            service.search('auth', 1)
    finally:
        service.lock.release()
    assert service.search('auth', 1)['results']
