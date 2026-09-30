import numpy as np

from ariadne.retrieval.pipeline import HybridPipeline


def dummy_encoder(texts):
	# Deterministic 4D embeddings
	res = []
	for t in texts:
		h = hash(t) % 1000
		v = np.array([float(h % 7), float((h // 7) % 5), float((h // 35) % 3), 1.0])
		res.append(v / np.linalg.norm(v))
	return np.vstack(res)


def test_hybrid_pipeline_retrieve_with_hyde():
	corpus = {
		"doc1": "def authenticate(user, password):\n    return check_hash(password)",
		"doc2": "def compute_metrics(y_true, y_pred):\n    return accuracy_score(y_true, y_pred)",
	}

	pipeline = HybridPipeline(corpus, encoder=dummy_encoder)

	# Retrieve without HyDE
	res_standard = pipeline.retrieve("authenticate user", k=2, use_hyde=False)
	assert len(res_standard) == 2

	# Retrieve with HyDE
	res_hyde = pipeline.retrieve("authenticate user", k=2, use_hyde=True)
	assert len(res_hyde) == 2
	assert "fusion_score" in res_hyde[0]
	assert "dense_score" in res_hyde[0]
