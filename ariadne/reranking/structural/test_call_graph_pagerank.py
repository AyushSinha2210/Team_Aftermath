from ariadne.reranking.structural.call_graph import compute_call_graph_pagerank


def test_compute_call_graph_pagerank_hub_highest():
	# node 'db:connect' is called by multiple modules (hub)
	graph = {
		"auth:login": ["db:connect"],
		"user:getProfile": ["db:connect"],
		"billing:charge": ["db:connect", "stripe:api"],
		"db:connect": [],
		"stripe:api": [],
	}

	scores = compute_call_graph_pagerank(graph)

	assert "db:connect" in scores
	# db:connect should have the highest centrality score
	assert scores["db:connect"] > scores["auth:login"]
	assert scores["db:connect"] > scores["user:getProfile"]

	# Sum of probabilities should be close to 1
	assert abs(sum(scores.values()) - 1.0) < 1e-4


def test_compute_call_graph_pagerank_empty():
	assert compute_call_graph_pagerank({}) == {}
