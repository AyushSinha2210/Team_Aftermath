from ariadne.reranking.structural.call_graph import (
	CallGraph,
	resolve_call_path,
	resolve_call_path_bidirectional,
)


def test_bidirectional_bfs_direct_and_multihop():
	graph = CallGraph({
		"app.js:main": ["auth.js:login"],
		"auth.js:login": ["crypto.js:hashPassword", "db.js:getUser"],
		"crypto.js:hashPassword": ["native.js:sha256"],
		"native.js:sha256": [],
		"db.js:getUser": [],
	})

	# Single hop
	path1 = resolve_call_path_bidirectional(graph, "app.js:main", "auth.js:login", max_depth=3)
	assert path1 == ["app.js:main", "auth.js:login"]

	# Multi-hop: main -> login -> hashPassword -> sha256
	path2 = resolve_call_path_bidirectional(graph, "app.js:main", "native.js:sha256", max_depth=3)
	assert path2 == ["app.js:main", "auth.js:login", "crypto.js:hashPassword", "native.js:sha256"]

	# Path matches unidirectional BFS
	path_uni = resolve_call_path(graph, "app.js:main", "native.js:sha256", max_depth=3)
	assert path2 == path_uni


def test_bidirectional_bfs_unreachable_or_exceeds_depth():
	graph = CallGraph({
		"a": ["b"],
		"b": ["c"],
		"c": ["d"],
		"d": [],
		"orphan": [],
	})

	# Unreachable
	assert resolve_call_path_bidirectional(graph, "a", "orphan", max_depth=5) is None

	# Depth limit exceeded
	assert resolve_call_path_bidirectional(graph, "a", "d", max_depth=2) is None
