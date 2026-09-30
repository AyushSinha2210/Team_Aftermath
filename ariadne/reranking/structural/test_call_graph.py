from ariadne.reranking.structural.call_graph import (
	calls_within_depth,
	resolve_call_path,
)


CALL_GRAPH = {
	"A": ["B"],
	"B": ["C"],
	"C": ["D"],
	"D": [],
}


def test_resolve_call_path_finds_shortest_two_hop_chain() -> None:
	assert resolve_call_path(CALL_GRAPH, "A", "C", max_depth=2) == ["A", "B", "C"]


def test_resolve_call_path_returns_none_outside_depth() -> None:
	assert resolve_call_path(CALL_GRAPH, "A", "D", max_depth=2) is None


def test_calls_within_depth_returns_reachable_functions() -> None:
	assert calls_within_depth(CALL_GRAPH, "A", max_depth=1) == {"B"}
	assert calls_within_depth(CALL_GRAPH, "A", max_depth=2) == {"B", "C"}


def test_cyclic_graph_terminates() -> None:
	cyclic_graph = {"A": ["B"], "B": ["A"]}

	assert resolve_call_path(cyclic_graph, "A", "A", max_depth=10) == ["A"]
	assert calls_within_depth(cyclic_graph, "A", max_depth=10) == {"B"}


def test_unknown_source_and_target_are_handled() -> None:
	assert resolve_call_path(CALL_GRAPH, "unknown", "A") is None
	assert resolve_call_path(CALL_GRAPH, "A", "unknown") is None
	assert calls_within_depth(CALL_GRAPH, "unknown") == set()