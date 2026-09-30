"""Integration test for multi-file demo repo call graph resolution using parse_repo."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest

from ariadne.reranking.structural.ast_parser import parse_repo
from ariadne.reranking.structural.call_graph import (
    AmbiguousFunctionNameError,
    CallGraph,
    build_call_graph,
    calls_within_depth,
    resolve_call_path,
)


@pytest.fixture
def demo_repo_call_graph() -> CallGraph:
    """Parses demo_repo via parse_repo() and builds a qualified call graph."""
    repo_dir = Path(__file__).resolve().parent / "test_fixtures" / "demo_repo"
    parsed_functions = parse_repo(str(repo_dir))
    return build_call_graph(parsed_functions)


def test_resolve_call_path_login_to_check_expiry(demo_repo_call_graph: CallGraph) -> None:
    """Asserts resolve_call_path() correctly finds login -> validateToken -> checkExpiry (2 hops)."""
    path = resolve_call_path(
        demo_repo_call_graph,
        source="login",
        target="checkExpiry",
        max_depth=2,
    )
    assert path == ["auth.js:login", "auth.js:validateToken", "auth.js:checkExpiry"]
    # Verify bare name equivalence
    assert [p.split(":")[-1] for p in path] == ["login", "validateToken", "checkExpiry"]


def test_calls_within_depth_login_direct(demo_repo_call_graph: CallGraph) -> None:
    """Asserts calls_within_depth(source="login", max_depth=1) returns exactly direct calls."""
    direct_calls = calls_within_depth(
        demo_repo_call_graph,
        source="login",
        max_depth=1,
    )
    assert direct_calls == {"auth.js:validateCredentials", "auth.js:validateToken"}
    assert {f.split(":")[-1] for f in direct_calls} == {"validateCredentials", "validateToken"}


def test_resolve_call_path_nonexistent_returns_none(demo_repo_call_graph: CallGraph) -> None:
    """Asserts a query for a path that should NOT exist returns None."""
    path = resolve_call_path(
        demo_repo_call_graph,
        source="login",
        target="destroySession",
        max_depth=5,
    )
    assert path is None


def test_ambiguous_function_name_handling(demo_repo_call_graph: CallGraph) -> None:
    """Confirms system detects ambiguous bare names across files and resolves via qualified names."""
    # Both auth.js and utils.js define 'validate'
    assert "validate" in demo_repo_call_graph.ambiguous_calls
    candidates = set(demo_repo_call_graph.ambiguous_calls["validate"])
    assert candidates == {"auth.js:validate", "utils.js:validate"}

    # Querying bare name 'validate' explicitly raises AmbiguousFunctionNameError
    with pytest.raises(AmbiguousFunctionNameError) as exc_info_calls:
        calls_within_depth(demo_repo_call_graph, source="validate")
    assert "matches multiple functions" in str(exc_info_calls.value)

    with pytest.raises(AmbiguousFunctionNameError) as exc_info_path:
        resolve_call_path(demo_repo_call_graph, source="validate", target="checkExpiry")
    assert "matches multiple functions" in str(exc_info_path.value)

    # Qualified names resolve unambiguously
    auth_calls = calls_within_depth(demo_repo_call_graph, source="auth.js:validate")
    assert auth_calls == set()

    utils_calls = calls_within_depth(demo_repo_call_graph, source="utils.js:validate")
    assert utils_calls == set()
