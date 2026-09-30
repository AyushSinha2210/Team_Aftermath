"""Integration tests for structural AST parsing and call graph on real-world validator.js repo."""

from __future__ import annotations

from pathlib import Path
import pytest

from ariadne.reranking.structural.ast_parser import parse_repo
from ariadne.reranking.structural.call_graph import (
    AmbiguousFunctionNameError,
    build_call_graph,
    calls_within_depth,
    resolve_call_path,
)

_VALIDATOR_JS_SRC = Path(__file__).resolve().parent / "external_repos" / "validator_js" / "src"


@pytest.fixture(scope="module")
def validator_js_functions():
    """Parses validator.js source directory once for the module test suite."""
    if not _VALIDATOR_JS_SRC.exists():
        pytest.skip(f"validator.js source not found at {_VALIDATOR_JS_SRC}")
    return parse_repo(str(_VALIDATOR_JS_SRC))


@pytest.fixture(scope="module")
def validator_js_call_graph(validator_js_functions):
    """Builds CallGraph instance from parsed validator.js functions."""
    return build_call_graph(validator_js_functions)


def test_parse_repo_finds_real_functions(validator_js_functions):
    """Asserts parse_repo parses validator.js without crashing and finds a substantial number of functions."""
    # validator.js has ~103 files in src/lib and src/lib/util, yielding >300 function nodes
    assert len(validator_js_functions) >= 200, f"Expected >= 200 functions, found {len(validator_js_functions)}"

    # Verify structural schema of parsed function dicts
    for fn in validator_js_functions:
        assert "name" in fn
        assert "short_name" in fn
        assert "file" in fn
        assert "start_line" in fn
        assert "end_line" in fn
        assert "calls" in fn
        assert ":" in fn["name"], f"Function name must be qualified as 'file:name', got {fn['name']}"

    # Verify key expected validator functions are present
    short_names = {fn["short_name"] for fn in validator_js_functions}
    for expected in ["isEmail", "isURL", "isByteLength", "assertString", "merge", "checkHost"]:
        assert expected in short_names, f"Expected '{expected}' to be parsed from validator.js"


def test_direct_cross_file_call_resolution(validator_js_call_graph):
    """Asserts resolve_call_path resolves real cross-file call relationships in validator.js."""
    # Real call 1: isEmail.js calls util/assertString.js
    path_assert_string = resolve_call_path(validator_js_call_graph, "isEmail", "assertString", max_depth=3)
    assert path_assert_string == ["isEmail.js:isEmail", "assertString.js:assertString"]

    # Real call 2: isEmail.js calls util/merge.js
    path_merge = resolve_call_path(validator_js_call_graph, "isEmail", "merge", max_depth=3)
    assert path_merge == ["isEmail.js:isEmail", "merge.js:merge"]

    # Real call 3: isEmail.js calls isByteLength.js
    path_byte_length = resolve_call_path(validator_js_call_graph, "isEmail", "isByteLength", max_depth=3)
    assert path_byte_length == ["isEmail.js:isEmail", "isByteLength.js:isByteLength"]

    # Real call 4: isURL.js calls util/checkHost.js
    path_check_host = resolve_call_path(validator_js_call_graph, "isURL", "checkHost", max_depth=3)
    assert path_check_host == ["isURL.js:isURL", "checkHost.js:checkHost"]


def test_calls_within_depth_reaches_helpers(validator_js_call_graph):
    """Asserts calls_within_depth identifies reachable functions across files."""
    reachable = calls_within_depth(validator_js_call_graph, "isEmail", max_depth=2)
    assert "assertString.js:assertString" in reachable
    assert "merge.js:merge" in reachable
    assert "isByteLength.js:isByteLength" in reachable
    assert len(reachable) >= 5


def test_real_world_ambiguity_detection_on_includes(validator_js_call_graph):
    """Asserts that real function-name collision ('includes') is detected and handled."""
    # Both includesArray.js and includesString.js define 'includes'
    assert "includes" in validator_js_call_graph.short_to_qualified
    candidates = validator_js_call_graph.short_to_qualified["includes"]
    assert len(candidates) == 2
    assert set(candidates) == {"includesArray.js:includes", "includesString.js:includes"}

    # Ambiguity must be registered in ambiguous_calls
    assert "includes" in validator_js_call_graph.ambiguous_calls

    # Calling resolve_call_path with bare name 'includes' must raise AmbiguousFunctionNameError
    with pytest.raises(AmbiguousFunctionNameError) as exc_info:
        resolve_call_path(validator_js_call_graph, "isBoolean", "includes")
    assert "is ambiguous across files" in str(exc_info.value)
    assert "includesArray.js:includes" in str(exc_info.value)
    assert "includesString.js:includes" in str(exc_info.value)

    # Qualified target name resolves deterministically without ambiguity
    qualified_path = resolve_call_path(
        validator_js_call_graph,
        "isBoolean.js:isBoolean",
        "includesString.js:includes",
        max_depth=3,
    )
    assert qualified_path == ["isBoolean.js:isBoolean", "includesString.js:includes"]
