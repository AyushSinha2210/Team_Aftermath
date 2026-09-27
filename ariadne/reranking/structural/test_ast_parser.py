from pathlib import Path

from ariadne.reranking.structural.ast_parser import parse_js_file


FIXTURE = Path(__file__).parent / "test_fixtures" / "sample.js"


def test_parse_js_file_finds_functions_and_direct_calls() -> None:
    functions = parse_js_file(str(FIXTURE))

    assert functions == [
        {
            "name": "sample.js:greet",
            "short_name": "greet",
            "file": "sample.js",
            "start_line": 1,
            "end_line": 4,
            "calls": ["formatName", "logMessage"],
        },
        {
            "name": "sample.js:formatName",
            "short_name": "formatName",
            "file": "sample.js",
            "start_line": 6,
            "end_line": 8,
            "calls": [],
        },
        {
            "name": "sample.js:loadUser",
            "short_name": "loadUser",
            "file": "sample.js",
            "start_line": 10,
            "end_line": 13,
            "calls": ["fetchUser", "greet"],
        },
        {
            "name": "sample.js:saveUser",
            "short_name": "saveUser",
            "file": "sample.js",
            "start_line": 16,
            "end_line": 19,
            "calls": ["validateUser", "persistUser"],
        },
        {
            "name": "sample.js:wrapper",
            "short_name": "wrapper",
            "file": "sample.js",
            "start_line": 22,
            "end_line": 24,
            "calls": ["loadUser"],
        },
    ]