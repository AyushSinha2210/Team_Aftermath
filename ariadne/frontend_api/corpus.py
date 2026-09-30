"""Function-bounded metadata for the checked-in JavaScript demo corpus."""
from pathlib import Path


def load_functions(repo: Path) -> dict[str, dict]:
    from ariadne.reranking.structural.ast_parser import parse_js_file

    documents = {}
    for path in sorted(repo.rglob("*.js")):
        lines = path.read_text(encoding="utf-8").splitlines()
        relative = path.relative_to(repo).as_posix()
        for function in parse_js_file(str(path), include_methods=True):
            start, end = function["start_line"], function["end_line"]
            identifier = f"{relative}:{start}:{function['short_name']}"
            documents[identifier] = {
                "id": identifier,
                "file": relative,
                "start_line": start,
                "end_line": end,
                "title": function["short_name"],
                "code": "\n".join(lines[start - 1:end]),
            }
    if not documents:
        raise ValueError("No JavaScript functions were found in the configured corpus")
    return documents
