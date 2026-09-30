"""AST-enriched structural preprocessing for semantic code retrieval.

Extracts function signatures, docstrings, and call dependencies into a
structured prefix header that aligns bi-encoder attention with interface semantics.
"""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional, Set


def extract_python_metadata(code: str) -> Dict[str, Any]:
    """Extracts structural interface metadata from Python code using the AST.

    Args:
        code: Python source code snippet.

    Returns:
        Dict with keys:
            - 'functions': list of function signatures.
            - 'classes': list of class names.
            - 'docstring': extracted top-level or first function docstring.
            - 'calls': list of invoked function names.
    """
    metadata: Dict[str, Any] = {
        "functions": [],
        "classes": [],
        "docstring": "",
        "calls": [],
    }

    if not code or not code.strip():
        return metadata

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fallback to lightweight regex if snippet is an incomplete block
        return _regex_fallback_metadata(code)

    doc = ast.get_docstring(tree)
    if doc:
        metadata["docstring"] = doc.split("\n\n")[0].strip().replace("\n", " ")

    called_names: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [arg.arg for arg in node.args.args if arg.arg != "self"]
            sig = f"{node.name}({', '.join(args)})"
            metadata["functions"].append(sig)

            # If no module docstring, use first function docstring
            if not metadata["docstring"]:
                fn_doc = ast.get_docstring(node)
                if fn_doc:
                    metadata["docstring"] = fn_doc.split("\n\n")[0].strip().replace("\n", " ")

        elif isinstance(node, ast.ClassDef):
            metadata["classes"].append(node.name)

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)

    metadata["calls"] = sorted(list(called_names))
    return metadata


def _regex_fallback_metadata(code: str) -> Dict[str, Any]:
    """Lightweight regex extractor for incomplete code fragments."""
    fn_matches = re.findall(r"def\s+([a-zA-Z_0-9]+)\s*\((.*?)\)", code)
    functions = [f"{name}({args})" for name, args in fn_matches]

    class_matches = re.findall(r"class\s+([a-zA-Z_0-9]+)", code)

    # Extract first triple-quote string as docstring
    doc_match = re.search(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', code, re.DOTALL)
    docstring = ""
    if doc_match:
        raw_doc = doc_match.group(1) or doc_match.group(2) or ""
        docstring = raw_doc.strip().split("\n\n")[0].replace("\n", " ")

    return {
        "functions": functions,
        "classes": class_matches,
        "docstring": docstring,
        "calls": [],
    }


def format_structural_prefix(metadata: Dict[str, Any]) -> str:
    """Formats metadata into a concise, semantic code prefix.

    Args:
        metadata: Metadata dictionary from extract_python_metadata.

    Returns:
        Formatted header string with trailing newline, or empty string if empty.
    """
    lines: List[str] = []

    if metadata.get("docstring"):
        lines.append(f"# SUMMARY: {metadata['docstring']}")

    if metadata.get("functions"):
        lines.append(f"# SIGNATURE: {', '.join(metadata['functions'][:3])}")

    if metadata.get("classes"):
        lines.append(f"# CLASSES: {', '.join(metadata['classes'][:3])}")

    if metadata.get("calls"):
        lines.append(f"# CALLS: {', '.join(metadata['calls'][:5])}")

    if not lines:
        return ""

    return "\n".join(lines) + "\n"


def enrich_code_snippet(code: str) -> str:
    """Prefixes code with extracted structural and interface metadata.

    Args:
        code: Raw Python code snippet.

    Returns:
        Enriched code string.
    """
    if not code:
        return ""

    meta = extract_python_metadata(code)
    prefix = format_structural_prefix(meta)
    return prefix + code


def enrich_corpus(corpus: Dict[str, str]) -> Dict[str, str]:
    """Enriches all code snippets in a corpus dictionary.

    Args:
        corpus: Mapping of doc_id -> code snippet.

    Returns:
        Mapping of doc_id -> enriched code snippet.
    """
    return {doc_id: enrich_code_snippet(code) for doc_id, code in corpus.items()}
