"""Ariadne Search Demo Application (owned by Person D).

Demonstrates the final locked Config A pipeline: Dense retrieval alone
(no BM25 fusion, no cross-encoder reranking) per RERANK_CARD.md recommendation.
Built with Streamlit for reliable, lightweight UI rendering.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure ariadne is on sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
_workspace_root = _repo_root.parent
for p in [str(_workspace_root), str(_repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import streamlit as st
import yaml

from ariadne.finetuning.embedder import encode


def load_config() -> Dict[str, Any]:
    """Loads configuration dictionary from config.yaml."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_path = repo_root / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Canonical code corpus reused from test fixtures and demo scripts
DEMO_CORPUS: List[Dict[str, str]] = [
    {
        "id": "quicksort",
        "title": "Quicksort Partitioning",
        "code": (
            "def quicksort(values):\n"
            "    if len(values) <= 1:\n"
            "        return values\n"
            "    pivot = values[0]\n"
            "    return (quicksort([v for v in values[1:] if v < pivot])\n"
            "            + [pivot]\n"
            "            + quicksort([v for v in values[1:] if v >= pivot]))"
        ),
    },
    {
        "id": "binary_search",
        "title": "Binary Search on Sorted Array",
        "code": (
            "def binary_search(values, target):\n"
            "    left, right = 0, len(values) - 1\n"
            "    while left <= right:\n"
            "        middle = (left + right) // 2\n"
            "        if values[middle] == target:\n"
            "            return middle\n"
            "        if values[middle] < target:\n"
            "            left = middle + 1\n"
            "        else:\n"
            "            right = middle - 1\n"
            "    return -1"
        ),
    },
    {
        "id": "fibonacci",
        "title": "Fibonacci Number Generation",
        "code": (
            "def fibonacci(count):\n"
            "    first, second = 0, 1\n"
            "    sequence = []\n"
            "    for _ in range(count):\n"
            "        sequence.append(first)\n"
            "        first, second = second, first + second\n"
            "    return sequence"
        ),
    },
    {
        "id": "lowest_temperature",
        "title": "Lowest Temperature Extractor",
        "code": (
            "def lowest_temperature(values):\n"
            "    readings = (int(value) for value in values.split())\n"
            "    return min(readings, default=None)"
        ),
    },
    {
        "id": "palindrome",
        "title": "Palindrome Checker",
        "code": (
            "def is_palindrome(value):\n"
            "    normalized = [char.lower() for char in value if char.isalnum()]\n"
            "    return normalized == normalized[::-1]"
        ),
    },
    {
        "id": "two_sum",
        "title": "Two Sum Problem",
        "code": (
            "def two_sum(values, target):\n"
            "    positions = {}\n"
            "    for index, value in enumerate(values):\n"
            "        complement = target - value\n"
            "        if complement in positions:\n"
            "            return positions[complement], index\n"
            "        positions[value] = index\n"
            "    return None"
        ),
    },
    {
        "id": "breadth_first_search",
        "title": "Breadth First Search (BFS)",
        "code": (
            "def breadth_first_search(graph, start):\n"
            "    queue = [start]\n"
            "    visited = {start}\n"
            "    while queue:\n"
            "        node = queue.pop(0)\n"
            "        for neighbor in graph[node]:\n"
            "            if neighbor not in visited:\n"
            "                visited.add(neighbor)\n"
            "                queue.append(neighbor)\n"
            "    return visited"
        ),
    },
    {
        "id": "merge_sort",
        "title": "Merge Sort",
        "code": (
            "def merge_sort(values):\n"
            "    if len(values) <= 1:\n"
            "        return values\n"
            "    middle = len(values) // 2\n"
            "    left = merge_sort(values[:middle])\n"
            "    right = merge_sort(values[middle:])\n"
            "    return merge(left, right)"
        ),
    },
]


@st.cache_resource(show_spinner="Encoding demo code corpus...")
def get_corpus_embeddings() -> np.ndarray:
    """Encodes and caches the demo corpus embeddings using the bi-encoder."""
    texts = [doc["code"] for doc in DEMO_CORPUS]
    return encode(texts)


def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Performs Config A (dense alone) retrieval against the demo corpus.

    Args:
        query: Natural language search string.
        top_k: Number of ranked matches to return.

    Returns:
        List of result dictionaries containing rank, id, title, score, and code.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    query_embedding = encode([cleaned_query])[0]
    corpus_embeddings = get_corpus_embeddings()

    # Dense retrieval alone: cosine similarity over normalized embeddings
    cosine_scores = np.dot(corpus_embeddings, query_embedding)
    ranked_indices = np.argsort(cosine_scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        doc = DEMO_CORPUS[idx]
        score = float(cosine_scores[idx])
        results.append(
            {
                "rank": rank,
                "id": doc["id"],
                "title": doc["title"],
                "score": score,
                "code": doc["code"],
            }
        )
    return results


def main() -> None:
    """Renders the Streamlit user interface."""
    st.set_page_config(
        page_title="Ariadne Code Search Demo",
        page_icon="🚀",
        layout="wide",
    )

    st.title("🚀 Ariadne Code Intelligence Demo")
    st.caption("**Theme 01: Agentic Code Intelligence** | *PRISM GenAI Hackathon*")

    st.info(
        "Powered by the validated **Config A** pipeline: Dense retrieval alone using "
        "the fine-tuned bi-encoder checkpoint (`best_biencoder`), per `RERANK_CARD.md` findings."
    )

    # Example query buttons for quick testing
    st.markdown("##### Quick Examples")
    example_cols = st.columns(4)
    examples = [
        "logarithmic search in an ordered list",
        "find minimum temperature value",
        "check if string reads same backwards",
        "find two numbers that sum to target",
    ]
    selected_example = None
    for col, ex in zip(example_cols, examples):
        if col.button(ex, use_container_width=True):
            selected_example = ex

    with st.form("search_form", clear_on_submit=False):
        default_val = selected_example or ""
        query = st.text_input(
            "Enter Natural Language Search Query:",
            value=default_val,
            placeholder="e.g. logarithmic search, minimum temperature, palindrome...",
        )
        submitted = st.form_submit_button("Search Codebase", type="primary")

    active_query = query if submitted else selected_example

    if active_query and active_query.strip():
        with st.spinner("Searching with fine-tuned bi-encoder..."):
            results = search(active_query, top_k=5)

        if not results:
            st.warning("No matches found.")
        else:
            st.markdown(
                f"### 🏆 Top {len(results)} Ranked Matches (Config A — Dense Alone)"
            )
            for r in results:
                with st.expander(
                    f"#{r['rank']} — {r['title']} (`{r['id']}`) — Cosine Score: {r['score']:.4f}",
                    expanded=(r["rank"] <= 2),
                ):
                    st.code(r["code"], language="python")


if __name__ == "__main__":
    main()
