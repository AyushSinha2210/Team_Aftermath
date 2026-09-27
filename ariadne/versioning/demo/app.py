"""Ariadne Search Demo Application (owned by Person D).

Demonstrates the final locked Config A pipeline: Dense retrieval alone
(no BM25 fusion, no cross-encoder reranking) per RERANK_CARD.md recommendation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np
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

# Cached embeddings for the demo corpus
_CORPUS_EMBEDDINGS: Optional[np.ndarray] = None


def get_corpus_embeddings() -> np.ndarray:
    """Lazily encodes and caches the demo corpus embeddings using the bi-encoder."""
    global _CORPUS_EMBEDDINGS
    if _CORPUS_EMBEDDINGS is None:
        texts = [doc["code"] for doc in DEMO_CORPUS]
        _CORPUS_EMBEDDINGS = encode(texts)
    return _CORPUS_EMBEDDINGS


def search(query: str, top_k: int = 5) -> str:
    """Performs Config A (dense alone) retrieval against the demo corpus.

    Args:
        query: Natural language search string.
        top_k: Number of ranked matches to return.

    Returns:
        Formatted Markdown string showing ranked code matches and scores.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return "⚠️ *Please enter a search query above to find matching code snippets.*"

    query_embedding = encode([cleaned_query])[0]
    corpus_embeddings = get_corpus_embeddings()

    # Dense retrieval alone: cosine similarity over normalized embeddings
    cosine_scores = np.dot(corpus_embeddings, query_embedding)
    ranked_indices = np.argsort(cosine_scores)[::-1][:top_k]

    result_blocks = [
        "### 🏆 Top Matches — Config A (Dense Retrieval Alone)\n",
        "*Pipeline: bi-encoder semantic search without fusion or cross-encoder reranking.*\n",
    ]

    for rank, idx in enumerate(ranked_indices, start=1):
        doc = DEMO_CORPUS[idx]
        score = float(cosine_scores[idx])
        result_blocks.append(
            f"#### **#{rank}** — `{doc['id']}` ({doc['title']}) | **Cosine Score: {score:.4f}**\n"
            f"```python\n{doc['code']}\n```\n"
        )

    return "\n".join(result_blocks)


def create_app() -> gr.Blocks:
    """Builds and returns the Gradio UI Blocks application."""
    config = load_config()
    demo_cfg = config.get("versioning", {}).get("demo", {})
    theme_name = demo_cfg.get("theme", "dark")
    theme = gr.themes.Soft(primary_hue="blue") if theme_name == "dark" else gr.themes.Default()

    with gr.Blocks(title="Ariadne Code Search Demo", theme=theme) as app:
        gr.Markdown(
            "# 🚀 Ariadne Code Intelligence Demo\n"
            "**Theme 01: Agentic Code Intelligence** | *PRISM GenAI Hackathon*\n\n"
            "Powered by the validated **Config A** pipeline: Dense retrieval alone using "
            "the fine-tuned bi-encoder checkpoint (`best_biencoder`), per `RERANK_CARD.md` findings."
        )

        with gr.Row():
            query_input = gr.Textbox(
                label="Enter Search Query",
                placeholder="e.g. find minimum temperature, logarithmic search, check palindrome...",
                lines=2,
                scale=4,
            )
            submit_btn = gr.Button("Search", variant="primary", scale=1)

        results_output = gr.Markdown(label="Ranked Results")

        submit_btn.click(fn=search, inputs=[query_input], outputs=[results_output])
        query_input.submit(fn=search, inputs=[query_input], outputs=[results_output])

        gr.Examples(
            examples=[
                ["logarithmic search in an ordered list"],
                ["find minimum temperature value from space separated integers"],
                ["check if string reads the same forwards and backwards"],
                ["sort a list by recursively splitting and merging halves"],
                ["find two numbers in list that sum to target"],
                ["traverse a graph level by level from starting node"],
            ],
            inputs=[query_input],
        )

    return app


# Module-level app instance constructed on import
app = create_app()


if __name__ == "__main__":
    config = load_config()
    demo_cfg = config.get("versioning", {}).get("demo", {})
    host = demo_cfg.get("host", "0.0.0.0")
    port = int(demo_cfg.get("port", 7860))
    app.launch(server_name=host, server_port=port)
