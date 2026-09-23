"""Inspect the raw score range produced by the configured cross-encoder."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ariadne.reranking.cross_encoder import load_cross_encoder


SNIPPETS = {
    "quicksort": (
        "def quicksort(arr): return arr if len(arr) <= 1 else "
        "quicksort(arr[1:])"
    ),
    "binary_search": (
        "def binary_search(arr, target): l, r = 0, len(arr)-1; "
        "return -1"
    ),
    "fibonacci": "def fibonacci(n): a, b = 0, 1; return a",
}

PAIR_QUERIES = [
    "logarithmic search in sorted array",
    "find an item in an ordered list efficiently",
    "divide and conquer sorting algorithm",
    "sort an array around a pivot",
    "compute the Fibonacci sequence",
    "generate a recursive number sequence",
    "search for a target with left and right bounds",
    "partition an unsorted array for sorting",
    "calculate the nth Fibonacci number",
    "binary lookup with logarithmic complexity",
    "recursive quick sort implementation",
    "iterative search in sorted data",
    "return Fibonacci values up to n",
    "rearrange values using a pivot element",
    "locate a target in sorted integers",
    "sequence where each value sums prior values",
    "average case divide and conquer sort",
    "halve the search interval repeatedly",
    "in-place style sorting procedure",
    "dynamic programming for Fibonacci",
]

PAIR_SNIPPET_NAMES = [
    "binary_search",
    "binary_search",
    "quicksort",
    "quicksort",
    "fibonacci",
    "fibonacci",
    "binary_search",
    "quicksort",
    "fibonacci",
    "binary_search",
    "quicksort",
    "binary_search",
    "fibonacci",
    "quicksort",
    "binary_search",
    "fibonacci",
    "quicksort",
    "binary_search",
    "quicksort",
    "fibonacci",
]


def _build_pairs() -> List[Tuple[str, str]]:
    """Builds 20 varied query/snippet pairs from the reranker test fixtures."""
    return [
        (query, SNIPPETS[snippet_name])
        for query, snippet_name in zip(PAIR_QUERIES, PAIR_SNIPPET_NAMES)
    ]


def _print_histogram(scores: np.ndarray, bucket_count: int = 10) -> None:
    """Prints counts in equal-width buckets spanning the observed score range."""
    minimum = float(np.min(scores))
    maximum = float(np.max(scores))
    if minimum == maximum:
        print(f"  [{minimum:.6f}, {maximum:.6f}] : {len(scores)}")
        return

    edges = np.linspace(minimum, maximum, bucket_count + 1)
    counts, _ = np.histogram(scores, bins=edges)
    for index, count in enumerate(counts):
        right_bracket = "]" if index == bucket_count - 1 else ")"
        print(
            f"  [{edges[index]:.6f}, {edges[index + 1]:.6f}{right_bracket} : {count}"
        )


def main() -> None:
    """Loads the CPU cross-encoder and reports its raw score distribution."""
    pairs = _build_pairs()
    model = load_cross_encoder()
    scores = np.asarray(model.predict(pairs), dtype=float)

    print(f"Scored {len(scores)} query/candidate pairs")
    print(f"Raw score min: {float(np.min(scores)):.6f}")
    print(f"Raw score max: {float(np.max(scores)):.6f}")
    print("Histogram (equal-width buckets over observed range):")
    _print_histogram(scores)


if __name__ == "__main__":
    main()
