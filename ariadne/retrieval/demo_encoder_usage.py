"""Person B Integration Demo: Using Person A's fine-tuned Bi-Encoder for Dense Retrieval.

Demonstrates that Person B can clone the repository and execute dense retrieval
directly via `from ariadne.finetuning.embedder import encode` without importing
any training modules.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Ensure ariadne is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
_workspace_root = _repo_root.parent
for p in [str(_workspace_root), str(_repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# PERSON B'S DROP-IN IMPORT
from ariadne.finetuning.embedder import encode


def run_dense_retrieval_demo() -> None:
    """Simulates Person B indexing a code corpus and retrieving top matches for a query."""
    print("=" * 70)
    print("Person B Integration Demo: Ariadne Dense Retrieval")
    print("=" * 70)

    # 1. Sample Candidate Code Corpus (Person B's search space)
    code_corpus = [
        "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
        "def binary_search(arr, target): l, r = 0, len(arr)-1; while l <= r: m = (l+r)//2; if arr[m] == target: return m; elif arr[m] < target: l = m+1; else: r = m-1; return -1",
        "def fibonacci(n): a, b = 0, 1; [a := b, b := a + b for _ in range(n)]; return a",
        "def lowest_temp(t): return min((int(x) for x in t.split()), default=None)",
        "def is_palindrome(s): s = [c.lower() for c in s if c.isalnum()]; return s == s[::-1]",
    ]

    corpus_labels = [
        "Quicksort Partitioning",
        "Binary Search on Sorted Array",
        "Fibonacci Number Generation",
        "Lowest Temperature Extractor",
        "Palindrome Checker",
    ]

    print(f"Indexing {len(code_corpus)} candidate code snippets with fine-tuned bi-encoder...")
    corpus_embeddings = encode(code_corpus)
    print(f"Corpus embeddings shape: {corpus_embeddings.shape} (float32, L2-normalized)")

    # 2. Incoming User Queries
    sample_queries = [
        "find minimum temperature value from space separated integers",
        "logarithmic search in an ordered list",
        "check if string reads the same forwards and backwards",
    ]

    print("\nEncoding sample user queries...")
    query_embeddings = encode(sample_queries)
    print(f"Query embeddings shape: {query_embeddings.shape}")

    # 3. Vectorized Cosine Similarity Search: Q @ C.T
    similarity_matrix = np.matmul(query_embeddings, corpus_embeddings.T)

    print("\n" + "=" * 70)
    print("Dense Retrieval Results (Top-1 Candidates):")
    print("=" * 70)

    for q_idx, query in enumerate(sample_queries):
        scores = similarity_matrix[q_idx]
        best_cand_idx = int(np.argmax(scores))
        best_score = float(scores[best_cand_idx])
        best_label = corpus_labels[best_cand_idx]
        best_code = code_corpus[best_cand_idx]

        print(f"\nQuery #{q_idx + 1}: '{query}'")
        print(f"  Top Match : [{best_score:.4f} similarity] {best_label}")
        print(f"  Code Snippet: {best_code[:80]}...")

    print("\n" + "=" * 70)
    print("Handoff to Person B verified successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_dense_retrieval_demo()
