from __future__ import annotations

from typing import Any

from ariadne.reranking import calibration, cross_encoder
from ariadne.retrieval.pipeline import HybridPipeline


CORPUS: dict[str, str] = {
    "binary_search": """
def binary_search(values, target):
    left, right = 0, len(values) - 1
    while left <= right:
        middle = (left + right) // 2
        if values[middle] == target:
            return middle
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle - 1
    return -1
""",
    "quicksort": """
def quicksort(values):
    if len(values) <= 1:
        return values
    pivot = values[0]
    return (quicksort([value for value in values[1:] if value < pivot])
            + [pivot]
            + quicksort([value for value in values[1:] if value >= pivot]))
""",
    "merge_sort": """
def merge_sort(values):
    if len(values) <= 1:
        return values
    middle = len(values) // 2
    left = merge_sort(values[:middle])
    right = merge_sort(values[middle:])
    return merge(left, right)
""",
    "fibonacci": """
def fibonacci(count):
    first, second = 0, 1
    sequence = []
    for _ in range(count):
        sequence.append(first)
        first, second = second, first + second
    return sequence
""",
    "factorial": """
def factorial(number):
    result = 1
    for value in range(2, number + 1):
        result *= value
    return result
""",
    "palindrome": """
def is_palindrome(value):
    normalized = [char.lower() for char in value if char.isalnum()]
    return normalized == normalized[::-1]
""",
    "lowest_temperature": """
def lowest_temperature(values):
    readings = (int(value) for value in values.split())
    return min(readings, default=None)
""",
    "word_frequency": """
def word_frequency(text):
    counts = {}
    for word in text.lower().split():
        counts[word] = counts.get(word, 0) + 1
    return counts
""",
    "deduplicate": """
def deduplicate(values):
    seen = set()
    unique = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique
""",
    "two_sum": """
def two_sum(values, target):
    positions = {}
    for index, value in enumerate(values):
        complement = target - value
        if complement in positions:
            return positions[complement], index
        positions[value] = index
    return None
""",
    "bfs": """
def breadth_first_search(graph, start):
    queue = [start]
    visited = {start}
    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited
""",
    "dfs": """
def depth_first_search(graph, start, visited=None):
    visited = set() if visited is None else visited
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            depth_first_search(graph, neighbor, visited)
    return visited
""",
    "topological_sort": """
def topological_sort(graph):
    order, temporary, permanent = [], set(), set()
    def visit(node):
        if node in temporary:
            raise ValueError('cycle')
        if node not in permanent:
            temporary.add(node)
            for child in graph.get(node, []):
                visit(child)
            temporary.remove(node)
            permanent.add(node)
            order.append(node)
    for node in graph:
        visit(node)
    return order[::-1]
""",
    "longest_common_subsequence": """
def longest_common_subsequence(first, second):
    table = [[0] * (len(second) + 1) for _ in range(len(first) + 1)]
    for row, left in enumerate(first, 1):
        for column, right in enumerate(second, 1):
            table[row][column] = table[row - 1][column - 1] + 1 if left == right else max(table[row - 1][column], table[row][column - 1])
    return table[-1][-1]
""",
    "http_retry": """
def request_with_retry(client, url, attempts=3):
    for attempt in range(attempts):
        try:
            return client.get(url)
        except TimeoutError:
            if attempt == attempts - 1:
                raise
""",
    "json_config": """
def load_json_config(path):
    import json
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)
""",
}

QUERIES = (
    "find an item with logarithmic search in a sorted array",
    "sort a list by recursively splitting and merging halves",
    "traverse a graph level by level from a starting node",
    "count how often each word appears in text",
)


def test_pipeline_output_runs_directly_through_reranker_and_calibration() -> None:
    pipeline = HybridPipeline(CORPUS)

    for query in QUERIES:
        pipeline_candidates = pipeline.retrieve(query, k=len(CORPUS))
        expected_fields: dict[str, dict[str, Any]] = {
            str(candidate["id"]): {
                "fusion_score": candidate["fusion_score"],
                "dense_score": candidate["dense_score"],
                "sparse_score": candidate["sparse_score"],
                "sources": candidate["sources"],
            }
            for candidate in pipeline_candidates
        }

        reranked = cross_encoder.rerank(query, pipeline_candidates)
        assert len(reranked) == len(pipeline_candidates)
        assert all("rerank_score" in candidate for candidate in reranked)

        for candidate in reranked:
            original = expected_fields[str(candidate["id"])]
            assert candidate["fusion_score"] == original["fusion_score"]
            assert candidate["dense_score"] == original["dense_score"]
            assert candidate["sparse_score"] == original["sparse_score"]
            assert candidate["sources"] == original["sources"]

        result = calibration.calibrate(reranked)
        assert set(("should_abstain", "confidence_variance", "top_candidate")) <= result.keys()
        assert isinstance(result["should_abstain"], bool)
        assert isinstance(result["confidence_variance"], float)
        assert result["top_candidate"] is None or isinstance(result["top_candidate"], dict)
