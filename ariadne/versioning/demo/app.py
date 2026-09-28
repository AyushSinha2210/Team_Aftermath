"""Ariadne Search Demo Application (owned by Person D).

Demonstrates:
1. Locked Config A Pipeline: Dense retrieval alone using fine-tuned bi-encoder (`best_biencoder`),
   no BM25 fusion, no cross-encoder reranking, per RERANK_CARD.md recommendation.
2. P1 Incremental Re-indexing Simulation: Live demonstration of content hashing and
   IncrementalIndex updating only changed files (re-embedding 1 vs unchanged 499).
Built with Streamlit for reliable, lightweight UI rendering.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure both repo root (Team_Aftermath) and ariadne package are on sys.path
_demo_dir = Path(__file__).resolve().parent
_versioning_dir = _demo_dir.parent
_ariadne_dir = _versioning_dir.parent
_team_root = _ariadne_dir.parent

for p in [str(_team_root), str(_ariadne_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import streamlit as st
import yaml

from ariadne.finetuning.embedder import encode
from ariadne.versioning.incremental_index import IncrementalIndex

logger = logging.getLogger("ariadne.versioning.demo")

# Measured benchmark baseline for full IncrementalIndex.build() on all 500 valid snippets
FULL_REBUILD_BENCHMARK_SEC = 53.26


def load_config() -> Dict[str, Any]:
    """Loads configuration dictionary from central config.yaml."""
    for root_candidate in [_ariadne_dir, _team_root]:
        config_path = root_candidate / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}


def _extract_title(record: Dict[str, Any], fallback_id: str) -> str:
    """Extracts a human-readable title or problem summary for UI display."""
    title = str(record.get("title") or "").strip()
    if title:
        return title

    query = str(record.get("query") or "").strip()
    if query:
        # Take the first non-empty line of the query prompt
        for line in query.splitlines():
            cleaned_line = line.strip().replace("$", "").replace("#", "").strip()
            if cleaned_line:
                if len(cleaned_line) > 70:
                    return cleaned_line[:67] + "..."
                return cleaned_line

    return f"Code Snippet ({fallback_id})"


def load_raw_corpus(limit: int = 500) -> Dict[str, Dict[str, Any]]:
    """Loads a genuine code snippet corpus from the downloaded CoIR Apps dataset.

    Prefers ariadne.data's valid split (500 real code snippets with query prompts),
    falling back to data/raw/corpus or a built-in fallback if files are unavailable.

    Args:
        limit: Maximum number of snippets to load. Defaults to 500.

    Returns:
        Mapping of doc_id to snippet metadata dictionary containing:
            id, code, title, and query.
    """
    corpus_items: Dict[str, Dict[str, Any]] = {}

    # 1. Try loading from valid split (500 real code snippets)
    valid_candidates = [
        _ariadne_dir / "data" / "raw" / "valid",
        _team_root / "data" / "raw" / "valid",
    ]
    for valid_path in valid_candidates:
        if valid_path.exists():
            try:
                from datasets import load_from_disk

                dataset = load_from_disk(str(valid_path))
                total = min(len(dataset), limit)
                for idx in range(total):
                    record = dataset[idx]
                    doc_id = str(record.get("corpus_id") or record.get("_id") or f"doc_{idx}")
                    code = str(record.get("code") or record.get("text") or "")
                    if not code.strip():
                        continue
                    title = _extract_title(record, doc_id)
                    query_text = str(record.get("query") or "")
                    corpus_items[doc_id] = {
                        "id": doc_id,
                        "code": code,
                        "title": title,
                        "query": query_text,
                    }
                if corpus_items:
                    return corpus_items
            except Exception as exc:
                logger.warning("Could not load valid split from %s: %s", valid_path, exc)

    # 2. Try loading from raw corpus directory
    corpus_candidates = [
        _ariadne_dir / "data" / "raw" / "corpus",
        _team_root / "data" / "raw" / "corpus",
    ]
    for corpus_path in corpus_candidates:
        if corpus_path.exists():
            try:
                from datasets import load_from_disk

                dataset = load_from_disk(str(corpus_path))
                total = min(len(dataset), limit)
                for idx in range(total):
                    record = dataset[idx]
                    doc_id = str(record.get("_id") or f"doc_{idx}")
                    code = str(record.get("text") or "")
                    if not code.strip():
                        continue
                    title = _extract_title(record, doc_id)
                    corpus_items[doc_id] = {
                        "id": doc_id,
                        "code": code,
                        "title": title,
                        "query": "",
                    }
                if corpus_items:
                    return corpus_items
            except Exception as exc:
                logger.warning("Could not load corpus from %s: %s", corpus_path, exc)

    # 3. Graceful fallback if dataset files are missing
    fallback_snippets = [
        ("quicksort", "Quicksort Partitioning", "def quicksort(values):\n    if len(values) <= 1: return values\n    pivot = values[0]\n    return quicksort([v for v in values[1:] if v < pivot]) + [pivot] + quicksort([v for v in values[1:] if v >= pivot])"),
        ("binary_search", "Binary Search on Sorted Array", "def binary_search(values, target):\n    left, right = 0, len(values) - 1\n    while left <= right:\n        middle = (left + right) // 2\n        if values[middle] == target: return middle\n        if values[middle] < target: left = middle + 1\n        else: right = middle - 1\n    return -1"),
        ("fibonacci", "Fibonacci Number Generation", "def fibonacci(count):\n    first, second = 0, 1\n    sequence = []\n    for _ in range(count):\n        sequence.append(first)\n        first, second = second, first + second\n    return sequence"),
        ("palindrome", "Palindrome Checker", "def is_palindrome(value):\n    normalized = [char.lower() for char in value if char.isalnum()]\n    return normalized == normalized[::-1]"),
        ("two_sum", "Two Sum Problem", "def two_sum(values, target):\n    positions = {}\n    for index, value in enumerate(values):\n        complement = target - value\n        if complement in positions: return positions[complement], index\n        positions[value] = index\n    return None"),
    ]
    return {
        item_id: {"id": item_id, "code": code, "title": title, "query": ""}
        for item_id, title, code in fallback_snippets
    }


@st.cache_resource(show_spinner="Loading corpus and initializing IncrementalIndex...")
def get_indexed_corpus() -> Tuple[Dict[str, Dict[str, Any]], IncrementalIndex]:
    """Loads corpus and initializes or builds the IncrementalIndex.

    Uses Person D's IncrementalIndex to cache embeddings on disk, ensuring
    fast subsequent startups and integrating P1 infrastructure directly into the demo.

    Returns:
        Tuple of (doc_lookup dictionary, loaded IncrementalIndex instance).
    """
    config = load_config()
    cache_rel = config.get("versioning", {}).get("incremental_cache_path", "versioning/.index_cache")
    cache_path = (_ariadne_dir / cache_rel).resolve()

    doc_lookup = load_raw_corpus(limit=500)
    corpus_dict = {doc_id: item["code"] for doc_id, item in doc_lookup.items()}

    index = IncrementalIndex(cache_path=cache_path)
    cached_ids, _ = index.get_embeddings()

    # If the cache is empty or does not match the loaded corpus, build it
    if not cached_ids or set(cached_ids) != set(corpus_dict.keys()):
        logger.info("Building IncrementalIndex cache for %d documents...", len(corpus_dict))
        index.build(corpus_dict)

    return doc_lookup, index


def init_session_state() -> None:
    """Initializes mutable corpus and index session state across Streamlit tabs."""
    if "doc_lookup" not in st.session_state or "index" not in st.session_state:
        base_lookup, index = get_indexed_corpus()
        st.session_state["doc_lookup"] = {k: dict(v) for k, v in base_lookup.items()}
        st.session_state["orig_doc_lookup"] = {k: dict(v) for k, v in base_lookup.items()}
        st.session_state["index"] = index
        st.session_state["last_update_summary"] = None
        st.session_state["last_update_time"] = None
        st.session_state["last_updated_id"] = None


def search(
    query: str,
    doc_lookup: Dict[str, Dict[str, Any]],
    index: IncrementalIndex,
    top_k: int = 5,
    min_confidence_threshold: float = 0.20,
) -> Dict[str, Any]:
    """Performs Config A (dense alone) retrieval against the indexed corpus.

    Handles arbitrary user input robustly (empty, whitespace, very long text,
    special characters, non-English text) and degrades gracefully for nonsense
    queries.

    Args:
        query: User search query string.
        doc_lookup: Mapping from doc_id to document metadata.
        index: Loaded IncrementalIndex instance.
        top_k: Number of top ranked matches to return.
        min_confidence_threshold: Cosine score below which matches are flagged
            as low confidence / no strong match.

    Returns:
        Dictionary containing:
            - status: 'empty' | 'success' | 'low_confidence' | 'error'
            - is_strong_match: bool flag
            - top_score: float maximum cosine similarity
            - message: user-facing explanation string
            - results: list of result dicts
    """
    if not query:
        return {
            "status": "empty",
            "is_strong_match": False,
            "top_score": 0.0,
            "message": "Please enter a search query.",
            "results": [],
        }

    # Clean query: strip whitespace and filter out non-printable control characters
    cleaned_query = "".join(
        c for c in str(query) if c.isprintable() or c in "\n\t "
    ).strip()

    if not cleaned_query:
        return {
            "status": "empty",
            "is_strong_match": False,
            "top_score": 0.0,
            "message": "Query contains no searchable characters.",
            "results": [],
        }

    # Robust handling for very long input text: truncate safely to prevent OOM
    max_query_chars = 2000
    if len(cleaned_query) > max_query_chars:
        cleaned_query = cleaned_query[:max_query_chars]

    doc_ids, corpus_embeddings = index.get_embeddings()
    if len(doc_ids) == 0 or corpus_embeddings.shape[0] == 0:
        return {
            "status": "error",
            "is_strong_match": False,
            "top_score": 0.0,
            "message": "Corpus index is empty.",
            "results": [],
        }

    try:
        query_embedding = encode([cleaned_query])[0]
    except Exception as exc:
        return {
            "status": "error",
            "is_strong_match": False,
            "top_score": 0.0,
            "message": f"Failed to encode query: {exc}",
            "results": [],
        }

    # Dense retrieval alone: cosine similarity over L2-normalized embeddings
    cosine_scores = np.dot(corpus_embeddings, query_embedding)
    max_score = float(np.max(cosine_scores)) if len(cosine_scores) > 0 else 0.0

    ranked_indices = np.argsort(cosine_scores)[::-1][:top_k]

    results: List[Dict[str, Any]] = []
    for rank, idx in enumerate(ranked_indices, start=1):
        doc_id = doc_ids[idx]
        doc = doc_lookup.get(doc_id, {"id": doc_id, "title": f"Snippet {doc_id}", "code": ""})
        score = float(cosine_scores[idx])
        results.append(
            {
                "rank": rank,
                "id": doc_id,
                "title": doc.get("title", f"Snippet {doc_id}"),
                "score": score,
                "code": doc.get("code", ""),
                "query": doc.get("query", ""),
            }
        )

    is_strong = max_score >= min_confidence_threshold
    status = "success" if is_strong else "low_confidence"
    message = (
        f"Found {len(results)} relevant matches."
        if is_strong
        else f"No strong match found for this query (top similarity score was only {max_score:.3f})."
    )

    return {
        "status": status,
        "is_strong_match": is_strong,
        "top_score": max_score,
        "message": message,
        "results": results,
    }


def _render_results(results: List[Dict[str, Any]]) -> None:
    """Renders ranked search results in the UI with score indicators and code."""
    for r in results:
        score_pct = max(0.0, min(1.0, r["score"]))
        with st.expander(
            f"#{r['rank']} — {r['title']} (`{r['id']}`) — Similarity: {r['score']:.4f}",
            expanded=(r["rank"] <= 2),
        ):
            st.progress(score_pct, text=f"Cosine Similarity: {r['score']:.4f}")
            if r.get("query"):
                with st.container():
                    st.caption("Problem Description:")
                    st.text(r["query"][:300] + ("..." if len(r["query"]) > 300 else ""))
            st.caption("Python Solution:")
            st.code(r["code"], language="python")


def render_search_tab() -> None:
    """Renders the main Semantic Search tab."""
    doc_lookup = st.session_state["doc_lookup"]
    index = st.session_state["index"]

    st.markdown("##### Quick Example Queries")
    example_cols = st.columns(4)
    examples = [
        "Destroy optimal village in tree network to minimize fighters",
        "Encrypt string by taking every second character alternating",
        "Bug fixing regex failure and pattern matching",
        "Ted loves prime numbers game optimal strategy",
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
            placeholder="e.g. tree network traversal, regex pattern bug, prime numbers game...",
        )
        col1, col2 = st.columns([1, 4])
        with col1:
            top_k = st.slider("Top Results (K)", min_value=1, max_value=20, value=5)
        submitted = st.form_submit_button("Search Codebase", type="primary")

    active_query = query if submitted else selected_example

    if active_query and active_query.strip():
        with st.spinner("Searching with fine-tuned bi-encoder..."):
            search_response = search(
                query=active_query,
                doc_lookup=doc_lookup,
                index=index,
                top_k=top_k,
                min_confidence_threshold=0.20,
            )

        status = search_response["status"]
        results = search_response["results"]
        is_strong = search_response["is_strong_match"]
        top_score = search_response["top_score"]

        if status == "empty":
            st.info("Please enter a valid search query.")
        elif status == "error":
            st.error(search_response["message"])
        elif not is_strong:
            if top_score < 0.12:
                st.warning(
                    f"⚠️ **No strong match found in the corpus.**\n\n"
                    f"The highest similarity score was only **{top_score:.4f}** (threshold: 0.2000). "
                    "The query does not appear to correspond to any code in the indexed repository. "
                    "Try rephrasing with specific programming terms or concepts."
                )
            else:
                st.warning(
                    f"⚠️ **Weak match detected.**\n\n"
                    f"Highest similarity score was **{top_score:.4f}** (below standard confidence threshold 0.2000). "
                    "Displaying closest candidate snippets below:"
                )
                _render_results(results)
        else:
            st.markdown(
                f"### 🏆 Top {len(results)} Ranked Matches (Config A — Dense Alone)"
            )
            _render_results(results)


def render_simulation_tab() -> None:
    """Renders the Version Update Simulation tab (P1 Incremental Indexing Proof)."""
    st.subheader("⚡ Version Update Simulation (P1 Live Proof Point)")
    st.markdown(
        "Demonstrates **Person D's content-hashed incremental re-indexing** in real time: "
        "when source code is modified in a repository, `IncrementalIndex.update()` detects "
        "content hash differences, re-embeds **only** the modified files, and reuses "
        "the existing cached embeddings for all unchanged files."
    )

    doc_lookup: Dict[str, Dict[str, Any]] = st.session_state["doc_lookup"]
    index: IncrementalIndex = st.session_state["index"]

    options = list(doc_lookup.keys())
    selected_id = st.selectbox(
        "Choose a snippet from the 500-item corpus to edit:",
        options=options,
        format_func=lambda doc_id: f"{doc_id} — {doc_lookup[doc_id]['title']}",
    )

    current_code = doc_lookup[selected_id]["code"]

    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        st.caption(f"Currently editing snippet **`{selected_id}`**: {doc_lookup[selected_id]['title']}")
    with col_btn2:
        use_template = st.button("Load Quicksort Template", help="Inserts a quicksort implementation to test dramatic rank change")

    if use_template:
        default_editor_code = (
            "def quicksort(arr):\n"
            "    # quicksort partitioning pivot recursive sorting algorithm\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    pivot = arr[len(arr) // 2]\n"
            "    left = [x for x in arr if x < pivot]\n"
            "    middle = [x for x in arr if x == pivot]\n"
            "    right = [x for x in arr if x > pivot]\n"
            "    return quicksort(left) + middle + quicksort(right)\n"
        )
    else:
        default_editor_code = current_code

    edited_code = st.text_area(
        "Edit Python code text:",
        value=default_editor_code,
        height=220,
    )

    col_act1, col_act2 = st.columns([1, 1])
    with col_act1:
        apply_btn = st.button("⚡ Apply Incremental Update", type="primary", use_container_width=True)
    with col_act2:
        reset_btn = st.button("🔄 Reset Corpus to Original", use_container_width=True)

    if reset_btn:
        orig = st.session_state["orig_doc_lookup"]
        revert_corpus = {doc_id: info["code"] for doc_id, info in orig.items()}
        t_revert_start = time.time()
        summary = index.update(revert_corpus)
        t_revert = time.time() - t_revert_start
        st.session_state["doc_lookup"] = {k: dict(v) for k, v in orig.items()}
        st.session_state["last_update_summary"] = summary
        st.session_state["last_update_time"] = t_revert
        st.session_state["last_updated_id"] = "reset"
        st.success("Corpus and index successfully reset to original 500 snippets!")
        st.rerun()

    if apply_btn:
        new_corpus = {
            doc_id: info["code"]
            for doc_id, info in doc_lookup.items()
        }
        new_corpus[selected_id] = edited_code

        with st.spinner("Incrementally updating vector index..."):
            t_start = time.time()
            summary = index.update(new_corpus)
            t_update = time.time() - t_start

        # Update session state lookup
        doc_lookup[selected_id]["code"] = edited_code
        st.session_state["last_update_summary"] = summary
        st.session_state["last_update_time"] = t_update
        st.session_state["last_updated_id"] = selected_id

    # Display update summary if an update has taken place
    if st.session_state.get("last_update_summary"):
        summary = st.session_state["last_update_summary"]
        t_update = st.session_state["last_update_time"]
        updated_id = st.session_state["last_updated_id"]

        st.markdown("---")
        st.markdown("#### 📊 Incremental Re-Indexing Proof Metrics")

        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("⚡ Re-Embedded", f"{summary['re_embedded']} file", delta="Recomputed", delta_color="inverse")
        with m_col2:
            st.metric("🔒 Unchanged", f"{summary['unchanged']} files", delta="Cache Hit", delta_color="normal")
        with m_col3:
            st.metric("Modified", f"{summary['changed']}")
        with m_col4:
            st.metric("Added", f"{summary['added']}")
        with m_col5:
            st.metric("Removed", f"{summary['removed']}")

        # Timing comparison
        full_time = FULL_REBUILD_BENCHMARK_SEC
        speedup = full_time / max(t_update, 0.001)

        st.markdown("#### ⏱️ Latency Comparison (Incremental Update vs Full Rebuild)")
        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            st.metric(
                label="Incremental Update Time",
                value=f"{t_update:.3f} s",
                help="Actual wall-clock time taken by index.update() (only 1 file embedded)"
            )
        with t_col2:
            st.metric(
                label="Full Rebuild Benchmark (500 items)",
                value=f"{full_time:.2f} s",
                help="Measured time taken by IncrementalIndex.build() on CPU for the entire corpus"
            )
        with t_col3:
            st.metric(
                label="Efficiency Speedup",
                value=f"{speedup:.1f}x faster",
                delta=f"-{full_time - t_update:.1f}s saved"
            )

        st.info(
            f"🎯 **Proof Verified:** Only `{summary['re_embedded']}` file was sent to the bi-encoder, while "
            f"`{summary['unchanged']}` files were resolved via SHA-256 hash matching without inference. "
            f"The index is immediately updated end-to-end. "
            "Switch to the **🔍 Semantic Search** tab to query the updated code live!"
        )


def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="Ariadne Code Search Demo",
        page_icon="🚀",
        layout="wide",
    )

    init_session_state()

    doc_lookup = st.session_state["doc_lookup"]
    index = st.session_state["index"]
    corpus_size = len(doc_lookup)

    st.title("🚀 Ariadne Code Intelligence Demo")
    st.caption("**Theme 01: Agentic Code Intelligence** | *PRISM GenAI Hackathon*")

    st.markdown(
        f"🔍 **Searching across {corpus_size:,} code snippets** from the validated CoIR Apps benchmark"
    )

    st.info(
        "Powered by the locked **Config A** pipeline: Dense retrieval alone using "
        "the fine-tuned bi-encoder (`best_biencoder`), with persistent caching via "
        "`IncrementalIndex` (P1 infrastructure), per `RERANK_CARD.md` findings."
    )

    # Sidebar metadata and statistics
    with st.sidebar:
        st.header("⚙️ System Status")
        st.metric("Indexed Corpus Size", f"{corpus_size:,} snippets")
        st.markdown("---")
        st.markdown("**Pipeline Configuration:**")
        st.markdown("- **Model:** Dense retrieval alone (Config A)")
        st.markdown("- **Checkpoint:** `finetuning/checkpoints/best_biencoder`")
        st.markdown("- **Embedding Dim:** 384 (float32)")
        st.markdown(f"- **Index Cache:** `{index.cache_path.name}`")
        st.markdown("- **Dedup Threshold:** 0.95")
        st.markdown("---")
        st.caption("Ariadne Code Search • Person D Demo")

    tab_search, tab_simulation = st.tabs([
        "🔍 Semantic Search",
        "⚡ Version Update Simulation (P1 Demo)",
    ])

    with tab_search:
        render_search_tab()

    with tab_simulation:
        render_simulation_tab()


if __name__ == "__main__":
    main()
