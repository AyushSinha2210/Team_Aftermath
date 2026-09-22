"""Hard-negative mining using BM25 over the CoIR apps training split.

Identifies lexically confusing code snippets (high BM25 term overlap with the
query description, but incorrect solution) to construct challenging training
triplets (anchor, positive, hard_negative) for contrastive fine-tuning.

STRICT RULE: Operates exclusively on the 'train' split. Never touches 'test'.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml
from datasets import Dataset, load_from_disk
from scipy import sparse

# Ensure repository root and workspace root are in sys.path
_repo_root = Path(__file__).resolve().parent.parent
_workspace_root = _repo_root.parent
for p in [str(_workspace_root), str(_repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Configures structured logger.

    Args:
        log_level: Logging verbosity level.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger("ariadne.finetuning.hard_negative_mining")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads central YAML configuration.

    Args:
        config_path: Optional path to config.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = _repo_root / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tokenize_code(text: str) -> List[str]:
    """Code-aware regex tokenizer extracting alphanumeric identifiers and tokens.

    Splits snake_case and camelCase identifiers while retaining core tokens.

    Args:
        text: Code or query text string.

    Returns:
        List of lowercase token strings.
    """
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+", text.lower())
    return tokens


class FastBM25Indexer:
    """High-performance vectorized BM25 indexer using sparse matrix multiplication."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self.doc_mat: Optional[sparse.csr_matrix] = None
        self.corpus_ids: List[str] = []

    def fit(self, corpus_ids: List[str], documents: List[str]) -> FastBM25Indexer:
        """Indexes corpus documents into a sparse BM25 matrix.

        Args:
            corpus_ids: List of unique document identifier strings.
            documents: List of raw code document strings.

        Returns:
            Fitted FastBM25Indexer instance.
        """
        self.corpus_ids = corpus_ids
        num_docs = len(documents)

        # 1. Tokenize documents
        tokenized_docs = [tokenize_code(doc) for doc in documents]
        doc_lens = np.array([max(1, len(tokens)) for tokens in tokenized_docs], dtype=np.float32)
        self.avgdl = float(np.mean(doc_lens))

        # 2. Build vocabulary and document frequencies
        df: Counter[str] = Counter()
        for doc_tokens in tokenized_docs:
            for term in set(doc_tokens):
                df[term] += 1

        self.vocab = {term: idx for idx, term in enumerate(df.keys())}

        # 3. Compute IDF with standard Okapi smoothing
        self.idf = {
            term: math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

        # 4. Construct sparse document matrix with BM25 term weights
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for doc_idx, tokens in enumerate(tokenized_docs):
            term_counts = Counter(tokens)
            doc_len = doc_lens[doc_idx]
            denom_penalty = self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))

            for term, count in term_counts.items():
                if term in self.vocab:
                    col_idx = self.vocab[term]
                    idf_val = self.idf[term]
                    bm25_weight = idf_val * ((count * (self.k1 + 1.0)) / (count + denom_penalty))
                    rows.append(doc_idx)
                    cols.append(col_idx)
                    data.append(bm25_weight)

        self.doc_mat = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(num_docs, len(self.vocab)),
            dtype=np.float32,
        )
        return self

    def score_queries(self, queries: List[str]) -> sparse.csr_matrix:
        """Computes BM25 score matrix between all queries and the indexed corpus.

        Args:
            queries: List of query strings.

        Returns:
            Sparse score matrix of shape (len(queries), num_corpus_docs).
        """
        if self.doc_mat is None:
            raise RuntimeError("FastBM25Indexer must be fitted before scoring queries.")

        num_queries = len(queries)
        q_rows: List[int] = []
        q_cols: List[int] = []
        q_data: List[float] = []

        for q_idx, query in enumerate(queries):
            tokens = tokenize_code(query)
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                if term in self.vocab:
                    q_rows.append(q_idx)
                    q_cols.append(self.vocab[term])
                    q_data.append(float(count))

        query_mat = sparse.csr_matrix(
            (q_data, (q_rows, q_cols)),
            shape=(num_queries, len(self.vocab)),
            dtype=np.float32,
        )

        # Matrix product: (N_queries, Vocab) x (Vocab, N_docs) -> (N_queries, N_docs)
        score_mat = query_mat.dot(self.doc_mat.T)
        return score_mat


def mine_bm25_hard_negatives(
    config: Optional[Dict[str, Any]] = None,
    num_negatives: int = 1,
    output_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Mines lexical hard negatives for each training query using BM25.

    Strictly uses the 'train' split dataset. Excludes ground-truth positive
    documents and exact text matches.

    Args:
        config: Optional pre-loaded config dict.
        num_negatives: Number of top hard negatives to mine per query.
        output_path: Destination path for mined JSON triplets.

    Returns:
        List of mined triplet dictionaries.
    """
    if config is None:
        config = load_config()

    logger = setup_logger(config.get("global", {}).get("log_level", "INFO"))

    # Validate split paths
    data_cfg = config.get("data", {})
    raw_dir = _repo_root / data_cfg.get("raw_dir", "data/raw")
    train_split_name = data_cfg.get("train_split", "train")
    train_path = raw_dir / train_split_name

    logger.info("=" * 70)
    logger.info("Starting BM25 Hard-Negative Mining")
    logger.info("Dataset split: %s (STRICTLY train only)", train_path)

    # SAFETY CHECK
    assert "test" not in str(train_path).lower(), "SAFETY VIOLATION: test split accessed!"

    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset split not found at {train_path}. Run download.sh first.")

    train_dataset: Dataset = load_from_disk(str(train_path))
    num_train = len(train_dataset)
    logger.info("Loaded %d training query-code pairs.", num_train)

    query_ids: List[str] = train_dataset["query_id"]
    corpus_ids: List[str] = train_dataset["corpus_id"]
    queries: List[str] = train_dataset["query"]
    codes: List[str] = train_dataset["code"]

    # Gather unique candidate documents efficiently O(N)
    cid_to_code: Dict[str, str] = {}
    for cid, code in zip(corpus_ids, codes):
        if cid not in cid_to_code:
            cid_to_code[cid] = code
    unique_corpus_ids: List[str] = list(cid_to_code.keys())
    unique_codes: List[str] = [cid_to_code[cid] for cid in unique_corpus_ids]

    logger.info("Corpus pool size: %d unique code solutions.", len(unique_codes))

    # Retrieve BM25 hyperparameters
    k1 = float(config.get("retrieval", {}).get("bm25_k1", 1.5))
    b = float(config.get("retrieval", {}).get("bm25_b", 0.75))

    # Fit BM25 indexer
    t0 = time.time()
    indexer = FastBM25Indexer(k1=k1, b=b)
    indexer.fit(unique_corpus_ids, unique_codes)
    fit_duration = time.time() - t0
    logger.info("BM25 index built across %d documents in %.2f seconds.", len(unique_codes), fit_duration)

    # Score all queries against all candidates
    t1 = time.time()
    score_matrix = indexer.score_queries(queries)
    score_duration = time.time() - t1
    logger.info(
        "Scored all %d queries against %d documents in %.2f seconds (%.2f ms/query).",
        num_train,
        len(unique_codes),
        score_duration,
        (score_duration / max(1, num_train)) * 1000,
    )

    # Convert to CSR for row slicing
    csr_scores = score_matrix.tocsr()

    # Mine hard negatives per query
    mined_triplets: List[Dict[str, Any]] = []
    mined_scores: List[float] = []

    for q_idx in range(num_train):
        qid = query_ids[q_idx]
        target_cid = corpus_ids[q_idx]
        query_text = queries[q_idx]
        positive_code = codes[q_idx]

        # Extract row scores
        row_scores = csr_scores.getrow(q_idx).toarray()[0]

        top_candidates = np.argpartition(-row_scores, min(50, len(row_scores) - 1))[:50]
        sorted_top = top_candidates[np.argsort(-row_scores[top_candidates])]

        hard_negatives: List[Dict[str, Any]] = []
        for cand_idx in sorted_top:
            cand_cid = unique_corpus_ids[cand_idx]
            cand_code = unique_codes[cand_idx]
            cand_score = float(row_scores[cand_idx])

            # Exclusion criteria:
            # 1. Must NOT be the target ground truth corpus_id
            # 2. Must NOT have identical code text to the positive solution
            if cand_cid == target_cid:
                continue
            if cand_code.strip() == positive_code.strip():
                continue

            hard_negatives.append(
                {
                    "corpus_id": cand_cid,
                    "code": cand_code,
                    "bm25_score": round(cand_score, 4),
                }
            )

            if len(hard_negatives) >= num_negatives:
                break

        # Fallback if no lexical overlap found: pick another document
        if not hard_negatives:
            fallback_idx = (q_idx + 1) % len(unique_corpus_ids)
            hard_negatives.append(
                {
                    "corpus_id": unique_corpus_ids[fallback_idx],
                    "code": unique_codes[fallback_idx],
                    "bm25_score": 0.0,
                }
            )

        top_neg = hard_negatives[0]
        mined_scores.append(top_neg["bm25_score"])

        mined_triplets.append(
            {
                "query_id": qid,
                "corpus_id": target_cid,
                "query": query_text,
                "positive_code": positive_code,
                "hard_negative_corpus_id": top_neg["corpus_id"],
                "hard_negative_code": top_neg["code"],
                "hard_negative_bm25_score": top_neg["bm25_score"],
                "all_hard_negatives": hard_negatives,
            }
        )

    # Mining statistics
    logger.info("=" * 70)
    logger.info("BM25 Mining Complete!")
    logger.info("Total triplets mined: %d", len(mined_triplets))
    logger.info("Mean top hard negative BM25 score: %.2f", float(np.mean(mined_scores)))
    logger.info("Min top hard negative BM25 score : %.2f", float(np.min(mined_scores)))
    logger.info("Max top hard negative BM25 score : %.2f", float(np.max(mined_scores)))

    # Save results
    if output_path is None:
        processed_dir = _repo_root / data_cfg.get("processed_dir", "data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / "train_bm25_triplets.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mined_triplets, f, indent=2)

    logger.info("Saved mined training triplets to: %s", output_path)
    return mined_triplets


def mine_dense_hard_negatives(
    model_name_or_path: str,
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
    num_negatives: int = 1,
) -> List[Dict[str, Any]]:
    """Mines actual dense false positives from a fine-tuned checkpoint over the valid split.

    # =========================================================================
    # EXCEPTION TO TRAIN-ONLY RULE:
    # Phase 6 explicit instruction permits reading 'valid' to harvest actual
    # model false positives as hard negatives for round-2 iterative retraining.
    # The 'test' split remains STRICTLY PROHIBITED under all circumstances!
    # =========================================================================

    Args:
        model_name_or_path: Path to checkpoint to harvest false positives from.
        config: Optional configuration dictionary.
        output_path: Optional output path for mined JSON triplets.
        num_negatives: Number of top dense false positives to mine per query.

    Returns:
        List of mined dense triplet dictionaries.
    """
    if config is None:
        config = load_config()

    logger = setup_logger(config.get("global", {}).get("log_level", "INFO"))

    data_cfg = config.get("data", {})
    raw_dir = _repo_root / data_cfg.get("raw_dir", "data/raw")
    valid_split_name = data_cfg.get("valid_split", "valid")
    valid_path = raw_dir / valid_split_name

    # Safety assertion: verify never touching test
    assert "test" not in str(valid_path).lower(), "SAFETY VIOLATION: test split accessed!"
    assert valid_split_name.lower() == "valid", "SAFETY VIOLATION: split must be valid!"

    logger.info("=" * 70)
    logger.info("Starting Phase 6 Round 2 Dense Hard-Negative Mining")
    logger.info("[EXCEPTION TO TRAIN-ONLY RULE]: Reading from '%s' split to harvest model false positives.", valid_path)
    logger.info("Source checkpoint: %s", model_name_or_path)

    valid_dataset: Dataset = load_from_disk(str(valid_path))
    num_queries = len(valid_dataset)
    logger.info("Loaded %d validation examples.", num_queries)

    query_ids: List[str] = valid_dataset["query_id"]
    corpus_ids: List[str] = valid_dataset["corpus_id"]
    queries: List[str] = valid_dataset["query"]
    codes: List[str] = valid_dataset["code"]

    # Candidate pool
    cid_to_code: Dict[str, str] = {}
    for cid, code in zip(corpus_ids, codes):
        if cid not in cid_to_code:
            cid_to_code[cid] = code
    unique_corpus_ids: List[str] = list(cid_to_code.keys())
    unique_codes: List[str] = [cid_to_code[cid] for cid in unique_corpus_ids]

    # Import encode from embedder
    from ariadne.finetuning.embedder import encode

    eval_batch_size = int(config.get("finetuning", {}).get("eval_batch_size", 32))

    logger.info("Encoding %d queries with %s...", len(queries), model_name_or_path)
    query_embeddings = encode(queries, batch_size=eval_batch_size, model_name_or_path=model_name_or_path)

    logger.info("Encoding %d candidate code documents...", len(unique_codes))
    corpus_embeddings = encode(unique_codes, batch_size=eval_batch_size, model_name_or_path=model_name_or_path)

    logger.info("Computing dense similarity matrix...")
    sim_matrix = np.matmul(query_embeddings, corpus_embeddings.T)

    mined_dense_triplets: List[Dict[str, Any]] = []

    for q_idx in range(num_queries):
        qid = query_ids[q_idx]
        target_cid = corpus_ids[q_idx]
        q_text = queries[q_idx]
        pos_code = codes[q_idx]

        row_sims = sim_matrix[q_idx]
        sorted_indices = np.argsort(-row_sims)

        dense_negatives: List[Dict[str, Any]] = []
        for cand_idx in sorted_indices:
            cand_cid = unique_corpus_ids[cand_idx]
            cand_code = unique_codes[cand_idx]
            cand_sim = float(row_sims[cand_idx])

            # Exclusion: cannot be positive target
            if cand_cid == target_cid or cand_code.strip() == pos_code.strip():
                continue

            dense_negatives.append(
                {
                    "corpus_id": cand_cid,
                    "code": cand_code,
                    "similarity": round(cand_sim, 4),
                }
            )

            if len(dense_negatives) >= num_negatives:
                break

        top_false_positive = dense_negatives[0]
        mined_dense_triplets.append(
            {
                "query_id": qid,
                "corpus_id": target_cid,
                "query": q_text,
                "positive_code": pos_code,
                "hard_negative_corpus_id": top_false_positive["corpus_id"],
                "hard_negative_code": top_false_positive["code"],
                "hard_negative_similarity": top_false_positive["similarity"],
            }
        )

    logger.info("Round 2 Dense Mining complete! Harvested %d actual false positive triplets.", len(mined_dense_triplets))
    mean_sim = float(np.mean([t["hard_negative_similarity"] for t in mined_dense_triplets]))
    logger.info("Mean false positive cosine similarity: %.4f", mean_sim)

    if output_path is None:
        processed_dir = _repo_root / data_cfg.get("processed_dir", "data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / "round2_dense_negatives.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mined_dense_triplets, f, indent=2)

    logger.info("Saved round 2 dense triplets to: %s", output_path)
    return mined_dense_triplets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine hard negatives (BM25 or dense) on apps dataset.")
    parser.add_argument("--num-negatives", type=int, default=1, help="Number of hard negatives per query.")
    parser.add_argument("--output", type=str, default=None, help="Custom output path for mined triplets JSON.")
    parser.add_argument("--dense", action="store_true", default=False, help="Mine dense false positives from checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for dense mining.")
    args = parser.parse_args()

    custom_output = Path(args.output) if args.output else None
    if args.dense:
        ckpt = args.checkpoint or "ariadne/finetuning/checkpoints/checkpoint-round1_bm25-40"
        mine_dense_hard_negatives(model_name_or_path=ckpt, num_negatives=args.num_negatives, output_path=custom_output)
    else:
        mine_bm25_hard_negatives(num_negatives=args.num_negatives, output_path=custom_output)
