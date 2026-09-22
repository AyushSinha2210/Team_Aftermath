# Model Card: Ariadne Bi-Encoder (Person A Handoff)

## 1. Model Overview
- **Architecture**: Bi-Encoder dense text/code representation model based on `sentence-transformers/all-MiniLM-L6-v2`.
- **Checkpoint Location**: `ariadne/finetuning/checkpoints/best_biencoder` (configured in [`ariadne/config.yaml`](config.yaml)).
- **Output Embedding**: 384-dimensional dense vectors, L2-normalized ($||v||_2 = 1.0$), `float32` precision.
- **Maximum Sequence Length**: 512 tokens.
- **Target Hardware**: CPU-first inference (<35M parameters, optimized for real-time hackathon deployment).

---

## 2. Drop-in Integration Contract for Person B (and Persons C & D)

The `encode()` interface is **100% backward-compatible** and has **never changed** since Phase 2:

```python
from ariadne.finetuning.embedder import encode

# Signature:
# encode(texts: list[str], batch_size: int = 32, normalize_embeddings: bool = True) -> np.ndarray

queries = ["How to find longest palindromic substring?", "binary search on sorted array"]
query_vectors = encode(queries)
# Returns: np.ndarray of shape (2, 384), float32, L2-normalized

code_snippets = ["def is_palindrome(s): ...", "def binary_search(arr, target): ..."]
code_vectors = encode(code_snippets)
# Returns: np.ndarray of shape (2, 384), float32, L2-normalized
```

> **Zero Training Code Dependency**: Person B can clone the repository, call `encode()`, and immediately use the fine-tuned bi-encoder without ever reading or importing `train.py`.

---

## 3. Training Data & Strategy
- **Base Model**: `sentence-transformers/all-MiniLM-L6-v2`.
- **Training Split**: CoIR `apps` training split (4,500 query-code pairs from `data/raw/train`).
- **Data Protection Guarantee**: Trained strictly on `train`. The `test` split was **never accessed** or loaded.
- **Fine-Tuning Stages**:
  1. **Phase 3 (In-Batch Negatives)**: Contrastive fine-tuning using `MultipleNegativesRankingLoss` (scale=20.0, lr=3e-5) with in-batch symmetric negative pairs.
  2. **Phase 4 (BM25 Hard Negatives)**: Mined 4,500 high-lexical-overlap wrong-answer triplets using vectorized BM25 indexer to suppress lexical false positives.
  3. **Phase 6 (Iterative Dense Hard Negatives)**: Harvested actual dense false positives (mean similarity: 0.4347) from the Phase 4 model over `valid` to target the hardest semantic confusions.

---

## 4. Benchmark Performance on Validation Split (`valid`)

Evaluated against the 500-sample validation split (`data/raw/valid`):

| Model / Phase | NDCG@10 | MRR@10 | Recall@1 | Recall@5 | Recall@10 | Relative Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 2 Baseline** (`all-MiniLM-L6-v2`) | `0.6090` | `0.5790` | `0.5360` | `0.6480` | `0.6940` | Baseline |
| **Phase 3 In-Batch** (`checkpoint-phase3-60`) | `0.7187` | `0.6864` | `0.6260` | `0.7600` | `0.8220` | +18.01% |
| **Phase 4 BM25 Hard Neg** (`checkpoint-round1_bm25-40`) | `0.7226` | `0.6926` | `0.6380` | `0.7720` | `0.8180` | +18.65% |
| **Phase 6 Final Bi-Encoder** (`best_biencoder`) | **`0.7737`** | **`0.7427`** | **`0.6900`** | **`0.8100`** | **`0.8740`** | **+27.04%** |

---

## 5. Known Limitations & Recommendations for Downstream Modules
1. **Context Window**: Max sequence length is 512 tokens. Extremely long monolithic scripts (>2,000 lines) should be chunked by function/class boundaries before encoding.
2. **Complementary Sparse Retrieval**: While dense semantic retrieval reaches 87.4% Recall@10, exact identifier / symbol matches (e.g. `lowest_temp`, specific variable names) benefit heavily from reciprocal rank fusion (RRF) with Person B's BM25 retriever.
3. **Cross-Encoder Calibration**: Person C's cross-encoder reranker should take the top-50 candidates output by hybrid retrieval and score them for final ranking.
