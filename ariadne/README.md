# Ariadne — Agentic Code Intelligence

> PRISM GenAI Hackathon (3rd Edition) — Theme 01: Agentic Code Intelligence

Ariadne is a specialized, CPU-friendly code retrieval system built on the CoIR `apps` benchmark. It combines fine-tuned contrastive dense embeddings, BM25 sparse retrieval via Reciprocal Rank Fusion (RRF), cross-encoder reranking with confidence calibration, and an incremental code-versioning engine.

---

## Repository Structure & Component Architecture

```
ariadne/
├── README.md
├── Dockerfile
├── requirements.txt
├── config.yaml
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── download.sh
├── finetuning/                 # Bi-encoder fine-tuning (Critical Path)
│   ├── train.py
│   ├── hard_negative_mining.py
│   ├── validate.py
│   ├── configs/biencoder.yaml
│   └── checkpoints/
├── retrieval/                  # Hybrid retrieval & MTEB harness
│   ├── encoder.py
│   ├── dense_retriever.py
│   ├── sparse_retriever.py
│   ├── fusion.py
│   ├── pipeline.py
│   └── run_mteb_eval.py
├── reranking/                  # Cross-encoder reranker & calibration
│   ├── cross_encoder.py
│   ├── calibration.py
│   ├── checkpoint_eval.py
│   └── structural/
│       ├── ast_parser.py
│       └── call_graph.py
├── versioning/                 # Versioning, incremental index & demo
│   ├── content_hash.py
│   ├── incremental_index.py
│   ├── dedup.py
│   ├── evolutionary_retrieval.py
│   └── demo/
│       ├── app.py
│       └── assets/
├── eval/                       # Shared: Evaluation metrics & logs
│   ├── metrics.py
│   └── results/
├── submission/                 # Final submission artifacts
│   ├── appsretrieval_results.json
│   └── release_notes.md
└── docs/                       # Documentation & presentations
    ├── architecture.md
    ├── ppt/
    └── demo_script.md
```

---

## Quick Start

1. **Install dependencies**:
   ```bash
   # Use Python 3.11 for the pinned PyTorch and NumPy versions.
   pip install -r requirements.txt
   ```

2. **Download dataset** (Train/Validation splits):
   ```bash
   bash data/download.sh
   ```

3. **Verify configuration**:
   All pipeline components read from `config.yaml` as the single source of truth.

## Module Usage & Developer API

### Component 1: Fine-Tuning & Embedder
```python
from ariadne.finetuning.embedder import encode

# Vectorize code snippets using the fine-tuned bi-encoder (384-d, float32)
embeddings = encode(["def quicksort(arr): ...", "class UserAuth: ..."])
```

### Component 2: Hybrid Retrieval & Pipeline
The live hybrid pipeline accepts a mapping from stable document IDs to code text:
```python
from ariadne.retrieval.pipeline import HybridPipeline

pipeline = HybridPipeline({"snippet-1": "def binary_search(items, target): ..."})
# Perform calibrated hybrid search (dense + CodeBM25 with RRF)
top_50 = pipeline.retrieve("find an item in a sorted list", k=50, use_hyde=False)
```

### Component 3: Cascade Router & Cross-Encoder Reranking
```python
from ariadne.reranking.cascade_router import CascadeRouter
from ariadne.reranking.cross_encoder import rerank

router = CascadeRouter(margin_threshold=0.08)
decision = router.route(top_50)

if decision.fast_path:
    final_results = decision.candidates
else:
    # Escalate to CPU cross-encoder only on low-margin ambiguous queries
    final_results = rerank("find an item in a sorted list", decision.candidates)
```

### Component 4: Incremental Indexing & Versioning
```python
from ariadne.versioning.incremental_index import IncrementalIndex

index = IncrementalIndex()
# Initial build
index.build({"file_1": "code chunk 1", "file_2": "code chunk 2"})

# Sub-second incremental update on commit
report = index.update({"file_1": "code chunk 1", "file_2": "code chunk 2 modified"})
print(report)  # {'added': 0, 'changed': 1, 'removed': 0, 're_embedded': 1}
```

---

## Benchmark Evaluation

Run the comparative evaluation across all 5 configurations:
```bash
# Rapid 50-query validation:
python ariadne/reranking/checkpoint_eval.py --limit 50

# Full validation split:
python ariadne/reranking/checkpoint_eval.py
```

Run the MTEB benchmark harness:
```bash
python -m ariadne.retrieval.run_mteb_eval --mode hybrid
python -m ariadne.retrieval.run_mteb_eval --mode dense --model-path ariadne/finetuning/checkpoints/best_biencoder
```

Run test suite:
```bash
python -m pytest -q
```

