# Ariadne — Agentic Code Intelligence

> PRISM GenAI Hackathon (3rd Edition) — Theme 01: Agentic Code Intelligence

Ariadne is a specialized, CPU-friendly code retrieval system built on the CoIR `apps` benchmark. It combines fine-tuned contrastive dense embeddings, BM25 sparse retrieval via Reciprocal Rank Fusion (RRF), cross-encoder reranking with confidence calibration, and an incremental code-versioning engine.

---

## Repository Structure & Team Ownership

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
├── finetuning/                 # Person A: Bi-encoder fine-tuning (Critical Path)
│   ├── train.py
│   ├── hard_negative_mining.py
│   ├── validate.py
│   ├── configs/biencoder.yaml
│   └── checkpoints/
├── retrieval/                  # Person B: Hybrid retrieval & MTEB harness
│   ├── encoder.py
│   ├── dense_retriever.py
│   ├── sparse_retriever.py
│   ├── fusion.py
│   ├── pipeline.py
│   └── run_mteb_eval.py
├── reranking/                  # Person C: Cross-encoder reranker & calibration
│   ├── cross_encoder.py
│   ├── calibration.py
│   ├── checkpoint_eval.py
│   └── structural/
│       ├── ast_parser.py
│       └── call_graph.py
├── versioning/                 # Person D: Versioning, incremental index & demo
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

## Person B: retrieval and benchmark

The live hybrid pipeline accepts a mapping from stable document IDs to code text:

```python
from ariadne.retrieval.pipeline import HybridPipeline

pipeline = HybridPipeline({"snippet-1": "def binary_search(items, target): ..."})
top_50 = pipeline.retrieve("find an item in a sorted list", k=50)
```

Each result includes `id`, `text`, `fusion_score`, the available dense/BM25 scores,
and `sources`. Person C can pass this list directly to its reranker. The embedder
automatically uses Person A's fine-tuned checkpoint when it exists at the path
in `config.yaml`; otherwise it uses the configured pretrained baseline. An
explicit checkpoint can be supplied to the evaluation command.

Run the official AppsRetrieval test evaluation from the repository root:

```bash
python -m ariadne.retrieval.run_mteb_eval --mode hybrid
python -m ariadne.retrieval.run_mteb_eval --mode dense --model-path ariadne/finetuning/checkpoints/best_biencoder
```

The first command uses MTEB's `SearchProtocol` to score the actual BM25 + dense
+ RRF ranking. `PrePostPipelineEncoder(AbsEncoder)` supports dense-only
embedding evaluation; BM25/RRF cannot be encoded as independent vectors. The
runner writes `submission/appsretrieval_results.json` only after MTEB returns
a completed test result. No result is checked in until the benchmark runs.
The checked-in requirements use MTEB 2.21.3, which contains `AppsRetrieval`;
the earlier 1.12.50 pin did not.

The checked-in result is a measured pretrained hybrid baseline (NDCG@10
`0.06452` on the official test split), not a fine-tuned or reranked final
submission. See `submission/release_notes.md` for the exact run context.
