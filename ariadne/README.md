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
   pip install -r requirements.txt
   ```

2. **Download dataset** (Train/Validation splits):
   ```bash
   bash data/download.sh
   ```

3. **Verify configuration**:
   All pipeline components read from `config.yaml` as the single source of truth.
