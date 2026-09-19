# Team Aftermath — Ariadne: Agentic Code Intelligence

> **Samsung PRISM GenAI Hackathon (3rd Edition)**  
> **Theme 01:** Agentic Code Intelligence

## Project Overview

**Ariadne** is a high-performance, CPU-efficient code retrieval system built on the CoIR `apps` benchmark. It combines:
1. **Contrastive Fine-Tuned Bi-Encoder** (Person A - Critical Path): Lightweight, CPU-friendly embeddings optimized on code retrieval.
2. **Hybrid Retrieval with BM25 + Reciprocal Rank Fusion** (Person B): Combining dense semantic search with sparse lexical matching.
3. **Cross-Encoder Reranking & Confidence Calibration** (Person C): High-precision shortlist scoring with score-spread abstention and AST structural analysis.
4. **Codebase Versioning & Evolutionary Retrieval** (Person D): Function-level hashing, incremental indexing, and cross-commit near-duplicate deduplication.

---

## Repository Structure

The complete project codebase is located in the [`ariadne/`](ariadne/) directory:

```text
ariadne/
├── README.md                   # Full system setup and run instructions
├── Dockerfile                  # Container environment
├── requirements.txt            # Pinned dependencies (CPU PyTorch, MTEB, etc.)
├── config.yaml                 # Central single source of truth configuration
├── .gitignore                  # Exclusion rules
├── data/                       # Dataset handling (download.sh, raw, processed)
├── finetuning/                 # Person A: Bi-encoder contrastive fine-tuning & negative mining
├── retrieval/                  # Person B: Hybrid retriever, RRF, and MTEB submission harness
├── reranking/                  # Person C: Cross-encoder reranker, calibration & AST parser
├── versioning/                 # Person D: Incremental indexer, dedup & demo UI
├── eval/                       # Metrics (NDCG@10, MRR) & experiment results
├── submission/                 # Final appsretrieval_results.json artifact & release notes
└── docs/                       # Architecture documentation, PPT assets, demo script
```

---

## Getting Started

Navigate to the `ariadne` directory:

```bash
cd ariadne
pip install -r requirements.txt
bash data/download.sh
```

Refer to [`ariadne/README.md`](ariadne/README.md) and [`ariadne/docs/architecture.md`](ariadne/docs/architecture.md) for detailed documentation.
