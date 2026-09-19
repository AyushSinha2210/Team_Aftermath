# Ariadne Architecture — PRISM GenAI Hackathon

## 1. Overview
Ariadne is a multi-stage, CPU-efficient code retrieval pipeline designed for Theme 01 (Agentic Code Intelligence).

## 2. Multi-Stage Pipeline
```
Query (Natural Language)
          │
          ▼
Query Preprocessing (Tokenization, normalization)
          │
    ┌─────┴────────────────┐
    ▼                      ▼
Dense Retrieval       Sparse Retrieval
(Fine-tuned            (BM25 lexical
Bi-Encoder)            matching)
    │                      │
    └─────┬────────────────┘
          ▼
Reciprocal Rank Fusion (RRF k=60)
          │
          ▼
Cross-Encoder Reranker (Top-k shortlist)
          │
          ▼
Confidence Calibration & Attribution
          │
          ▼
Final Ranked Snippets
```

## 3. Code Versioning & Incremental Indexing (P1 & Bonus)
- SHA-256 function-level chunk hashing.
- Unchanged chunks reuse precomputed dense embeddings directly from cache.
- Only altered chunks are passed through the bi-encoder.
- Cosine-similarity thresholding collapses near-duplicate functions across git commits.
