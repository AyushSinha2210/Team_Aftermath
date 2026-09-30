# Ariadne Architecture — PRISM GenAI Hackathon

## 1. Overview
Ariadne is a multi-stage, CPU-efficient code retrieval and ranking system designed for Theme 01 (Agentic Code Intelligence). It features contrastive bi-encoder fine-tuning, code-aware sparse retrieval, dynamic confidence cascade routing, and incremental git-commit versioning.

## 2. Multi-Stage Pipeline Architecture

```text
Query (Natural Language / Code Intent)
                  │
                  ▼
   ┌───────────────────────────────┐
   │ Query Expansion (Synonyms)    │
   └──────────────┬────────────────┘
                  │
      ┌───────────┴────────────────────────┐
      ▼                                    ▼
Dense Retrieval (Config A)       Sparse Retrieval (CodeBM25)
(Fine-tuned Bi-Encoder,          (camelCase/snake_case split,
 384-d, NDCG@10: 0.7737)          AST token weighting)
      │                                    │
      └───────────┬────────────────────────┘
                  ▼
   ┌───────────────────────────────┐
   │ Calibrated Hybrid Fusion      │ (Dense: 0.85, Sparse: 0.15, RRF k=20)
   └──────────────┬────────────────┘
                  │
                  ▼
   ┌───────────────────────────────┐
   │   Confidence Cascade Router   │
   └──────────────┬────────────────┘
      Margin >= τ │                │ Margin < τ (Ambiguous)
      (Fast-Path) │                ▼
                  │       ┌────────────────────────────────┐
                  │       │ Cross-Encoder Reranker         │
                  │       │ (MS-MARCO + Score Caching)     │
                  │       └────────┬───────────────────────┘
                  │                ▼
                  │       ┌────────────────────────────────┐
                  │       │ AST Structural Call Graph      │
                  │       │ Resolution & Verification      │
                  │       └────────┬───────────────────────┘
                  │                ▼
                  │       ┌────────────────────────────────┐
                  │       │ Confidence Calibration         │
                  │       │ (Softmax Variance Abstention)  │
                  │       └────────┬───────────────────────┘
                  │                │
                  └───────┬────────┘
                          ▼
                Final Ranked Snippets
```

## 3. Core Architectural Components

### 3.1 Contrastive Fine-Tuned Bi-Encoder (Person A - Critical Path)
- **Base**: `all-MiniLM-L6-v2` (<35M parameters, CPU-first inference).
- **Optimization**: Multi-stage negative mining (In-batch -> BM25 hard negatives -> Iterative dense false-positive mining).
- **Validation**: Achieves **`0.7737` NDCG@10** on the CoIR `apps` benchmark.

### 3.2 Code-Aware BM25 Tokenizer (`CodeBM25`)
- Decomposes `camelCase`, `PascalCase`, `snake_case`, and alphanumeric compounds into constituent sub-tokens while preserving composite names.
- Filters syntactic boilerplate keywords to focus lexical matching on API identifiers, function names, and parameter signatures.

### 3.3 Confidence-Gated Cascade Router (`CascadeRouter`)
- Computes dense score margin $\Delta = \text{Score}_1 - \text{Score}_2$.
- **High Confidence ($\Delta \ge 0.08$)**: Immediately returns top dense results without reranking overhead, preserving Config A's high accuracy.
- **Ambiguous Queries ($\Delta < 0.08$)**: Escalate candidates to the cross-encoder and AST structural verifier.

### 3.4 AST Structural Preprocessor & Call Graph
- Tree-sitter and AST-based extraction of function signatures, docstrings, and intra-module call dependencies.
- Prefixes code snippets with structural interface headers (`# SIGNATURE`, `# SUMMARY`, `# CALLS`).

### 3.5 Code Versioning & Incremental Indexing (Person D)
- SHA-256 function-level chunk hashing.
- Unchanged chunks reuse precomputed dense embeddings directly from cache (500-snippet benchmark: 53.26s full rebuild -> <1s incremental update).
- Cosine-similarity thresholding collapses near-duplicate functions across git commits.
