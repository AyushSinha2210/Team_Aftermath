# Ariadne 5-Minute Live Demo Script

## Structure
1. **0:00 - 1:00 Problem & Constraints**
   - Theme 01: Agentic Code Intelligence.
   - Challenge: Code retrieval across thousands of snippets with minimal GPU / CPU constraints.
2. **1:00 - 2:30 P0 Retrieval Demo**
   - Live query input in demo UI (`demo/app.py`).
   - Attribution panel showing dense vs BM25 vs cross-encoder contributions.
   - Latency benchmark on CPU.
3. **2:30 - 3:45 P1 & Bonus: Versioning & Evolutionary Retrieval**
   - Live commit change: show cache hits vs incremental re-indexing speedup.
   - Retrieval on changed code vs historical commits with deduplication.
4. **3:45 - 5:00 Ablations & Q&A**
   - Validation split metrics: Fine-tuned vs Off-the-shelf vs Hybrid + Reranker.
