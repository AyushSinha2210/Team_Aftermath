# AppsRetrieval Baseline Evaluation — Ariadne (Theme 01)

## PRISM GenAI Hackathon (3rd Edition)

- **Artifact**: `appsretrieval_results.json` is the unmodified MTEB `TaskResult` from a completed test run.
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` pretrained fallback. No fine-tuned checkpoint was present in this checkout.
- **Ranking**: Dense cosine retrieval + code-aware BM25 + reciprocal rank fusion through MTEB's `SearchProtocol`. Cross-encoder reranking was not used in this result.
- **Benchmark**: MTEB `AppsRetrieval`, test split, dataset revision `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`, MTEB 2.21.3.
- **Scores**: NDCG@10 `0.06452`; MRR@10 `0.050888`; recall@10 `0.10890`.
- **Reproduce**: `python -m ariadne.retrieval.run_mteb_eval --mode hybrid` from the repository root, using the pinned dependencies in `ariadne/requirements.txt`.

This is a measured baseline, not the team's locked final configuration. Person C's valid-split comparison must choose the final pipeline, and Person A's checkpoint must be supplied before claiming a fine-tuned result. Rerun the official test only after that decision; replace this artifact and update these notes together.
