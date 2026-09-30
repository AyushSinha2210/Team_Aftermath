"""Browser-facing adapter to the existing CPU retrieval and versioning modules."""
import math
import threading
import time
from pathlib import Path

from .corpus import load_functions

ROOT = Path(__file__).resolve().parents[2]


class BusyError(Exception):
    pass


def validate_request(payload: object) -> tuple[str, int]:
    if not isinstance(payload, dict):
        raise ValueError("A JSON object is required.")
    query = payload.get("query")
    if not isinstance(query, str):
        raise ValueError("Query must be text.")
    query = "".join(char for char in query if char.isprintable() or char in "\n\t").strip()
    if not query or len(query) > 2000:
        raise ValueError("Enter between 1 and 2000 searchable characters.")
    top_k = payload.get("top_k", 5)
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("top_k must be an integer between 1 and 10.")
    return query, top_k


class SearchService:
    def __init__(self, repo: Path | None = None, mode: str = "dense", cache: Path | None = None):
        if mode not in {"dense", "hybrid"}:
            raise ValueError("Unsupported retrieval mode")
        self.mode = mode
        self.documents = load_functions(repo or ROOT / "ariadne/data/voice_assistant_js")
        corpus = {key: record["code"] for key, record in self.documents.items()}
        self.lock = threading.Lock()
        if mode == "dense":
            from ariadne.versioning.incremental_index import IncrementalIndex
            self.index = IncrementalIndex(cache_path=cache or ROOT / ".cache/frontend-dense.pkl")
            self.index.update(corpus)
        else:
            from ariadne.retrieval.pipeline import HybridPipeline
            self.pipeline = HybridPipeline(corpus)

    def health(self) -> dict:
        return {"status": "ready", "mode": self.mode, "documents": len(self.documents), "api_version": 1}

    def search(self, query: str, top_k: int) -> dict:
        if not self.lock.acquire(blocking=False):
            raise BusyError("Inference is busy")
        try:
            started = time.perf_counter()
            if self.mode == "dense":
                import numpy as np
                from ariadne.finetuning.embedder import encode
                ids, embeddings = self.index.get_embeddings()
                vector = encode([query])[0]
                vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
                embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
                scores = embeddings @ vector
                order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))[:top_k]
                candidates = [{"id": ids[i], "score": float(scores[i]), "dense_score": float(scores[i]), "sparse_score": None, "sources": ["dense"]} for i in order if scores[i] >= .20]
            else:
                # Hybrid rank scores are raw RRF scores, not confidence estimates.
                candidates = [{**item, "score": item["fusion_score"]} for item in self.pipeline.retrieve(query, k=top_k) if item.get("dense_score", 0) is not None and item["dense_score"] >= .20]
            results = []
            for candidate in candidates:
                for field in ("score", "dense_score", "sparse_score"):
                    if candidate.get(field) is not None and not math.isfinite(float(candidate[field])):
                        raise ValueError("Nonfinite model output")
                results.append({**self.documents[candidate["id"]], **{key: candidate.get(key) for key in ("score", "dense_score", "sparse_score", "sources")}})
            return {"api_version": 1, "status": "success" if results else "empty", "mode": self.mode, "query": query, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "results": results}
        finally:
            self.lock.release()
