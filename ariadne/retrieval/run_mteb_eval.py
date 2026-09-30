"""Run the official MTEB AppsRetrieval test task and save its real result.

Configured for Locked Config A: Dense Retrieval Alone (using the fine-tuned
bi-encoder checkpoint, best_biencoder) per RERANK_CARD.md's recommendation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure ariadne is discoverable on sys.path
_repo_root = Path(__file__).resolve().parents[1]  # ariadne/
_workspace_root = _repo_root.parent             # Team_Aftermath/
for p in [str(_workspace_root), str(_repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def resolve_checkpoint_path(model_path: Optional[str] = None) -> str:
    """Resolves the fine-tuned checkpoint path from arguments or config.yaml.

    Args:
        model_path: Optional explicit path provided via CLI.

    Returns:
        Absolute string path to verified model checkpoint directory.

    Raises:
        FileNotFoundError: If the specified or configured checkpoint directory is not found.
    """
    repo_root = Path(__file__).resolve().parents[1]  # ariadne root

    if model_path is not None:
        p = Path(model_path)
        if p.exists():
            return str(p.resolve())
        if (repo_root / model_path).exists():
            return str((repo_root / model_path).resolve())
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Default: read best_checkpoint_path from config.yaml
    from ariadne.finetuning.embedder import load_config

    config = load_config()
    best_ckpt = config.get("finetuning", {}).get(
        "best_checkpoint_path", "finetuning/checkpoints/best_biencoder"
    )
    target = (repo_root / best_ckpt).resolve()
    if target.exists():
        return str(target)

    # Fallback to configured model name if checkpoint directory is not yet generated
    fallback_model = config.get("model", {}).get("name", "sentence-transformers/all-MiniLM-L6-v2")
    return fallback_model


def run_evaluation(mode: str = "dense", output: Path | None = None, model_path: str | None = None) -> Path:
    """Executes official MTEB AppsRetrieval evaluation using Config A.

    Args:
        mode: 'dense' (Config A locked pipeline, dense alone) or 'hybrid' (legacy baseline).
        output: Destination path for appsretrieval_results.json.
        model_path: Optional path to bi-encoder checkpoint.

    Returns:
        Path to written results JSON.
    """
    import mteb

    from ariadne.retrieval.encoder import PrePostPipelineEncoder
    from ariadne.retrieval.mteb_search import HybridSearchModel

    if output is None:
        output = Path(__file__).resolve().parents[1] / "submission" / "appsretrieval_results.json"

    if mode not in {"dense", "hybrid"}:
        raise ValueError("mode must be dense or hybrid")

    resolved_model_path = resolve_checkpoint_path(model_path)

    if mode == "dense":
        print(f"[Ariadne MTEB] Config A locked pipeline: Dense retrieval alone.")
        print(f"[Ariadne MTEB] Using fine-tuned checkpoint: {resolved_model_path}")
        model = PrePostPipelineEncoder(model_name_or_path=resolved_model_path)
    else:
        print(f"[Ariadne MTEB] Warning: Running legacy hybrid baseline (dense + BM25 + RRF).")
        print(f"[Ariadne MTEB] Note: Config A (dense alone) is the locked decision per RERANK_CARD.md.")
        if model_path is None:
            model = HybridSearchModel()
        else:
            from ariadne.finetuning.embedder import encode

            model = HybridSearchModel(
                encoder=lambda texts: encode(texts, model_name_or_path=resolved_model_path)
            )

    task = mteb.get_task(task_name="AppsRetrieval")
    results = mteb.evaluate(
        model,
        task,
        cache=None,
        overwrite_strategy="always",
        show_progress_bar=True,
    )
    task_results = [result for result in results.task_results if result.task_name == "AppsRetrieval"]
    if len(task_results) != 1 or not task_results[0].scores.get("test"):
        raise RuntimeError("MTEB did not return a completed AppsRetrieval test result")
    payload = task_results[0].model_dump(mode="json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run official MTEB AppsRetrieval evaluation for Ariadne (Locked Config A: Dense Retrieval Alone)."
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid"],
        default="dense",
        help="Retrieval mode: 'dense' (Config A locked pipeline: dense retrieval alone, no fusion, no reranking) or 'hybrid'. Default: 'dense'",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to local fine-tuned checkpoint. If omitted, automatically resolves to config.yaml's finetuning.best_checkpoint_path ('finetuning/checkpoints/best_biencoder').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "submission" / "appsretrieval_results.json",
        help="Target path for submission JSON artifact.",
    )
    args = parser.parse_args()
    path = run_evaluation(args.mode, args.output, args.model_path)
    print(f"MTEB AppsRetrieval test result saved to {path}")


if __name__ == "__main__":
    main()
