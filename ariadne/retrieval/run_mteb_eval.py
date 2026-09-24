"""Run the official MTEB AppsRetrieval test task and save its real result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_evaluation(mode: str, output: Path, model_path: str | None = None) -> Path:
    import mteb

    from ariadne.retrieval.encoder import PrePostPipelineEncoder
    from ariadne.retrieval.mteb_search import HybridSearchModel

    if mode not in {"dense", "hybrid"}:
        raise ValueError("mode must be dense or hybrid")
    if model_path is not None and not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if mode == "dense":
        model = PrePostPipelineEncoder(model_name_or_path=model_path)
    elif model_path is None:
        model = HybridSearchModel()
    else:
        from ariadne.finetuning.embedder import encode

        model = HybridSearchModel(
            encoder=lambda texts: encode(texts, model_name_or_path=model_path)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--model-path", help="Local fine-tuned checkpoint; omit for A's baseline fallback")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "submission" / "appsretrieval_results.json",
    )
    args = parser.parse_args()
    path = run_evaluation(args.mode, args.output, args.model_path)
    print(f"MTEB AppsRetrieval test result saved to {path}")


if __name__ == "__main__":
    main()
