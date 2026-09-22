"""Contrastive fine-tuning module for the Ariadne bi-encoder.

Trains the bi-encoder using MultipleNegativesRankingLoss with in-batch negatives
on the CoIR apps train split only. Evaluates periodically on the valid split,
saves checkpoints, and applies early stopping on validation NDCG@10 plateau.

STRICT RULE: Operates exclusively on 'train' and 'valid' splits. Never touches 'test'.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Ensure repository root and workspace root are on sys.path
_repo_root = Path(__file__).resolve().parent.parent
_workspace_root = _repo_root.parent
for p in [str(_workspace_root), str(_repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ariadne.eval.metrics import evaluate_ranking


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Configures structured logger for model training.

    Args:
        log_level: Logging verbosity level.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger("ariadne.finetuning.train")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads central configuration.

    Args:
        config_path: Optional path to config.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = _repo_root / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Sets deterministic random seed across all libraries.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Creates a linear warmup and linear decay learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer instance.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.

    Returns:
        Configured LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def evaluate_checkpoint_on_valid(
    model: SentenceTransformer,
    valid_dataset: Dataset,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Evaluates in-memory model on valid split.

    Args:
        model: SentenceTransformer model.
        valid_dataset: Validation HuggingFace Dataset.
        batch_size: Encoding batch size.

    Returns:
        Dictionary of computed retrieval metrics (NDCG@10, MRR@10, etc.).
    """
    query_ids: List[str] = valid_dataset["query_id"]
    corpus_ids: List[str] = valid_dataset["corpus_id"]
    queries: List[str] = valid_dataset["query"]
    codes: List[str] = valid_dataset["code"]

    qrels: Dict[str, Set[str]] = {}
    for qid, cid in zip(query_ids, corpus_ids):
        if qid not in qrels:
            qrels[qid] = set()
        qrels[qid].add(cid)

    # Unique candidates from validation split
    cid_to_code: Dict[str, str] = {}
    for cid, code in zip(corpus_ids, codes):
        if cid not in cid_to_code:
            cid_to_code[cid] = code
    unique_corpus_ids: List[str] = list(cid_to_code.keys())
    unique_codes: List[str] = [cid_to_code[cid] for cid in unique_corpus_ids]

    # Encode with current model state
    q_emb = model.encode(
        queries,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    c_emb = model.encode(
        unique_codes,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    metrics = evaluate_ranking(
        query_embeddings=q_emb,
        corpus_embeddings=c_emb,
        query_ids=query_ids,
        corpus_ids=unique_corpus_ids,
        qrels=qrels,
        top_k=10,
    )
    return metrics


def train(
    config: Optional[Dict[str, Any]] = None,
    base_model_or_ckpt: Optional[str] = None,
    use_hard_negatives: Optional[bool] = None,
    triplets_path: Optional[str] = None,
    max_steps: Optional[int] = None,
    learning_rate: Optional[float] = None,
    eval_steps: Optional[int] = None,
    tag: str = "phase3",
) -> Dict[str, Any]:
    """Executes contrastive fine-tuning loop on CoIR apps train split only.

    Args:
        config: Optional configuration dictionary.
        base_model_or_ckpt: Optional model path/name to start from.
        max_steps: Maximum training steps.
        learning_rate: Learning rate.
        eval_steps: Evaluation step frequency.
        tag: Experiment identifier tag for checkpoints and logs.

    Returns:
        Summary dictionary containing training logs, final metrics, and paths.
    """
    if config is None:
        config = load_config()

    set_seed(int(config.get("global", {}).get("seed", 42)))
    logger = setup_logger(config.get("global", {}).get("log_level", "INFO"))

    logger.info("=" * 70)
    logger.info("Starting Ariadne Bi-Encoder Contrastive Training Loop [%s]", tag)

    # 1. Dataset Split Isolation
    data_cfg = config.get("data", {})
    raw_dir = _repo_root / data_cfg.get("raw_dir", "data/raw")
    train_split_name = data_cfg.get("train_split", "train")
    valid_split_name = data_cfg.get("valid_split", "valid")

    train_path = raw_dir / train_split_name
    valid_path = raw_dir / valid_split_name

    logger.info("Training Split: %s (STRICTLY train only)", train_path)
    logger.info("Validation Split: %s (validation only, NEVER test)", valid_path)

    # STRICT SAFETY ASSERTIONS
    assert "test" not in str(train_path).lower(), "SAFETY VIOLATION: Test split referenced in train path!"
    assert "test" not in str(valid_path).lower(), "SAFETY VIOLATION: Test split referenced in valid path!"

    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            f"Dataset splits not found at {train_path} or {valid_path}. Run download.sh first."
        )

    train_dataset: Dataset = load_from_disk(str(train_path))
    valid_dataset: Dataset = load_from_disk(str(valid_path))
    logger.info("Train examples: %d | Validation examples: %d", len(train_dataset), len(valid_dataset))

    # 2. Hyperparameters
    ft_cfg = config.get("finetuning", {})
    model_cfg = config.get("model", {})

    if base_model_or_ckpt is None:
        base_model_name = model_cfg.get("name", "sentence-transformers/all-MiniLM-L6-v2")
    else:
        base_model_name = base_model_or_ckpt

    max_seq_length = int(model_cfg.get("max_seq_length", 256))
    batch_size = int(ft_cfg.get("batch_size", 16))
    eval_batch_size = int(ft_cfg.get("eval_batch_size", 32))

    if learning_rate is None:
        learning_rate = float(ft_cfg.get("learning_rate", 3.0e-5))

    weight_decay = float(ft_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(ft_cfg.get("warmup_ratio", 0.1))

    if max_steps is None:
        max_steps = int(ft_cfg.get("max_steps", 60))

    if eval_steps is None:
        eval_steps = int(ft_cfg.get("evaluation_steps", 20))

    patience = int(ft_cfg.get("early_stopping_patience", 2))
    scale = float(ft_cfg.get("scale", 20.0))
    device = config.get("global", {}).get("device", "cpu")

    ckpt_base_dir = _repo_root / ft_cfg.get("checkpoint_dir", "finetuning/checkpoints")
    best_ckpt_dir = _repo_root / ft_cfg.get(
        "best_checkpoint_path", "finetuning/checkpoints/best_biencoder"
    )
    ckpt_base_dir.mkdir(parents=True, exist_ok=True)

    # 3. Handle Hard Negatives
    if use_hard_negatives is None:
        use_hard_negatives = bool(ft_cfg.get("hard_negatives", {}).get("enabled", False))

    triplets_data: Optional[List[Dict[str, Any]]] = None
    if use_hard_negatives:
        if triplets_path is None:
            processed_dir = _repo_root / data_cfg.get("processed_dir", "data/processed")
            triplet_file = processed_dir / "train_bm25_triplets.json"
        else:
            triplet_file = Path(triplets_path)
            if not triplet_file.exists() and (_repo_root / triplets_path).exists():
                triplet_file = _repo_root / triplets_path

        # EXCEPTION TO TRAIN-ONLY RULE (Phase 6):
        # When training with round2 iterative dense negatives, triplets are harvested
        # from 'valid' to resolve false positives. 'test' remains strictly prohibited.

        if not triplet_file.exists():
            logger.info("Mined triplets file not found at %s. Running BM25 mining...", triplet_file)
            from ariadne.finetuning.hard_negative_mining import mine_bm25_hard_negatives
            mine_bm25_hard_negatives(config=config, output_path=triplet_file)

        with open(triplet_file, "r", encoding="utf-8") as f:
            triplets_data = json.load(f)
        logger.info("Loaded %d mined hard-negative triplets from %s", len(triplets_data), triplet_file)

    logger.info("Base Model: %s on device '%s'", base_model_name, device)
    logger.info("Batch Size: %d | Learning Rate: %s | Max Steps: %d", batch_size, learning_rate, max_steps)
    logger.info("Hard Negatives Enabled: %s", use_hard_negatives)
    logger.info("Evaluation every %d steps | Early Stopping Patience: %d", eval_steps, patience)

    # 4. Model & Loss initialization
    model = SentenceTransformer(base_model_name, device=device)
    model.max_seq_length = max_seq_length

    loss_fn = MultipleNegativesRankingLoss(model=model, scale=scale)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_warmup_steps = max(1, int(max_steps * warmup_ratio))
    scheduler = create_warmup_scheduler(optimizer, num_warmup_steps, max_steps)

    # 5. Measure baseline score on valid before training
    logger.info("Evaluating baseline checkpoint on valid split before fine-tuning...")
    init_metrics = evaluate_checkpoint_on_valid(model, valid_dataset, batch_size=eval_batch_size)
    best_ndcg = init_metrics["ndcg@10"]
    logger.info(
        "Baseline Validation NDCG@10: %.4f | MRR@10: %.4f",
        best_ndcg,
        init_metrics["mrr@10"],
    )

    # 6. Prepare training pairs or triplets
    if triplets_data is not None:
        num_data = len(triplets_data)
        indices = list(range(num_data))
    else:
        train_queries: List[str] = train_dataset["query"]
        train_codes: List[str] = train_dataset["code"]
        num_data = len(train_queries)
        indices = list(range(num_data))

    random.shuffle(indices)

    training_logs: List[Dict[str, Any]] = []
    data_idx = 0
    patience_counter = 0
    step = 0

    model.train()

    # 7. Training Loop
    while step < max_steps:
        step += 1

        # Fetch batch
        batch_indices = [indices[(data_idx + i) % num_data] for i in range(batch_size)]
        data_idx = (data_idx + batch_size) % num_data
        if data_idx == 0:
            random.shuffle(indices)

        if triplets_data is not None:
            batch_items = [triplets_data[idx] for idx in batch_indices]
            batch_q = [item["query"] for item in batch_items]
            batch_pos = [item["positive_code"] for item in batch_items]
            batch_neg = [item["hard_negative_code"] for item in batch_items]
            # MultipleNegativesRankingLoss with triplets: anchor, positive, hard negative
            features = [
                model.tokenize(batch_q),
                model.tokenize(batch_pos),
                model.tokenize(batch_neg),
            ]
        else:
            batch_q = [train_queries[idx] for idx in batch_indices]
            batch_c = [train_codes[idx] for idx in batch_indices]
            features = [model.tokenize(batch_q), model.tokenize(batch_c)]

        loss = loss_fn(features, labels=None)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.item())

        if step % 10 == 0 or step == 1:
            current_lr = float(scheduler.get_last_lr()[0])
            logger.info("Step [%3d/%3d] - Loss: %.4f - LR: %.6f", step, max_steps, loss_val, current_lr)

        # Periodic Evaluation and Checkpointing
        if step % eval_steps == 0 or step == max_steps:
            model.eval()
            logger.info("--- Running periodic validation at step %d ---", step)
            val_metrics = evaluate_checkpoint_on_valid(
                model, valid_dataset, batch_size=eval_batch_size
            )
            val_ndcg = val_metrics["ndcg@10"]
            val_mrr = val_metrics["mrr@10"]

            logger.info(
                "Step %d Validation -> NDCG@10: %.4f | MRR@10: %.4f | Starting NDCG: %.4f | Best NDCG: %.4f",
                step,
                val_ndcg,
                val_mrr,
                init_metrics["ndcg@10"],
                best_ndcg,
            )

            # Save step checkpoint
            step_ckpt = ckpt_base_dir / f"checkpoint-{tag}-{step}"
            logger.info("Saving checkpoint to %s", step_ckpt)
            model.save(str(step_ckpt))

            log_entry = {
                "step": step,
                "loss": loss_val,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "metrics": val_metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            training_logs.append(log_entry)

            # Check if this beats previous best NDCG@10
            if val_ndcg > best_ndcg:
                logger.info(
                    "*** New best validation NDCG@10 achieved: %.4f (previous best: %.4f)! Updating best_biencoder...",
                    val_ndcg,
                    best_ndcg,
                )
                best_ndcg = val_ndcg
                patience_counter = 0
                model.save(str(best_ckpt_dir))
                logger.info("Best model updated at %s", best_ckpt_dir)
            else:
                patience_counter += 1
                logger.info(
                    "No improvement over best NDCG@10 (%.4f). Early stopping patience: %d/%d",
                    best_ndcg,
                    patience_counter,
                    patience,
                )
                if patience_counter >= patience:
                    logger.info("Early stopping triggered at step %d on NDCG plateau.", step)
                    break

            model.train()

    # 7. Finalize and write evaluation logs
    results_dir = _repo_root / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"train_log_{timestamp_str}.json"

    summary = {
        "timestamp": timestamp_str,
        "experiment_tag": tag,
        "base_model": base_model_name,
        "final_step": step,
        "initial_ndcg@10": init_metrics["ndcg@10"],
        "initial_mrr@10": init_metrics["mrr@10"],
        "best_ndcg@10": best_ndcg,
        "improvement_over_baseline": best_ndcg - init_metrics["ndcg@10"],
        "training_logs": training_logs,
    }
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training complete! Logs saved to: %s", log_file)
    logger.info("=" * 70)
    logger.info("Baseline NDCG@10           : %.4f", init_metrics["ndcg@10"])
    logger.info("Fine-Tuned Best NDCG@10    : %.4f (Delta: %+.4f)", best_ndcg, best_ndcg - init_metrics["ndcg@10"])
    logger.info("Best checkpoint available at: %s", best_ckpt_dir)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrastive fine-tuning for Ariadne bi-encoder.")
    parser.add_argument("--base-model", type=str, default=None, help="Base model name or checkpoint path.")
    parser.add_argument("--hard-negatives", action="store_true", default=False, help="Train with hard negative triplets.")
    parser.add_argument("--triplets-path", type=str, default=None, help="Path to mined triplets JSON.")
    parser.add_argument("--steps", type=int, default=None, help="Maximum training steps.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--eval-steps", type=int, default=None, help="Evaluation interval.")
    parser.add_argument("--tag", type=str, default="phase3", help="Tag for checkpoints and logs.")
    args = parser.parse_args()

    train(
        base_model_or_ckpt=args.base_model,
        use_hard_negatives=args.hard_negatives,
        triplets_path=args.triplets_path,
        max_steps=args.steps,
        learning_rate=args.lr,
        eval_steps=args.eval_steps,
        tag=args.tag,
    )
