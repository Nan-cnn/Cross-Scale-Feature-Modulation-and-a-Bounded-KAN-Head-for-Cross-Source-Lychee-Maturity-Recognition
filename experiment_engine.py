"""Training, evaluation, checkpointing, and aggregation for experiment matrices."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from experiment_metrics import compute_binary_metrics


_T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def confidence_interval_95_half_width(values: Sequence[float]) -> float:
    """Student-t 95% CI half-width for repeated-run means."""
    numeric = [float(value) for value in values]
    if len(numeric) <= 1:
        return 0.0
    degrees_of_freedom = len(numeric) - 1
    critical = _T_CRITICAL_975.get(degrees_of_freedom, 1.96)
    return critical * stdev(numeric) / math.sqrt(len(numeric))


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(destination)


def _make_optimizer(model: nn.Module, cfg: Mapping[str, Any]):
    name = str(cfg.get("optimizer", "adamw")).lower()
    learning_rate = float(cfg.get("learning_rate", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=float(cfg.get("momentum", 0.9)),
            nesterov=True,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def _make_scheduler(optimizer, cfg: Mapping[str, Any]):
    epochs = int(cfg.get("epochs", 80))
    warmup = int(cfg.get("warmup_epochs", 5))
    initial_lr = float(cfg.get("learning_rate", 3e-4))
    minimum_lr = float(cfg.get("minimum_learning_rate", 1e-6))
    minimum_ratio = minimum_lr / max(initial_lr, 1e-12)

    def schedule(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return max(minimum_ratio, float(epoch + 1) / warmup)
        denominator = max(1, epochs - warmup - 1)
        progress = min(1.0, max(0.0, float(epoch - warmup) / denominator))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, device: torch.device, criterion: nn.Module):
    model.eval()
    labels, predictions, probabilities, paths = [], [], [], []
    loss_sum = 0.0
    for images, targets, batch_paths in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += float(criterion(logits, targets).item()) * targets.size(0)
        positive = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        labels.extend(targets.cpu().tolist())
        predictions.extend(pred.cpu().tolist())
        probabilities.extend(positive.cpu().tolist())
        paths.extend(batch_paths)

    metrics = compute_binary_metrics(labels, predictions, probabilities)
    metrics["loss"] = loss_sum / max(1, len(labels))
    records = [
        {
            "path": path,
            "label": int(label),
            "prediction": int(prediction),
            "probability_positive": float(probability),
            "confidence": float(max(probability, 1.0 - probability)),
            "correct": int(label == prediction),
        }
        for path, label, prediction, probability in zip(paths, labels, predictions, probabilities)
    ]
    return metrics, records


def _write_predictions(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "label", "prediction", "probability_positive", "confidence", "correct"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _write_history(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_balanced_accuracy",
        "val_ece",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def train_and_evaluate(
    model: nn.Module,
    train_loader,
    validation_loader,
    evaluation_loaders: Mapping[str, Any],
    device: torch.device,
    training_cfg: Mapping[str, Any],
    run_dir: str | Path,
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_metadata.json", dict(metadata))

    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training_cfg.get("label_smoothing", 0.1)))
    optimizer = _make_optimizer(model, training_cfg)
    scheduler = _make_scheduler(optimizer, training_cfg)
    epochs = int(training_cfg.get("epochs", 80))
    patience = int(training_cfg.get("patience", 12))
    accumulation = max(1, int(training_cfg.get("gradient_accumulation", 1)))
    gradient_clip = float(training_cfg.get("gradient_clip", 5.0))
    spline_regularization = float(training_cfg.get("spline_regularization", 0.0))
    amp_enabled = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    checkpoint_path = run_dir / "best_checkpoint.pth"
    history = []
    best_score = -float("inf")
    best_loss = float("inf")
    no_improvement = 0
    started = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_labels, train_predictions, train_probabilities = [], [], []
        epoch_loss = 0.0
        sample_count = 0

        for batch_index, (images, targets, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
                if spline_regularization > 0.0 and hasattr(model, "spline_regularization_loss"):
                    loss = loss + spline_regularization * model.spline_regularization_loss()
                scaled_loss = loss / accumulation

            scaler.scale(scaled_loss).backward()
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
            predictions = logits.detach().argmax(dim=1)
            batch_size = targets.size(0)
            epoch_loss += float(loss.detach().item()) * batch_size
            sample_count += batch_size
            train_labels.extend(targets.detach().cpu().tolist())
            train_predictions.extend(predictions.cpu().tolist())
            train_probabilities.extend(probabilities.cpu().tolist())

        train_metrics = compute_binary_metrics(train_labels, train_predictions, train_probabilities)
        train_metrics["loss"] = epoch_loss / max(1, sample_count)
        validation_metrics, _ = evaluate_model(model, validation_loader, device, criterion)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": learning_rate,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": validation_metrics["loss"],
                "val_accuracy": validation_metrics["accuracy"],
                "val_macro_f1": validation_metrics["macro_f1"],
                "val_balanced_accuracy": validation_metrics["balanced_accuracy"],
                "val_ece": validation_metrics["ece"],
            }
        )
        _write_history(run_dir / "history.csv", history)

        score = float(validation_metrics["macro_f1"])
        val_loss = float(validation_metrics["loss"])
        improved = score > best_score + 1e-8 or (math.isclose(score, best_score) and val_loss < best_loss)
        if improved:
            best_score, best_loss = score, val_loss
            no_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "validation_metrics": validation_metrics,
                    "metadata": dict(metadata),
                    "no_pretrained_weights": True,
                },
                checkpoint_path,
            )
        else:
            no_improvement += 1

        print(
            f"  epoch {epoch + 1:03d}/{epochs} | train loss {train_metrics['loss']:.4f} | "
            f"val F1 {score:.4f} | val acc {validation_metrics['accuracy']:.4f}"
        )
        scheduler.step()
        if no_improvement >= patience:
            print(f"  early stopping after {epoch + 1} epochs")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    split_metrics: Dict[str, Dict[str, float]] = {}
    all_loaders = {"validation": validation_loader, **dict(evaluation_loaders)}
    for split_name, loader in all_loaders.items():
        metrics, records = evaluate_model(model, loader, device, criterion)
        split_metrics[split_name] = metrics
        _write_predictions(run_dir / f"predictions_{split_name}.csv", records)

    result = {
        **dict(metadata),
        "best_epoch": int(checkpoint["epoch"]),
        "elapsed_seconds": float(time.time() - started),
        "selection_metric": "validation_macro_f1",
        "no_pretrained_weights": True,
        "splits": split_metrics,
    }
    write_json(run_dir / "metrics.json", result)
    write_json(
        run_dir / "completed.json",
        {
            "complete": True,
            "best_epoch": result["best_epoch"],
            "protocol_fingerprint": metadata.get("protocol_fingerprint"),
            "no_pretrained_weights": True,
        },
    )
    return result


def aggregate_results(output_root: str | Path) -> None:
    output_root = Path(output_root)
    metrics_files = sorted(output_root.rglob("metrics.json"))
    rows = []
    for metrics_file in metrics_files:
        with metrics_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        base = {
            "suite": payload["suite"],
            "model": payload["model"],
            "ratio": payload["ratio"],
            "seed": payload["seed"],
            "parameters": payload["parameters"],
            "model_size_mb": payload["model_size_mb"],
            "best_epoch": payload["best_epoch"],
            "elapsed_seconds": payload["elapsed_seconds"],
        }
        model_spec = payload.get("model_spec", {})
        for key in (
            "head_type",
            "kan_layers",
            "kan_grid_size",
            "kan_spline_order",
            "hidden_dim",
            "dropout",
            "use_cgfm",
            "use_mixstyle",
        ):
            base[key] = model_spec.get(key, "")
        for split_name, split_metrics in payload["splits"].items():
            row = {**base, "split": split_name}
            row.update(split_metrics)
            rows.append(row)

    if not rows:
        print("[WARN] No completed metrics found; aggregation skipped")
        return
    fields = list(rows[0].keys())
    with (output_root / "all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metrics_to_summarize = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "nll",
        "brier",
        "ece",
    ]
    grouped: Dict[tuple, list] = {}
    for row in rows:
        key = (row["suite"], row["model"], row["ratio"], row["split"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key, members in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        summary = {"suite": key[0], "model": key[1], "ratio": key[2], "split": key[3], "n": len(members)}
        for metric in metrics_to_summarize:
            values = [float(member[metric]) for member in members if member.get(metric) not in (None, "")]
            metric_mean = mean(values)
            metric_std = stdev(values) if len(values) > 1 else 0.0
            summary[f"{metric}_mean"] = metric_mean
            summary[f"{metric}_std"] = metric_std
            summary[f"{metric}_ci95"] = confidence_interval_95_half_width(values)
        summary_rows.append(summary)

    with (output_root / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


__all__ = [
    "aggregate_results",
    "confidence_interval_95_half_width",
    "evaluate_model",
    "train_and_evaluate",
    "write_json",
]
