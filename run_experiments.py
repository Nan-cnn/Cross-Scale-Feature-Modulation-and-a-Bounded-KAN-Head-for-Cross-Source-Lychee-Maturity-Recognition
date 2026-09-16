"""Run comparison and ablation matrices; no single-model training path is used."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from experiment_data import (
    balanced_nested_subset,
    build_transforms,
    make_loader,
    scan_prefix_dataset,
    seed_everything,
    write_manifest,
)
from experiment_engine import aggregate_results, train_and_evaluate, write_json
from model.factory import build_model, model_size_megabytes, trainable_parameter_count


def task_protocol_fingerprint(
    config: Mapping[str, Any],
    suite: str,
    model_name: str,
    model_spec: Mapping[str, Any],
    ratio: float,
    seed: int,
) -> str:
    """Stable signature used to prevent stale completed tasks from being resumed."""
    payload = {
        "schema_version": 1,
        "suite": str(suite),
        "model": str(model_name),
        "model_spec": dict(model_spec),
        "ratio": float(ratio),
        "seed": int(seed),
        "image_size": int(config["image_size"]),
        "robust_augmentation": bool(config.get("robust_augmentation", True)),
        "data": config["data"],
        "training": config["training"],
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_name(value: Any) -> str:
    text = str(value).lower().replace(".", "p")
    return re.sub(r"[^a-z0-9_+-]+", "_", text).strip("_")


def _validate_no_pretrained_keys(value: Any, location: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in {"weights", "pretrained", "checkpoint"}:
                raise ValueError(f"Forbidden pretrained/checkpoint key at {location}.{key}")
            _validate_no_pretrained_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_pretrained_keys(child, f"{location}[{index}]")


def _comparison_models(config: Mapping[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    models = []
    for entry in config["comparison"]["models"]:
        name = str(entry["name"])
        spec = deepcopy(entry["spec"])
        models.append((name, spec))
    if len(models) < 2:
        raise ValueError("Comparison suite must contain at least two models")
    return models


def _ablation_models(config: Mapping[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    section = config["ablation"]
    reference = deepcopy(section["reference"])
    models: List[Tuple[str, Dict[str, Any]]] = [("reference", reference)]
    seen = {json.dumps(reference, sort_keys=True)}
    for factor, values in section["factors"].items():
        for value in values:
            spec = deepcopy(reference)
            spec[factor] = value
            signature = json.dumps(spec, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            models.append((f"{_safe_name(factor)}_{_safe_name(value)}", spec))
    if len(models) < 2:
        raise ValueError("Ablation suite must contain the reference and at least one variant")
    return models


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    _validate_no_pretrained_keys(config)
    if int(config.get("image_size", 224)) != 224:
        print(f"[WARN] image_size={config['image_size']}; the recommended full protocol uses 224")
    return config


def _unique_specs(config: Mapping[str, Any], suites: Sequence[str]):
    collected, seen = [], set()
    if "comparison" in suites and config.get("comparison", {}).get("enabled", True):
        collected.extend(_comparison_models(config))
    if "ablation" in suites and config.get("ablation", {}).get("enabled", True):
        collected.extend(_ablation_models(config))
    for name, spec in collected:
        signature = json.dumps(spec, sort_keys=True)
        if signature not in seen:
            seen.add(signature)
            yield name, spec


def dry_run(config: Mapping[str, Any], suites: Sequence[str]) -> None:
    image_size = int(config["image_size"])
    print(f"Dry-run at {image_size}x{image_size}; every model is randomly initialized")
    for name, spec in _unique_specs(config, suites):
        model = build_model(spec, num_classes=2).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, image_size, image_size))
        if tuple(output.shape) != (1, 2):
            raise RuntimeError(f"{name}: expected output (1, 2), got {tuple(output.shape)}")
        print(
            f"  {name:<34} output={tuple(output.shape)} "
            f"params={trainable_parameter_count(model):,} size={model_size_megabytes(model):.2f} MB"
        )
        del model, output
        gc.collect()
    print("Dry-run complete")


def _suite_tasks(config: Mapping[str, Any], suites: Sequence[str]):
    if "comparison" in suites and config.get("comparison", {}).get("enabled", True):
        section = config["comparison"]
        for name, spec in _comparison_models(config):
            for ratio in section["ratios"]:
                for seed in section["seeds"]:
                    yield "comparison", name, spec, float(ratio), int(seed)
    if "ablation" in suites and config.get("ablation", {}).get("enabled", True):
        section = config["ablation"]
        ratio = float(section["ratio"])
        for name, spec in _ablation_models(config):
            for seed in section["seeds"]:
                yield "ablation", name, spec, ratio, int(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full_224.json")
    parser.add_argument("--suites", choices=["all", "comparison", "ablation"], default="all")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="Build and forward every unique model without training")
    parser.add_argument("--no-resume", action="store_true", help="Fail rather than skip completed runs")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config_path = _resolve_path(project_root, args.config)
    config = _load_config(config_path)
    suites = ["comparison", "ablation"] if args.suites == "all" else [args.suites]
    if args.dry_run:
        dry_run(config, suites)
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    data_cfg = config["data"]
    prefix_to_label = data_cfg["prefix_to_label"]
    train_samples = scan_prefix_dataset(_resolve_path(project_root, data_cfg["train"]), prefix_to_label)
    validation_samples = scan_prefix_dataset(_resolve_path(project_root, data_cfg["validation"]), prefix_to_label)
    optional_splits = {}
    for split_name in ("test", "test1"):
        split_value = data_cfg.get(split_name)
        if split_value:
            optional_splits[split_name] = scan_prefix_dataset(
                _resolve_path(project_root, split_value), prefix_to_label
            )

    image_size = int(config["image_size"])
    train_transform, eval_transform = build_transforms(
        image_size, robust_augmentation=bool(config.get("robust_augmentation", True))
    )
    training_cfg = config["training"]
    batch_size = int(training_cfg.get("batch_size", 8))
    workers = int(training_cfg.get("workers", 0))
    output_root = _resolve_path(project_root, config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    # Preserve the configuration that produced an already-populated suite before
    # the legacy resolved_config.json is replaced by a later suite invocation.
    legacy_config_path = output_root / "resolved_config.json"
    if legacy_config_path.exists():
        with legacy_config_path.open("r", encoding="utf-8") as handle:
            legacy_config = json.load(handle)
        for existing_suite in ("ablation", "comparison"):
            suite_dir = output_root / existing_suite
            suite_snapshot = output_root / f"resolved_config_{existing_suite}.json"
            if suite_dir.exists() and not suite_snapshot.exists():
                write_json(suite_snapshot, legacy_config)

    write_json(output_root / "resolved_config.json", config)
    for suite in suites:
        write_json(output_root / f"resolved_config_{suite}.json", config)

    tasks = list(_suite_tasks(config, suites))
    if not tasks:
        raise RuntimeError("No experiment tasks were generated")
    print(f"Device: {device}; tasks: {len(tasks)}; image size: {image_size}")

    for task_index, (suite, model_name, model_spec, ratio, seed) in enumerate(tasks, start=1):
        run_dir = output_root / suite / _safe_name(model_name) / f"ratio_{ratio:.2f}" / f"seed_{seed}"
        completed = run_dir / "completed.json"
        protocol_fingerprint = task_protocol_fingerprint(
            config, suite, model_name, model_spec, ratio, seed
        )
        if completed.exists():
            if args.no_resume:
                raise FileExistsError(f"Completed run already exists: {run_dir}")
            with completed.open("r", encoding="utf-8") as handle:
                completion_payload = json.load(handle)
            existing_fingerprint = completion_payload.get("protocol_fingerprint")
            if existing_fingerprint and existing_fingerprint != protocol_fingerprint:
                raise RuntimeError(f"Completed task protocol mismatch: {run_dir}")
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                with metrics_path.open("r", encoding="utf-8") as handle:
                    metrics_payload = json.load(handle)
                metrics_fingerprint = metrics_payload.get("protocol_fingerprint")
                if metrics_fingerprint and metrics_fingerprint != protocol_fingerprint:
                    raise RuntimeError(f"Completed metrics protocol mismatch: {metrics_path}")
            if not existing_fingerprint:
                print(f"[WARN] legacy completion has no protocol fingerprint: {run_dir}")
            print(f"[{task_index}/{len(tasks)}] skip completed {suite}/{model_name}/r={ratio:.2f}/s={seed}")
            continue

        print(f"[{task_index}/{len(tasks)}] {suite}/{model_name}/r={ratio:.2f}/s={seed}")
        seed_everything(seed, deterministic=bool(training_cfg.get("deterministic", True)))
        selected_train = balanced_nested_subset(train_samples, ratio, seed)
        write_manifest(run_dir / "train_manifest.csv", selected_train, project_root)
        train_loader = make_loader(
            selected_train, train_transform, batch_size, True, workers, seed
        )
        validation_loader = make_loader(
            validation_samples, eval_transform, batch_size, False, workers, seed + 1
        )
        evaluation_loaders = {
            split_name: make_loader(samples, eval_transform, batch_size, False, workers, seed + offset + 2)
            for offset, (split_name, samples) in enumerate(optional_splits.items())
        }

        model = None
        try:
            model = build_model(model_spec, num_classes=2)
            metadata = {
                "suite": suite,
                "model": model_name,
                "model_spec": model_spec,
                "ratio": ratio,
                "seed": seed,
                "image_size": image_size,
                "train_samples": len(selected_train),
                "validation_samples": len(validation_samples),
                "parameters": trainable_parameter_count(model),
                "model_size_mb": model_size_megabytes(model),
                "from_scratch": True,
                "protocol_fingerprint": protocol_fingerprint,
            }
            train_and_evaluate(
                model,
                train_loader,
                validation_loader,
                evaluation_loaders,
                device,
                training_cfg,
                run_dir,
                metadata,
            )
        except KeyboardInterrupt:
            raise
        except Exception as error:
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                run_dir / "failure.json",
                {
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "suite": suite,
                    "model": model_name,
                    "ratio": ratio,
                    "seed": seed,
                    "protocol_fingerprint": protocol_fingerprint,
                },
            )
            print(f"[ERROR] run failed and was recorded: {error}")
        finally:
            del model, train_loader, validation_loader, evaluation_loaders
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        aggregate_results(output_root)

    aggregate_results(output_root)
    print(f"Experiment matrix finished. Results: {output_root}")


if __name__ == "__main__":
    main()
