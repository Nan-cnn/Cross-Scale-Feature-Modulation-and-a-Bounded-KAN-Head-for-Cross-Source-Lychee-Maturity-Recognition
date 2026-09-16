"""Select the final KAN head from the validation-only confirmation matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from experiment_engine import aggregate_results, confidence_interval_95_half_width
from run_experiments import task_protocol_fingerprint


EXPECTED_MODELS = {"KANBestObserved", "KANFactorwiseCombined"}
EXPECTED_SEEDS = {47, 48, 49, 50, 51}
KAN_KEYS = (
    "kan_layers",
    "kan_grid_size",
    "kan_spline_order",
    "hidden_dim",
    "bottleneck_dim",
    "dropout",
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _safe_name(value: str) -> str:
    text = str(value).lower().replace(".", "p")
    return re.sub(r"[^a-z0-9_+-]+", "_", text).strip("_")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_confirmation_protocol(
    config_path: Path, results_dir: Path, project_root: Path
) -> Dict[str, Any]:
    config = _load_json(config_path)
    configured_output = Path(config["output_root"])
    if not configured_output.is_absolute():
        configured_output = (project_root / configured_output).resolve()
    if configured_output != results_dir:
        raise ValueError(
            f"Confirmation config output_root resolves to {configured_output}, not {results_dir}"
        )
    if "test" in config["data"] or "test1" in config["data"]:
        raise ValueError("Confirmation protocol must not expose test or test1")
    if config.get("ablation", {}).get("enabled", True):
        raise ValueError("Confirmation protocol must have ablation.enabled=false")

    section = config["comparison"]
    model_names = {str(entry["name"]) for entry in section["models"]}
    seeds = {int(seed) for seed in section["seeds"]}
    ratios = {float(ratio) for ratio in section["ratios"]}
    if model_names != EXPECTED_MODELS or len(section["models"]) != len(EXPECTED_MODELS):
        raise ValueError(f"Expected models {sorted(EXPECTED_MODELS)}, found {sorted(model_names)}")
    if seeds != EXPECTED_SEEDS or len(section["seeds"]) != len(EXPECTED_SEEDS):
        raise ValueError(f"Expected independent seeds {sorted(EXPECTED_SEEDS)}, found {sorted(seeds)}")
    if ratios != {0.6} or len(section["ratios"]) != 1:
        raise ValueError(f"Expected confirmation ratio 0.6, found {sorted(ratios)}")
    snapshot_path = results_dir / "resolved_config_comparison.json"
    if not snapshot_path.exists() or _load_json(snapshot_path) != config:
        raise ValueError(f"Missing or mismatched confirmation configuration snapshot: {snapshot_path}")
    return config


def _validate_metric_matrix(results_dir: Path, config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    section = config["comparison"]
    ratio = float(section["ratios"][0])
    image_size = int(config["image_size"])
    expected_paths = set()
    specs: Dict[str, Dict[str, Any]] = {}
    manifest_hashes: Dict[int, set[str]] = {seed: set() for seed in EXPECTED_SEEDS}

    for entry in section["models"]:
        model_name = str(entry["name"])
        expected_spec = dict(entry["spec"])
        specs[model_name] = expected_spec
        observed_parameters = set()
        observed_sizes = set()
        for seed_value in section["seeds"]:
            seed = int(seed_value)
            run_dir = (
                results_dir
                / "comparison"
                / _safe_name(model_name)
                / f"ratio_{ratio:.2f}"
                / f"seed_{seed}"
            )
            metrics_path = run_dir / "metrics.json"
            completed_path = run_dir / "completed.json"
            manifest_path = run_dir / "train_manifest.csv"
            expected_paths.add(metrics_path.resolve())
            if not metrics_path.exists() or not completed_path.exists() or not manifest_path.exists():
                raise FileNotFoundError(f"Incomplete confirmation task: {run_dir}")
            manifest_hashes[seed].add(_sha256_file(manifest_path))
            completed = _load_json(completed_path)
            if completed.get("complete") is not True:
                raise ValueError(f"Invalid completion marker: {completed_path}")
            payload = _load_json(metrics_path)
            expected_fingerprint = task_protocol_fingerprint(
                config, "comparison", model_name, expected_spec, ratio, seed
            )
            if completed.get("protocol_fingerprint") != expected_fingerprint:
                raise ValueError(f"Completion protocol fingerprint mismatch: {completed_path}")
            if payload.get("protocol_fingerprint") != expected_fingerprint:
                raise ValueError(f"Metrics protocol fingerprint mismatch: {metrics_path}")
            expected_metadata = {
                "suite": "comparison",
                "model": model_name,
                "ratio": ratio,
                "seed": seed,
                "image_size": image_size,
            }
            for key, expected_value in expected_metadata.items():
                if payload.get(key) != expected_value:
                    raise ValueError(
                        f"{metrics_path}: expected {key}={expected_value!r}, found {payload.get(key)!r}"
                    )
            if payload.get("model_spec") != expected_spec:
                raise ValueError(f"Stale or mixed model specification in {metrics_path}")
            if payload.get("from_scratch") is not True:
                raise ValueError(f"Run is not marked from-scratch: {metrics_path}")
            if payload.get("no_pretrained_weights") is not True:
                raise ValueError(f"Run is not certified as no-pretrained: {metrics_path}")
            if set(payload.get("splits", {})) != {"validation"}:
                raise ValueError(f"Confirmation task must contain validation metrics only: {metrics_path}")
            observed_parameters.add(int(payload["parameters"]))
            observed_sizes.add(float(payload["model_size_mb"]))
        if len(observed_parameters) != 1 or len(observed_sizes) != 1:
            raise ValueError(f"Model size metadata changed across seeds for {model_name}")

    for seed, hashes in manifest_hashes.items():
        if len(hashes) != 1:
            raise ValueError(f"Paired models used different train manifests for seed {seed}")

    actual_paths = {path.resolve() for path in results_dir.rglob("metrics.json")}
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths.difference(actual_paths))
        extra = sorted(str(path) for path in actual_paths.difference(expected_paths))
        raise ValueError(f"Confirmation metric matrix mismatch; missing={missing}, extra={extra}")
    return specs


def _validate_and_group(rows: Iterable[Mapping[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    all_rows = [dict(row) for row in rows]
    if len(all_rows) != len(EXPECTED_MODELS) * len(EXPECTED_SEEDS):
        raise ValueError(f"Expected exactly 10 validation rows, found {len(all_rows)}")
    if any(row.get("suite") != "comparison" or row.get("split") != "validation" for row in all_rows):
        raise ValueError("Confirmation summary may contain comparison/validation rows only")
    validation_rows = all_rows
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in validation_rows:
        grouped.setdefault(str(row["model"]), []).append(row)

    if set(grouped) != EXPECTED_MODELS:
        raise ValueError(f"Expected confirmation models {sorted(EXPECTED_MODELS)}, found {sorted(grouped)}")
    for model_name, model_rows in grouped.items():
        seeds = {int(row["seed"]) for row in model_rows}
        if seeds != EXPECTED_SEEDS or len(model_rows) != len(EXPECTED_SEEDS):
            raise ValueError(
                f"{model_name} must contain exactly seeds {sorted(EXPECTED_SEEDS)}; found {sorted(seeds)}"
            )
        ratios = {float(row["ratio"]) for row in model_rows}
        if ratios != {0.6}:
            raise ValueError(f"{model_name} must use ratio 0.6; found {sorted(ratios)}")
    return grouped


def _summarize(grouped: Mapping[str, Sequence[Mapping[str, str]]]) -> Dict[str, Dict[str, float]]:
    summaries: Dict[str, Dict[str, float]] = {}
    for model_name, rows in grouped.items():
        values = [float(row["macro_f1"]) for row in rows]
        summaries[model_name] = {
            "n": float(len(values)),
            "macro_f1_mean": mean(values),
            "macro_f1_std": stdev(values),
            "macro_f1_ci95": confidence_interval_95_half_width(values),
            "macro_f1_se": stdev(values) / math.sqrt(len(values)),
            "parameters": float(rows[0]["parameters"]),
            "model_size_mb": float(rows[0]["model_size_mb"]),
        }
    return summaries


def _paired_difference(
    grouped: Mapping[str, Sequence[Mapping[str, str]]],
    left_name: str,
    right_name: str,
) -> Dict[str, Any]:
    left = {int(row["seed"]): float(row["macro_f1"]) for row in grouped[left_name]}
    right = {int(row["seed"]): float(row["macro_f1"]) for row in grouped[right_name]}
    seeds = sorted(set(left).intersection(right))
    differences = [left[seed] - right[seed] for seed in seeds]
    difference_mean = mean(differences)
    half_width = confidence_interval_95_half_width(differences)
    return {
        "left": left_name,
        "right": right_name,
        "seeds": seeds,
        "differences": differences,
        "mean_difference": difference_mean,
        "ci95_low": difference_mean - half_width,
        "ci95_high": difference_mean + half_width,
    }


def _select_model(
    summaries: Mapping[str, Mapping[str, float]], paired: Mapping[str, Any]
) -> tuple[str, str]:
    ranked = sorted(
        summaries,
        key=lambda name: (
            -float(summaries[name]["macro_f1_mean"]),
            float(summaries[name]["parameters"]),
            float(summaries[name]["macro_f1_std"]),
            name,
        ),
    )
    best_mean_name = ranked[0]
    if float(paired["ci95_low"]) > 0:
        return str(paired["left"]), "paired Student-t 95% CI favors the left model"
    if float(paired["ci95_high"]) < 0:
        return str(paired["right"]), "paired Student-t 95% CI favors the right model"

    best_mean = float(summaries[best_mean_name]["macro_f1_mean"])
    one_se_floor = best_mean - float(summaries[best_mean_name]["macro_f1_se"])
    eligible = [
        name for name in summaries if float(summaries[name]["macro_f1_mean"]) >= one_se_floor
    ]
    selected = min(
        eligible,
        key=lambda name: (
            float(summaries[name]["parameters"]),
            float(summaries[name]["macro_f1_std"]),
            -float(summaries[name]["macro_f1_mean"]),
            name,
        ),
    )
    return selected, "paired CI overlaps zero; applied the validation one-standard-error parsimony rule"


def _apply_to_full_config(
    config_path: Path, selected_spec: Mapping[str, Any], project_root: Path
) -> tuple[bool, Path | None]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    models = {entry["name"]: entry for entry in config["comparison"]["models"]}
    required = {"MobileNetV2KANPlus", "MobileNetV2KANHeadOnly"}
    if not required.issubset(models):
        raise ValueError(f"Target config is missing models: {sorted(required.difference(models))}")

    changed = False
    for model_name in sorted(required):
        target_spec = models[model_name]["spec"]
        for key in KAN_KEYS:
            if key not in selected_spec:
                raise ValueError(f"Selected model spec is missing {key}")
            if target_spec.get(key) != selected_spec[key]:
                target_spec[key] = selected_spec[key]
                changed = True

    if not changed:
        return False, None

    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    comparison_root = output_root / "comparison"
    comparison_artifacts = list(comparison_root.rglob("*")) if comparison_root.exists() else []
    if comparison_artifacts:
        raise RuntimeError(
            "Refusing to change the formal KAN specification after formal comparison artifacts exist. "
            f"Found {len(comparison_artifacts)} paths under {comparison_root}. "
            "Preserve those results and start the selected configuration in a new output_root."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = config_path.with_name(f"{config_path.stem}.before_confirmation_{timestamp}.json")
    with config_path.open("r", encoding="utf-8") as handle:
        original = json.load(handle)
    _atomic_write_json(backup_path, original)
    _atomic_write_json(config_path, config)
    return True, backup_path


def _write_report(
    results_dir: Path,
    summaries: Mapping[str, Mapping[str, float]],
    paired: Mapping[str, Any],
    selected_name: str,
    selected_spec: Mapping[str, Any],
    reason: str,
    apply_requested: bool,
    config_changed: bool,
    backup_path: Path | None,
) -> None:
    payload = {
        "selection_split": "validation",
        "selection_metric": "macro_f1",
        "test_metrics_used_for_selection": False,
        "summaries": summaries,
        "paired_comparison": paired,
        "selected_model": selected_name,
        "selected_spec": dict(selected_spec),
        "selection_reason": reason,
        "apply_requested": apply_requested,
        "full_config_changed": config_changed,
        "backup_path": str(backup_path) if backup_path else None,
    }
    _atomic_write_json(results_dir / "confirmation_selection.json", payload)

    lines = [
        "# KAN confirmation selection",
        "",
        "Selection used validation macro-F1 only; test and test1 were excluded.",
        "",
        "| Model | n | Validation Macro-F1 mean | SD | Student-t 95% CI half-width | Parameters |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in sorted(summaries):
        item = summaries[model_name]
        lines.append(
            f"| {model_name} | {int(item['n'])} | {item['macro_f1_mean']:.6f} | "
            f"{item['macro_f1_std']:.6f} | {item['macro_f1_ci95']:.6f} | {int(item['parameters']):,} |"
        )
    lines.extend(
        [
            "",
            f"Paired difference ({paired['left']} minus {paired['right']}): "
            f"{paired['mean_difference']:.6f}, 95% CI "
            f"[{paired['ci95_low']:.6f}, {paired['ci95_high']:.6f}].",
            "",
            f"Selected model: `{selected_name}`.",
            "",
            f"Reason: {reason}.",
            "",
            "```json",
            json.dumps(dict(selected_spec), ensure_ascii=False, indent=2),
            "```",
            "",
            f"Apply requested: {apply_requested}.",
            "",
            f"Full comparison config changed: {config_changed}.",
        ]
    )
    if backup_path:
        lines.extend(["", f"Previous full config backup: `{backup_path}`."])
    _atomic_write_text(results_dir / "CONFIRMATION_SELECTION.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="outputs/kan_confirmation_224")
    parser.add_argument("--confirmation-config", default="configs/kan_confirmation_224.json")
    parser.add_argument("--target-config", default="configs/full_224.json")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the selected KAN hyperparameters into both formal KAN comparison models",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (project_root / results_dir).resolve()
    target_config = Path(args.target_config)
    if not target_config.is_absolute():
        target_config = (project_root / target_config).resolve()
    confirmation_config = Path(args.confirmation_config)
    if not confirmation_config.is_absolute():
        confirmation_config = (project_root / confirmation_config).resolve()

    protocol = _load_confirmation_protocol(confirmation_config, results_dir, project_root)
    model_specs = _validate_metric_matrix(results_dir, protocol)
    aggregate_results(results_dir)
    all_runs_path = results_dir / "all_runs.csv"
    if not all_runs_path.exists():
        raise FileNotFoundError(f"No completed confirmation summary found in {results_dir}")

    grouped = _validate_and_group(_read_csv(all_runs_path))
    summaries = _summarize(grouped)
    paired = _paired_difference(grouped, "KANFactorwiseCombined", "KANBestObserved")
    selected_name, reason = _select_model(summaries, paired)
    selected_spec = model_specs[selected_name]

    config_changed, backup_path = (
        _apply_to_full_config(target_config, selected_spec, project_root)
        if args.apply
        else (False, None)
    )
    _write_report(
        results_dir,
        summaries,
        paired,
        selected_name,
        selected_spec,
        reason,
        bool(args.apply),
        config_changed,
        backup_path,
    )
    print(f"Selected: {selected_name}")
    print(json.dumps({key: selected_spec[key] for key in KAN_KEYS}, ensure_ascii=False, indent=2))
    print(f"Reason: {reason}")
    if args.apply:
        if config_changed:
            print(f"Updated: {target_config}")
            print(f"Backup: {backup_path}")
        else:
            print(f"Already matched selected specification: {target_config}")


if __name__ == "__main__":
    main()
