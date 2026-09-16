"""Validate the 35-run banana matrix and generate publication-ready result tables.

The frozen KAN configuration comes from the lychee validation-only confirmation.
Banana test metrics are used for final reporting, never for model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from scipy import stats

from experiment_engine import aggregate_results, confidence_interval_95_half_width
from run_experiments import task_protocol_fingerprint


EXPECTED_MODELS = (
    "MobileNetV2KANPlus",
    "MobileNetV2KANHeadOnly",
    "MobileNetV2",
    "ShuffleNetV2",
    "ResNet50",
    "VGG16",
    "EfficientNetB3",
)
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_RATIO = 1.0
EXPECTED_SPLITS = {"validation", "test"}
MAIN_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "roc_auc",
    "pr_auc",
    "nll",
    "brier",
    "ece",
)


def _safe_name(value: str) -> str:
    text = str(value).lower().replace(".", "p")
    return re.sub(r"[^a-z0-9_+-]+", "_", text).strip("_")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_mean_sd(mean_value: float, sd_value: float) -> str:
    return f"{mean_value:.4f} ± {sd_value:.4f}"


def _load_protocol(config_path: Path, results_dir: Path, project_root: Path) -> Dict[str, Any]:
    config = _load_json(config_path)
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    if output_root != results_dir:
        raise ValueError(f"Config output_root resolves to {output_root}, not {results_dir}")
    if int(config["image_size"]) != 224:
        raise ValueError("The banana publication protocol must use 224x224 inputs")
    if "test1" in config["data"]:
        raise ValueError("Banana has no independent test1 split; do not label test as cross-source")
    if config.get("ablation", {}).get("enabled", True):
        raise ValueError("Banana replication must not repeat KAN ablation")

    comparison = config["comparison"]
    names = tuple(str(entry["name"]) for entry in comparison["models"])
    if names != EXPECTED_MODELS:
        raise ValueError(f"Expected models {EXPECTED_MODELS}, found {names}")
    if tuple(int(seed) for seed in comparison["seeds"]) != EXPECTED_SEEDS:
        raise ValueError(f"Expected seeds {EXPECTED_SEEDS}")
    if [float(ratio) for ratio in comparison["ratios"]] != [EXPECTED_RATIO]:
        raise ValueError("Banana replication must contain ratio=1.0 only")
    return config


def _validate_matrix(config: Mapping[str, Any], results_dir: Path) -> None:
    snapshot_path = results_dir / "resolved_config_comparison.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing training configuration snapshot: {snapshot_path}")
    if _load_json(snapshot_path) != config:
        raise ValueError("Training snapshot differs from configs/banana_224.json")

    expected_metric_paths = set()
    manifest_hashes: Dict[int, set[str]] = {seed: set() for seed in EXPECTED_SEEDS}
    train_counts: Dict[int, set[int]] = {seed: set() for seed in EXPECTED_SEEDS}
    model_specs = {str(entry["name"]): dict(entry["spec"]) for entry in config["comparison"]["models"]}

    for model_name in EXPECTED_MODELS:
        spec = model_specs[model_name]
        for seed in EXPECTED_SEEDS:
            run_dir = (
                results_dir
                / "comparison"
                / _safe_name(model_name)
                / "ratio_1.00"
                / f"seed_{seed}"
            )
            required = (
                run_dir / "completed.json",
                run_dir / "metrics.json",
                run_dir / "history.csv",
                run_dir / "best_checkpoint.pth",
                run_dir / "train_manifest.csv",
                run_dir / "predictions_validation.csv",
                run_dir / "predictions_test.csv",
            )
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Incomplete banana task {run_dir}; missing={missing}")

            completed = _load_json(run_dir / "completed.json")
            metrics = _load_json(run_dir / "metrics.json")
            fingerprint = task_protocol_fingerprint(
                config, "comparison", model_name, spec, EXPECTED_RATIO, seed
            )
            if completed.get("complete") is not True:
                raise ValueError(f"Invalid completion marker: {run_dir / 'completed.json'}")
            if completed.get("protocol_fingerprint") != fingerprint:
                raise ValueError(f"Completion fingerprint mismatch: {run_dir}")
            if metrics.get("protocol_fingerprint") != fingerprint:
                raise ValueError(f"Metrics fingerprint mismatch: {run_dir}")
            if metrics.get("model_spec") != spec:
                raise ValueError(f"Model specification mismatch: {run_dir}")
            if metrics.get("from_scratch") is not True or metrics.get("no_pretrained_weights") is not True:
                raise ValueError(f"Task is not certified as from-scratch: {run_dir}")
            if int(metrics.get("image_size", 0)) != 224:
                raise ValueError(f"Task did not use 224px inputs: {run_dir}")
            if set(metrics.get("splits", {})) != EXPECTED_SPLITS:
                raise ValueError(f"Expected validation/test metrics only: {run_dir}")
            if int(metrics["splits"]["test"]["n"]) != 267:
                raise ValueError(f"Unexpected banana test count: {run_dir}")

            expected_metric_paths.add((run_dir / "metrics.json").resolve())
            manifest_hashes[seed].add(_sha256(run_dir / "train_manifest.csv"))
            train_counts[seed].add(int(metrics["train_samples"]))

    actual_metric_paths = {path.resolve() for path in results_dir.rglob("metrics.json")}
    if actual_metric_paths != expected_metric_paths:
        missing = sorted(str(path) for path in expected_metric_paths - actual_metric_paths)
        extra = sorted(str(path) for path in actual_metric_paths - expected_metric_paths)
        raise ValueError(f"Expected exactly 35 result files; missing={missing}, extra={extra}")
    for seed in EXPECTED_SEEDS:
        if len(manifest_hashes[seed]) != 1:
            raise ValueError(f"Models did not share the same train manifest for seed {seed}")
        if train_counts[seed] != {4698}:
            raise ValueError(
                f"Expected 4,698 balanced training images at ratio=1.0; "
                f"seed {seed} contains {sorted(train_counts[seed])}"
            )


def _group_rows(rows: Iterable[Mapping[str, str]], split: str) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {name: [] for name in EXPECTED_MODELS}
    for raw in rows:
        row = dict(raw)
        if row.get("suite") != "comparison" or row.get("split") != split:
            continue
        if row.get("model") not in grouped:
            raise ValueError(f"Unexpected model in aggregate: {row.get('model')}")
        grouped[row["model"]].append(row)
    for model_name, model_rows in grouped.items():
        seeds = {int(row["seed"]) for row in model_rows}
        if seeds != set(EXPECTED_SEEDS) or len(model_rows) != len(EXPECTED_SEEDS):
            raise ValueError(f"{model_name}/{split} does not contain exactly five seeds")
    return grouped


def _summary_table(grouped: Mapping[str, Sequence[Mapping[str, str]]], split: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_name in EXPECTED_MODELS:
        members = grouped[model_name]
        result: Dict[str, Any] = {
            "dataset": "banana",
            "split": split,
            "model": model_name,
            "n_seeds": len(members),
            "parameters": int(members[0]["parameters"]),
            "parameters_million": int(members[0]["parameters"]) / 1_000_000.0,
            "model_size_mb": float(members[0]["model_size_mb"]),
        }
        for metric in MAIN_METRICS:
            values = [float(member[metric]) for member in members]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = stdev(values)
            result[f"{metric}_ci95"] = confidence_interval_95_half_width(values)
            result[f"{metric}_mean_sd"] = _format_mean_sd(mean(values), stdev(values))
        rows.append(result)
    rows.sort(key=lambda row: (-float(row["macro_f1_mean"]), int(row["parameters"])))
    return rows


def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _paired_effects(grouped: Mapping[str, Sequence[Mapping[str, str]]]) -> List[Dict[str, Any]]:
    proposed_name = "MobileNetV2KANPlus"
    proposed = {int(row["seed"]): float(row["macro_f1"]) for row in grouped[proposed_name]}
    rows: List[Dict[str, Any]] = []
    raw_p_values: List[float] = []
    for baseline_name in EXPECTED_MODELS:
        if baseline_name == proposed_name:
            continue
        baseline = {int(row["seed"]): float(row["macro_f1"]) for row in grouped[baseline_name]}
        differences = [proposed[seed] - baseline[seed] for seed in EXPECTED_SEEDS]
        difference_mean = mean(differences)
        difference_std = stdev(differences)
        half_width = confidence_interval_95_half_width(differences)
        if math.isclose(difference_std, 0.0, abs_tol=1e-15):
            statistic = 0.0 if math.isclose(difference_mean, 0.0, abs_tol=1e-15) else math.copysign(math.inf, difference_mean)
            p_value = 1.0 if statistic == 0.0 else 0.0
            cohens_dz = 0.0 if statistic == 0.0 else math.copysign(math.inf, difference_mean)
        else:
            test = stats.ttest_rel(
                [proposed[seed] for seed in EXPECTED_SEEDS],
                [baseline[seed] for seed in EXPECTED_SEEDS],
            )
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
            cohens_dz = difference_mean / difference_std
        raw_p_values.append(p_value)
        rows.append(
            {
                "comparison": f"{proposed_name} - {baseline_name}",
                "left_model": proposed_name,
                "right_model": baseline_name,
                "n_pairs": len(EXPECTED_SEEDS),
                "macro_f1_difference_mean": difference_mean,
                "macro_f1_difference_std": difference_std,
                "ci95_low": difference_mean - half_width,
                "ci95_high": difference_mean + half_width,
                "t_statistic": statistic,
                "p_value": p_value,
                "cohens_dz": cohens_dz,
            }
        )
    for row, adjusted in zip(rows, _holm_adjust(raw_p_values)):
        row["p_holm"] = adjusted
        row["significant_holm_0p05"] = bool(adjusted < 0.05)
    return rows


def _training_stability(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    validation = _group_rows(rows, "validation")
    result: List[Dict[str, Any]] = []
    for model_name in EXPECTED_MODELS:
        members = validation[model_name]
        epochs = [int(row["best_epoch"]) for row in members]
        elapsed = [float(row["elapsed_seconds"]) for row in members]
        result.append(
            {
                "model": model_name,
                "best_epoch_mean": mean(epochs),
                "best_epoch_std": stdev(epochs),
                "elapsed_minutes_mean": mean(elapsed) / 60.0,
                "elapsed_minutes_std": stdev(elapsed) / 60.0,
            }
        )
    return result


def _representative_seed(grouped_validation: Mapping[str, Sequence[Mapping[str, str]]]) -> int:
    rows = grouped_validation["MobileNetV2KANPlus"]
    values = {int(row["seed"]): float(row["macro_f1"]) for row in rows}
    target = mean(values.values())
    return min(values, key=lambda seed: (abs(values[seed] - target), seed))


def _paper_guide(
    test_table: Sequence[Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
    representative_seed: int,
) -> str:
    by_model = {str(row["model"]): row for row in test_table}
    plus = by_model["MobileNetV2KANPlus"]
    head = by_model["MobileNetV2KANHeadOnly"]
    mobile = by_model["MobileNetV2"]
    winner = test_table[0]
    effects_by_right = {str(row["right_model"]): row for row in effects}
    plus_mobile = effects_by_right["MobileNetV2"]
    plus_head = effects_by_right["MobileNetV2KANHeadOnly"]

    def effect_sentence(label: str, item: Mapping[str, Any]) -> str:
        significance = (
            "Holm校正后达到0.05显著性水平"
            if bool(item["significant_holm_0p05"])
            else "Holm校正后未达到0.05显著性水平"
        )
        return (
            f"{label}的同种子配对Macro-F1差为{float(item['macro_f1_difference_mean']):+.4f}，"
            f"95% CI [{float(item['ci95_low']):+.4f}, {float(item['ci95_high']):+.4f}]，"
            f"Holm校正p={float(item['p_holm']):.4f}，{significance}。"
        )

    return f"""# 香蕉数据集论文结果使用说明

## 实验协议

- 香蕉原始划分：train=5,871（o=2,349，r=3,522）、validation=568（o=229，r=339）、test=267（o=113，r=154）。
- `ratio=1.0`遵循荔枝实验相同的类别平衡嵌套抽样规则，实际每次训练使用4,698张图像（o/r各2,349张），不是5,871张全部训练图像。
- 7个模型均使用224×224输入，从随机初始化开始训练；任何模型都未调用预训练权重。
- 每个模型运行5个随机种子（42–46），检查点只依据validation Macro-F1选择。
- 香蕉没有独立跨源`test1`，因此这里只能写“第二数据集复现”或“跨数据集可迁移性证据”，不能把banana/test称为跨源测试。
- KAN参数由荔枝验证集冻结为layers=3、grid=3、spline order=2、hidden=128、dropout=0.5；香蕉结果未参与调参。

## 正文使用哪些文件

1. `banana_test_main_table.csv`：论文香蕉主结果表，正文优先使用Macro-F1、Balanced Accuracy、MCC，并保留Accuracy、AUC和ECE。
2. `banana_paired_effects.csv`：MobileNetV2KANPlus与其余模型的同种子配对差、Student-t 95% CI和Holm校正p值；用于判断能否写“显著优于”。
3. `banana_validation_table.csv`：说明模型选择与早停表现，可放补充材料，不用它代替test主结果。
4. `banana_test_per_seed.csv`：五种子原始点和复核依据，建议作为补充材料。
5. `banana_training_stability.csv`：最佳epoch与运行时间，适合复现性/效率说明。
6. `figures/01_banana_test_macro_f1`：正文总体性能图；均值、Student-t 95% CI和五个种子原始点同时呈现。
7. `figures/03_banana_paired_effects`：正文或补充材料的配对效应图。
8. `figures/04_banana_kanplus_diagnostics`：validation选择的代表种子{representative_seed}之混淆矩阵与校准曲线，只用于诊断。

## 当前真实结果摘要（脚本自动生成）

- 香蕉test Macro-F1均值最高的模型为{winner['model']}：{winner['macro_f1_mean_sd']}。
- MobileNetV2KANPlus：Macro-F1={plus['macro_f1_mean_sd']}，Balanced Accuracy={plus['balanced_accuracy_mean_sd']}，MCC={plus['mcc_mean_sd']}。
- MobileNetV2KANHeadOnly：Macro-F1={head['macro_f1_mean_sd']}。
- 原生MobileNetV2：Macro-F1={mobile['macro_f1_mean_sd']}。
- {effect_sentence('MobileNetV2KANPlus相对MobileNetV2', plus_mobile)}
- {effect_sentence('MobileNetV2KANPlus相对HeadOnly', plus_head)}

## 写作约束

- 主表写“均值±标准差”，图中误差条使用Student-t 95% CI，二者不要混淆。
- 只有`banana_paired_effects.csv`中`significant_holm_0p05=True`时，才写“显著优于”；否则写“均值更高/更低，但未检出显著差异”。
- 香蕉是冻结配置的独立数据集复现；不要根据banana/test重新修改KAN结构或训练超参数。
- 不要把代表种子的混淆矩阵当作五种子总体结果，图注明确写`representative seed={representative_seed}`。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/banana_224.json")
    parser.add_argument("--results-dir", default="outputs/banana_224")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (project_root / results_dir).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "paper_results"
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()

    config = _load_protocol(config_path, results_dir, project_root)
    _validate_matrix(config, results_dir)
    aggregate_results(results_dir)
    all_runs_path = results_dir / "all_runs.csv"
    rows = _read_csv(all_runs_path)
    validation_grouped = _group_rows(rows, "validation")
    test_grouped = _group_rows(rows, "test")
    validation_table = _summary_table(validation_grouped, "validation")
    test_table = _summary_table(test_grouped, "test")
    effects = _paired_effects(test_grouped)
    stability = _training_stability(rows)
    representative_seed = _representative_seed(validation_grouped)

    table_fields = list(test_table[0].keys())
    effect_fields = list(effects[0].keys())
    stability_fields = list(stability[0].keys())
    per_seed_fields = list(rows[0].keys())
    test_per_seed = [row for row in rows if row["split"] == "test"]

    _write_csv(output_dir / "banana_test_main_table.csv", test_table, table_fields)
    _write_csv(output_dir / "banana_validation_table.csv", validation_table, table_fields)
    _write_csv(output_dir / "banana_test_per_seed.csv", test_per_seed, per_seed_fields)
    _write_csv(output_dir / "banana_paired_effects.csv", effects, effect_fields)
    _write_csv(output_dir / "banana_training_stability.csv", stability, stability_fields)
    _write_json(
        output_dir / "representative_seed.json",
        {
            "model": "MobileNetV2KANPlus",
            "seed": representative_seed,
            "selection_split": "validation",
            "selection_rule": "closest validation macro-F1 to the five-seed mean",
            "test_used_for_selection": False,
        },
    )
    _write_text(
        output_dir / "PAPER_RESULTS_GUIDE.md",
        _paper_guide(test_table, effects, representative_seed),
    )
    print("Banana result matrix verified: 7 models x 5 seeds = 35 from-scratch runs")
    print(f"Publication tables written to: {output_dir}")
    print(f"Representative seed selected from validation only: {representative_seed}")


if __name__ == "__main__":
    main()
