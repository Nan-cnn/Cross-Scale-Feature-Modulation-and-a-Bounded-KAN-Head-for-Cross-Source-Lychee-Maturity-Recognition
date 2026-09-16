"""Generate publication-oriented figures from completed experiment matrices."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from experiment_engine import aggregate_results, confidence_interval_95_half_width


PALETTE = ["#22577A", "#38A3A5", "#57CC99", "#F4A261", "#E76F51", "#7B2CBF", "#577590"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )


def save_figure(fig, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def selected_ratio(frame: pd.DataFrame) -> float:
    return float(frame["ratio"].astype(float).max())


def plot_generalization_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    comparison = summary[summary["suite"] == "comparison"].copy()
    test = comparison[comparison["split"] == "test"].set_index(["model", "ratio"])
    cross = comparison[comparison["split"] == "test1"].set_index(["model", "ratio"])
    common = test.index.intersection(cross.index)
    if common.empty:
        return
    gap = (test.loc[common, "macro_f1_mean"] - cross.loc[common, "macro_f1_mean"]).rename("gap")
    matrix = gap.reset_index().pivot(index="model", columns="ratio", values="gap")

    fig, ax = plt.subplots(figsize=(8.2, max(3.8, 0.55 * len(matrix))))
    image = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto", vmin=0.0)
    ax.set_xticks(range(len(matrix.columns)), [f"{float(value):.1f}" for value in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Training subset ratio")
    ax.set_title("Same-source to cross-source macro-F1 degradation")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            if pd.notna(value):
                ax.text(column, row, f"{value * 100:.1f} pp", ha="center", va="center", fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Macro-F1 gap")
    save_figure(fig, output_dir, "01_generalization_gap_heatmap")


def plot_dumbbell(summary: pd.DataFrame, output_dir: Path) -> None:
    comparison = summary[summary["suite"] == "comparison"].copy()
    ratio = selected_ratio(comparison)
    subset = comparison[np.isclose(comparison["ratio"].astype(float), ratio)]
    test = subset[subset["split"] == "test"].set_index("model")
    cross = subset[subset["split"] == "test1"].set_index("model")
    models = sorted(set(test.index).intersection(cross.index), key=lambda name: cross.loc[name, "macro_f1_mean"])
    if not models:
        return
    y = np.arange(len(models))
    same_values = np.array([test.loc[name, "macro_f1_mean"] for name in models])
    cross_values = np.array([cross.loc[name, "macro_f1_mean"] for name in models])
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.6 * len(models))))
    for index in range(len(models)):
        ax.plot([cross_values[index], same_values[index]], [y[index], y[index]], color="#B8C0C8", lw=3)
    ax.scatter(cross_values, y, color="#E76F51", s=65, label="Cross-source test1", zorder=3)
    ax.scatter(same_values, y, color="#22577A", s=65, label="Same-source test", zorder=3)
    ax.set_yticks(y, models)
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Macro-F1")
    ax.set_title(f"Generalization shift at training ratio {ratio:.1f}")
    ax.legend(loc="lower right")
    save_figure(fig, output_dir, "02_same_vs_cross_dumbbell")


def plot_efficiency_bubbles(all_runs: pd.DataFrame, output_dir: Path) -> None:
    frame = all_runs[(all_runs["suite"] == "comparison") & (all_runs["split"] == "test1")].copy()
    if frame.empty:
        return
    ratio = selected_ratio(frame)
    frame = frame[np.isclose(frame["ratio"].astype(float), ratio)]
    grouped = frame.groupby("model", as_index=False).agg(
        macro_f1=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        parameters=("parameters", "first"),
        model_size_mb=("model_size_mb", "first"),
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for index, row in grouped.iterrows():
        color = PALETTE[index % len(PALETTE)]
        size = 80 + 8 * math.sqrt(max(1.0, float(row["model_size_mb"])))
        ax.scatter(row["parameters"] / 1e6, row["macro_f1"], s=size, color=color, alpha=0.85)
        ax.errorbar(
            row["parameters"] / 1e6,
            row["macro_f1"],
            yerr=0.0 if pd.isna(row["macro_f1_std"]) else row["macro_f1_std"],
            color=color,
            capsize=3,
            lw=1,
        )
        ax.annotate(row["model"], (row["parameters"] / 1e6, row["macro_f1"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (million, log scale)")
    ax.set_ylabel("Cross-source macro-F1")
    ax.set_title(f"Accuracy-efficiency trade-off at ratio {ratio:.1f}")
    save_figure(fig, output_dir, "03_performance_efficiency_bubble")


def plot_seed_violin(all_runs: pd.DataFrame, output_dir: Path) -> None:
    frame = all_runs[(all_runs["suite"] == "comparison") & (all_runs["split"] == "test1")].copy()
    if frame.empty:
        return
    ratio = selected_ratio(frame)
    frame = frame[np.isclose(frame["ratio"].astype(float), ratio)]
    models = sorted(frame["model"].unique())
    values = [frame.loc[frame["model"] == model, "macro_f1"].astype(float).values for model in models]
    fig, ax = plt.subplots(figsize=(max(8.0, len(models) * 1.15), 5.5))
    parts = ax.violinplot(values, showmeans=True, showmedians=True, widths=0.8)
    for body, color in zip(parts["bodies"], PALETTE * 3):
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_alpha(0.72)
    for index, series in enumerate(values, start=1):
        jitter = np.linspace(-0.08, 0.08, len(series)) if len(series) > 1 else [0.0]
        ax.scatter(index + jitter, series, color="#263238", s=18, alpha=0.8, zorder=3)
    ax.set_xticks(range(1, len(models) + 1), models, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Cross-source macro-F1")
    ax.set_title(f"Random-seed distribution at ratio {ratio:.1f}")
    save_figure(fig, output_dir, "04_seed_distribution_violin")


def plot_ablation_effects(all_runs: pd.DataFrame, output_dir: Path) -> None:
    frame = all_runs[(all_runs["suite"] == "ablation") & (all_runs["split"] == "validation")].copy()
    if frame.empty or "reference" not in set(frame["model"]):
        return
    reference = frame[frame["model"] == "reference"].set_index("seed")["macro_f1"].astype(float)
    effects = []
    for model_name, group in frame[frame["model"] != "reference"].groupby("model"):
        values = group.set_index("seed")["macro_f1"].astype(float)
        common = reference.index.intersection(values.index)
        if common.empty:
            continue
        difference = values.loc[common] - reference.loc[common]
        effects.append(
            (
                model_name,
                float(difference.mean()),
                confidence_interval_95_half_width(difference.astype(float).tolist()),
            )
        )
    if not effects:
        return
    effects.sort(key=lambda item: item[1])
    names = [item[0] for item in effects]
    means = np.array([item[1] for item in effects])
    cis = np.array([item[2] for item in effects])
    colors = ["#E76F51" if value < 0 else "#38A3A5" for value in means]
    fig, ax = plt.subplots(figsize=(9.0, max(5.0, 0.42 * len(names))))
    y = np.arange(len(names))
    ax.axvline(0.0, color="#263238", lw=1)
    ax.errorbar(means, y, xerr=cis, fmt="none", ecolor="#6C757D", capsize=3, lw=1.2)
    ax.scatter(means, y, c=colors, s=55, zorder=3)
    ax.set_yticks(y, names)
    ax.set_xlabel("Paired validation macro-F1 change vs reference")
    ax.set_title("One-factor KAN ablation effect with Student-t 95% CI")
    save_figure(fig, output_dir, "05_ablation_effect_forest")


def _proposed_selection(all_runs: pd.DataFrame):
    frame = all_runs[(all_runs["suite"] == "comparison") & (all_runs["model"] == "MobileNetV2KANPlus")]
    if frame.empty:
        return None
    ratio = selected_ratio(frame)
    validation = frame[
        (frame["split"] == "validation") & np.isclose(frame["ratio"].astype(float), ratio)
    ].copy()
    if validation.empty:
        return None
    median = float(validation["macro_f1"].median())
    row = validation.iloc[(validation["macro_f1"] - median).abs().argsort().iloc[0]]
    return ratio, int(row["seed"])


def _prediction_path(run_dir: Path, model: str, ratio: float, seed: int, split: str) -> Path:
    model_name = model.lower().replace(".", "p")
    model_name = "".join(character if character.isalnum() or character in "_+-" else "_" for character in model_name)
    return run_dir / "comparison" / model_name / f"ratio_{ratio:.2f}" / f"seed_{seed}" / f"predictions_{split}.csv"


def plot_calibration_and_confusion(all_runs: pd.DataFrame, run_dir: Path, output_dir: Path) -> None:
    selection = _proposed_selection(all_runs)
    if selection is None:
        return
    ratio, seed = selection
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    for panel, split in enumerate(("test", "test1")):
        prediction_path = _prediction_path(run_dir, "MobileNetV2KANPlus", ratio, seed, split)
        if not prediction_path.exists():
            axes[panel].set_visible(False)
            continue
        frame = pd.read_csv(prediction_path)
        matrix = np.zeros((2, 2), dtype=float)
        for label, prediction in zip(frame["label"].astype(int), frame["prediction"].astype(int)):
            matrix[label, prediction] += 1
        matrix = matrix / np.maximum(1.0, matrix.sum(axis=1, keepdims=True))
        image = axes[panel].imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        for row in range(2):
            for column in range(2):
                axes[panel].text(column, row, f"{matrix[row, column] * 100:.1f}%", ha="center", va="center")
        axes[panel].set_xticks([0, 1], ["Unripe", "Ripe"])
        axes[panel].set_yticks([0, 1], ["Unripe", "Ripe"])
        axes[panel].set_xlabel("Predicted")
        axes[panel].set_ylabel("True")
        axes[panel].set_title(f"Normalized confusion: {split}")

    prediction_path = _prediction_path(run_dir, "MobileNetV2KANPlus", ratio, seed, "test1")
    if prediction_path.exists():
        frame = pd.read_csv(prediction_path)
        probability = frame["probability_positive"].astype(float).values
        label = frame["label"].astype(int).values
        confidence = np.maximum(probability, 1.0 - probability)
        correctness = (frame["prediction"].astype(int).values == label).astype(float)
        centers, observed, counts = [], [], []
        edges = np.linspace(0.5, 1.0, 11)
        for lower, upper in zip(edges[:-1], edges[1:]):
            mask = (confidence > lower) & (confidence <= upper)
            if mask.any():
                centers.append(float(confidence[mask].mean()))
                observed.append(float(correctness[mask].mean()))
                counts.append(int(mask.sum()))
        axes[2].plot([0.5, 1.0], [0.5, 1.0], "--", color="#6C757D", label="Ideal")
        axes[2].plot(centers, observed, marker="o", color="#E76F51", label="Observed")
        axes[2].set_xlim(0.5, 1.0)
        axes[2].set_ylim(0.5, 1.02)
        axes[2].set_xlabel("Mean confidence")
        axes[2].set_ylabel("Empirical accuracy")
        axes[2].set_title("Cross-source reliability")
        axes[2].legend()
    else:
        axes[2].set_visible(False)
    fig.suptitle(f"MobileNetV2KANPlus, ratio {ratio:.1f}, representative seed {seed}", y=1.03)
    save_figure(fig, output_dir, "06_confusion_and_calibration")


def plot_error_atlas(all_runs: pd.DataFrame, run_dir: Path, output_dir: Path) -> None:
    selection = _proposed_selection(all_runs)
    if selection is None:
        return
    ratio, seed = selection
    prediction_path = _prediction_path(run_dir, "MobileNetV2KANPlus", ratio, seed, "test1")
    if not prediction_path.exists():
        return
    frame = pd.read_csv(prediction_path)
    errors = frame[frame["correct"] == 0].sort_values("confidence", ascending=False).head(8)
    uncertain = frame[frame["correct"] == 1].sort_values("confidence", ascending=True).head(4)
    selected = pd.concat([errors, uncertain], ignore_index=True).head(12)
    if selected.empty:
        return
    class_names = {0: "Unripe", 1: "Ripe"}
    fig, axes = plt.subplots(3, 4, figsize=(13.6, 11.0))
    for axis, (_, row) in zip(axes.flat, selected.iterrows()):
        path = Path(str(row["path"]))
        if not path.exists() and "data" in path.parts:
            data_index = path.parts.index("data")
            path = run_dir.parents[1] / Path(*path.parts[data_index:])
        if not path.exists():
            axis.set_visible(False)
            continue
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        status = "High-confidence error" if int(row["correct"]) == 0 else "Low-confidence correct"
        true_name = class_names[int(row["label"])]
        predicted_name = class_names[int(row["prediction"])]
        axis.set_title(
            f"{status}\nTrue {true_name} | Pred {predicted_name} | Conf. {float(row['confidence']):.2f}",
            fontsize=10,
        )
        axis.axis("off")
    for axis in axes.flat[len(selected) :]:
        axis.set_visible(False)
    fig.suptitle(
        f"Cross-source failure atlas (MobileNetV2-KAN+, ratio {ratio:.1f} balanced training subset, seed {seed})",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    save_figure(fig, output_dir, "12_cross_source_failure_atlas_100pct")


def write_figure_index(output_dir: Path) -> None:
    content = """# Figure index for the manuscript

1. `01_generalization_gap_heatmap`: main-text figure; shows how same-source saturation hides cross-source degradation.
2. `02_same_vs_cross_dumbbell`: main-text figure; direct model-by-model domain-shift comparison.
3. `03_performance_efficiency_bubble`: main-text or discussion figure; reports the accuracy/parameter trade-off.
4. `04_seed_distribution_violin`: supplementary figure; exposes initialization sensitivity rather than only mean values.
5. `05_ablation_effect_forest`: main ablation figure; paired validation effect and Student-t 95% confidence interval versus the reference configuration.
6. `06_confusion_and_calibration`: main-text figure; class-wise behavior and confidence reliability.
7. `07_cross_source_error_atlas`: qualitative analysis; high-confidence failures and uncertain correct cases.

Use PNG for Word drafting and PDF for journal submission. Do not copy values from a smoke run into the manuscript; only use the completed `full_224` matrix.
"""
    (output_dir / "FIGURE_INDEX.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/full_224")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (project_root / run_dir).resolve()
    if not (run_dir / "all_runs.csv").exists():
        aggregate_results(run_dir)
    all_runs_path = run_dir / "all_runs.csv"
    summary_path = run_dir / "summary_metrics.csv"
    if not all_runs_path.exists() or not summary_path.exists():
        raise FileNotFoundError("No completed run summaries found. Run run_experiments.py first.")
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    all_runs = pd.read_csv(all_runs_path)
    summary = pd.read_csv(summary_path)
    plot_generalization_heatmap(summary, output_dir)
    plot_dumbbell(summary, output_dir)
    plot_efficiency_bubbles(all_runs, output_dir)
    plot_seed_violin(all_runs, output_dir)
    plot_ablation_effects(all_runs, output_dir)
    plot_calibration_and_confusion(all_runs, run_dir, output_dir)
    plot_error_atlas(all_runs, run_dir, output_dir)
    write_figure_index(output_dir)
    print(f"Figures written to {output_dir}")


if __name__ == "__main__":
    main()
