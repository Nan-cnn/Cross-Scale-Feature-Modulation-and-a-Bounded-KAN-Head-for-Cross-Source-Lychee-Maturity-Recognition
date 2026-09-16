"""Create publication figures for the completed banana replication matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "MobileNetV2KANPlus": "#C43C39",
    "MobileNetV2KANHeadOnly": "#E6843D",
    "MobileNetV2": "#2C6EBA",
    "ShuffleNetV2": "#5A9F68",
    "ResNet50": "#8B6BB1",
    "VGG16": "#7A7A7A",
    "EfficientNetB3": "#31A6A0",
}

DISPLAY_NAMES = {
    "MobileNetV2KANPlus": "MobileNetV2-KAN+",
    "MobileNetV2KANHeadOnly": "MobileNetV2-KAN Head-Only",
    "MobileNetV2": "MobileNetV2",
    "ShuffleNetV2": "ShuffleNetV2",
    "ResNet50": "ResNet50",
    "VGG16": "VGG16",
    "EfficientNetB3": "EfficientNet-B3",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 320,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_test_macro_f1(main: pd.DataFrame, per_seed: pd.DataFrame, output_dir: Path) -> None:
    ordered = main.sort_values("macro_f1_mean", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for index, row in ordered.iterrows():
        model = str(row["model"])
        color = COLORS.get(model, "#555555")
        values = per_seed.loc[per_seed["model"] == model, "macro_f1"].astype(float).to_numpy()
        offsets = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(values, np.full_like(values, index, dtype=float) + offsets, s=27, color=color, alpha=0.55, zorder=2)
        ax.errorbar(
            float(row["macro_f1_mean"]),
            index,
            xerr=float(row["macro_f1_ci95"]),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=7,
            capsize=4,
            linewidth=1.8,
            zorder=3,
        )
    ax.set_yticks(y, [DISPLAY_NAMES.get(str(model), str(model)) for model in ordered["model"]])
    ax.set_xlabel("Test Macro-F1")
    ax.set_title("Banana replication: five-seed test performance")
    ax.text(
        0.01,
        -0.14,
        "Large points: mean; error bars: Student-t 95% CI; small points: seeds 42–46",
        transform=ax.transAxes,
        fontsize=8.5,
    )
    save_figure(fig, output_dir, "01_banana_test_macro_f1")


def plot_efficiency(main: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    label_offsets = {
        "MobileNetV2KANHeadOnly": (6, -18),
        "EfficientNetB3": (7, 7),
        "MobileNetV2": (7, 7),
        "MobileNetV2KANPlus": (7, 7),
        "ShuffleNetV2": (7, 7),
        "ResNet50": (7, 7),
        "VGG16": (7, 7),
    }
    for _, row in main.iterrows():
        model = str(row["model"])
        x = float(row["parameters_million"])
        y = float(row["macro_f1_mean"])
        ax.errorbar(
            x,
            y,
            yerr=float(row["macro_f1_ci95"]),
            fmt="o",
            markersize=8,
            capsize=3,
            color=COLORS.get(model, "#555555"),
        )
        ax.annotate(
            DISPLAY_NAMES.get(model, model),
            (x, y),
            xytext=label_offsets.get(model, (7, 7)),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (million, log scale)")
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Banana performance–complexity trade-off")
    save_figure(fig, output_dir, "02_banana_efficiency_tradeoff")


def plot_paired_effects(effects: pd.DataFrame, output_dir: Path) -> None:
    ordered = effects.sort_values("macro_f1_difference_mean").reset_index(drop=True)
    labels = [DISPLAY_NAMES.get(str(model), str(model)) for model in ordered["right_model"]]
    values = ordered["macro_f1_difference_mean"].astype(float).to_numpy()
    lower = values - ordered["ci95_low"].astype(float).to_numpy()
    upper = ordered["ci95_high"].astype(float).to_numpy() - values
    significant = ordered["significant_holm_0p05"].astype(str).str.lower().eq("true").to_numpy()
    colors = np.where(significant, "#C43C39", "#667085")

    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    y = np.arange(len(ordered))
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    for index in range(len(ordered)):
        ax.errorbar(
            values[index],
            y[index],
            xerr=np.array([[lower[index]], [upper[index]]]),
            fmt="o",
            color=colors[index],
            capsize=4,
            linewidth=1.7,
        )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Paired test Macro-F1 difference (KANPlus − baseline)")
    ax.set_title("Banana paired effects across identical seeds")
    ax.text(
        0.01,
        -0.15,
        "Intervals: two-sided Student-t 95% CI; red indicates Holm-adjusted p < 0.05",
        transform=ax.transAxes,
        fontsize=8.5,
    )
    save_figure(fig, output_dir, "03_banana_paired_effects")


def plot_diagnostics(results_dir: Path, paper_dir: Path, output_dir: Path) -> None:
    selection = json.loads((paper_dir / "representative_seed.json").read_text(encoding="utf-8"))
    seed = int(selection["seed"])
    prediction_path = (
        results_dir
        / "comparison"
        / "mobilenetv2kanplus"
        / "ratio_1.00"
        / f"seed_{seed}"
        / "predictions_test.csv"
    )
    frame = pd.read_csv(prediction_path)
    labels = frame["label"].astype(int).to_numpy()
    predictions = frame["prediction"].astype(int).to_numpy()
    probabilities = frame["probability_positive"].astype(float).to_numpy()
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    correctness = (labels == predictions).astype(float)

    confusion = np.zeros((2, 2), dtype=int)
    for label, prediction in zip(labels, predictions):
        confusion[label, prediction] += 1
    normalized = confusion / np.maximum(1, confusion.sum(axis=1, keepdims=True))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    image = axes[0].imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    for row in range(2):
        for column in range(2):
            axes[0].text(
                column,
                row,
                f"{confusion[row, column]}\n({normalized[row, column]:.1%})",
                ha="center",
                va="center",
                color="white" if normalized[row, column] > 0.55 else "black",
            )
    axes[0].set_xticks([0, 1], ["Overripe", "Ripe"])
    axes[0].set_yticks([0, 1], ["Overripe", "Ripe"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Ground truth")
    axes[0].set_title("Row-normalized confusion")

    edges = np.linspace(0.0, 1.0, 11)
    bin_confidence, bin_accuracy, bin_size = [], [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            bin_confidence.append(float(confidence[mask].mean()))
            bin_accuracy.append(float(correctness[mask].mean()))
            bin_size.append(int(mask.sum()))
    axes[1].plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    axes[1].plot(bin_confidence, bin_accuracy, marker="o", color=COLORS["MobileNetV2KANPlus"])
    for x, y, count in zip(bin_confidence, bin_accuracy, bin_size):
        axes[1].annotate(str(count), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
    axes[1].set_xlim(0.45, 1.01)
    axes[1].set_ylim(0.45, 1.01)
    axes[1].set_xlabel("Mean confidence")
    axes[1].set_ylabel("Empirical accuracy")
    axes[1].set_title("Reliability diagram (labels show bin n)")
    fig.suptitle(f"MobileNetV2-KAN+ banana diagnostics, validation-selected seed {seed}")
    save_figure(fig, output_dir, "04_banana_kanplus_diagnostics")


def write_figure_index(output_dir: Path) -> None:
    content = """# Banana figure index

1. `01_banana_test_macro_f1`: main result figure; five raw seed points, mean and Student-t 95% CI.
2. `02_banana_efficiency_tradeoff`: parameter count versus test Macro-F1; use in the efficiency discussion.
3. `03_banana_paired_effects`: paired KANPlus-minus-baseline effects; use to support or reject significance claims.
4. `04_banana_kanplus_diagnostics`: confusion and calibration for a validation-selected representative seed; diagnostic only.

PNG files are intended for Word drafting and PDF files for publication. Banana/test is a same-dataset held-out split, not a cross-source split.
"""
    (output_dir / "FIGURE_INDEX.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="outputs/banana_224")
    parser.add_argument("--paper-dir", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (project_root / results_dir).resolve()
    paper_dir = Path(args.paper_dir) if args.paper_dir else results_dir / "paper_results"
    if not paper_dir.is_absolute():
        paper_dir = (project_root / paper_dir).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else paper_dir / "figures"
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()

    required = (
        paper_dir / "banana_test_main_table.csv",
        paper_dir / "banana_test_per_seed.csv",
        paper_dir / "banana_paired_effects.csv",
        paper_dir / "representative_seed.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing analyzed banana results: {missing}. Run analyze_banana_results.py first."
        )

    setup_style()
    main_table = pd.read_csv(required[0])
    per_seed = pd.read_csv(required[1])
    effects = pd.read_csv(required[2])
    plot_test_macro_f1(main_table, per_seed, output_dir)
    plot_efficiency(main_table, output_dir)
    plot_paired_effects(effects, output_dir)
    plot_diagnostics(results_dir, paper_dir, output_dir)
    write_figure_index(output_dir)
    print(f"Banana publication figures written to: {output_dir}")


if __name__ == "__main__":
    main()
