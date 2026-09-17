"""
Plots — Publication-ready PDF figures
=====================================
All figures for the quantum corrosion classification paper.
Large serif fonts, vector PDF output, consistent styling.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from src.config import PlotConfig
from src.evaluation import AggregatedResult
import logging

logger = logging.getLogger(__name__)


def _apply_style(cfg: PlotConfig):
    """Set global matplotlib style for publication."""
    plt.rcParams.update({
        "font.family": cfg.font_family,
        "font.size": cfg.font_size,
        "axes.titlesize": cfg.title_size,
        "axes.labelsize": cfg.font_size,
        "xtick.labelsize": cfg.tick_size,
        "ytick.labelsize": cfg.tick_size,
        "legend.fontsize": cfg.legend_size,
        "figure.dpi": cfg.dpi,
        "lines.linewidth": cfg.linewidth,
        "axes.grid": True,
        "grid.alpha": cfg.grid_alpha,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "text.usetex": False,  # set True if LaTeX available
    })


def _save(fig, path: Path, cfg: PlotConfig):
    fig.savefig(path, format=cfg.format, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Figure saved → {path}")


# ── Figure 1: Pipeline diagram ──────────────────────────────────────────────

def plot_pipeline(out_dir: Path, cfg: PlotConfig):
    """System pipeline: IQ → FFT → Spectrogram → PCA → Encoding → VQC → Class."""
    _apply_style(cfg)
    fig, ax = plt.subplots(1, 1, figsize=(14, 3))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2)
    ax.axis("off")

    stages = [
        "IQ\nSamples", "FFT\nWindowed", "Spectrogram\n(128×4096)",
        "Resize\n(224×224)", "PCA\nReduction", "Quantum\nEncoding",
        "VQC\nCircuit", "Classification\nOutput"
    ]
    x_positions = np.linspace(0.5, 13.5, len(stages))
    box_w, box_h = 1.4, 1.2

    for i, (x, label) in enumerate(zip(x_positions, stages)):
        color = cfg.colors[i % len(cfg.colors)]
        bbox = FancyBboxPatch(
            (x - box_w / 2, 1 - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.1", facecolor=color, alpha=0.2,
            edgecolor=color, linewidth=2,
        )
        ax.add_patch(bbox)
        ax.text(x, 1, label, ha="center", va="center",
                fontsize=11, fontweight="bold")

        if i < len(stages) - 1:
            x_next = x_positions[i + 1]
            ax.annotate(
                "", xy=(x_next - box_w / 2 - 0.05, 1),
                xytext=(x + box_w / 2 + 0.05, 1),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2),
            )

    _save(fig, out_dir / "fig_01_pipeline.pdf", cfg)


# ── Figure 2: Example spectrograms ─────────────────────────────────────────

def plot_spectrograms(
    images: Dict[str, np.ndarray],
    out_dir: Path,
    cfg: PlotConfig,
):
    """Show example spectrograms for different oxide quantities."""
    _apply_style(cfg)
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, img) in zip(axes, images.items()):
        ax.imshow(img, cmap="gray", aspect="auto")
        ax.set_title(label, fontsize=cfg.title_size)
        ax.set_xlabel("Frequency Bin")
        ax.set_ylabel("FFT Stack Index")

    fig.suptitle("Spectrogram Examples by Oxide Quantity", fontsize=cfg.title_size + 2)
    plt.tight_layout()
    _save(fig, out_dir / "fig_02_spectrograms.pdf", cfg)


# ── Figure 3: Hybrid architecture diagram ──────────────────────────────────

def plot_architecture(out_dir: Path, cfg: PlotConfig):
    """Block diagram of the hybrid quantum-classical architecture."""
    _apply_style(cfg)
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")

    blocks = [
        ("Input\nSpectrogram", 0.8, "#E3F2FD", "#1565C0"),
        ("Classical\nEncoder", 2.8, "#E8F5E9", "#2E7D32"),
        ("PCA\nReduction", 4.8, "#FFF3E0", "#E65100"),
        ("Quantum\nEncoding", 6.8, "#F3E5F5", "#6A1B9A"),
        ("VQC\nLayers", 8.8, "#FCE4EC", "#AD1457"),
        ("Classical\nHead", 10.8, "#E0F7FA", "#00695C"),
    ]

    for label, x, fc, ec in blocks:
        bbox = FancyBboxPatch(
            (x - 0.7, 0.8), 1.4, 1.4,
            boxstyle="round,pad=0.12", facecolor=fc, edgecolor=ec, linewidth=2.5,
        )
        ax.add_patch(bbox)
        ax.text(x, 1.5, label, ha="center", va="center",
                fontsize=12, fontweight="bold", color=ec)

    for i in range(len(blocks) - 1):
        x1 = blocks[i][1] + 0.7
        x2 = blocks[i + 1][1] - 0.7
        ax.annotate("", xy=(x2, 1.5), xytext=(x1, 1.5),
                     arrowprops=dict(arrowstyle="-|>", color="gray", lw=2.5))

    fig.suptitle("Hybrid Quantum-Classical Architecture", fontsize=cfg.title_size + 2)
    _save(fig, out_dir / "fig_03_architecture.pdf", cfg)


# ── Figure 5: Metrics comparison bar chart ──────────────────────────────────

def plot_metrics_comparison(
    aggregated: List[AggregatedResult],
    out_dir: Path,
    cfg: PlotConfig,
):
    """Grouped bar chart: accuracy, F1, precision, recall for all models."""
    _apply_style(cfg)
    metrics = ["mean_accuracy", "mean_f1", "mean_precision", "mean_recall"]
    labels = ["Accuracy", "F1-Score", "Precision", "Recall"]
    models = [a.model_name for a in aggregated]
    n_models = len(models)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=cfg.figsize_double)
    for i, agg in enumerate(aggregated):
        vals = [getattr(agg, m) for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=agg.model_name,
                      color=cfg.colors[i % len(cfg.colors)], alpha=0.85,
                      edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("Classification Performance Comparison")
    _save(fig, out_dir / "fig_05_metrics_comparison.pdf", cfg)


# ── Figure 6: Parameter count comparison ────────────────────────────────────

def plot_param_comparison(
    aggregated: List[AggregatedResult],
    out_dir: Path,
    cfg: PlotConfig,
):
    """Log-scale bar plot of trainable parameters."""
    _apply_style(cfg)
    fig, ax = plt.subplots(figsize=cfg.figsize_single)
    models = [a.model_name for a in aggregated]
    params = [a.n_params_trainable for a in aggregated]

    bars = ax.barh(models, params, color=cfg.colors[:len(models)],
                   alpha=0.85, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (log scale)")
    ax.set_title("Model Complexity Comparison")

    for bar, p in zip(bars, params):
        ax.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height() / 2,
                f"{p:,}", va="center", fontsize=10)

    plt.tight_layout()
    _save(fig, out_dir / "fig_06_param_comparison.pdf", cfg)


# ── Figure 7: Learning curves (data efficiency) ────────────────────────────

def plot_learning_curves(
    results: Dict[str, List[Tuple[float, float, float]]],
    fractions: List[float],
    out_dir: Path,
    cfg: PlotConfig,
    metric_name: str = "F1-Score",
):
    """
    Learning curves: fraction of training data vs metric.
    results: {model_name: [(mean, std, ...), ...]}
    """
    _apply_style(cfg)
    fig, ax = plt.subplots(figsize=cfg.figsize_single)
    pct = [f * 100 for f in fractions]

    for i, (name, data) in enumerate(results.items()):
        means = [d[0] for d in data]
        stds = [d[1] for d in data]
        color = cfg.colors[i % len(cfg.colors)]
        ax.plot(pct, means, 'o-', color=color, label=name, markersize=6)
        ax.fill_between(pct,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.15, color=color)

    ax.set_xlabel("Training Data (%)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Data Efficiency: {metric_name} vs Training Fraction")
    ax.legend(framealpha=0.9)
    ax.set_xlim(0, 105)
    _save(fig, out_dir / "fig_07_learning_curves.pdf", cfg)


# ── Figure 8: Confusion matrices ───────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: Path,
    cfg: PlotConfig,
):
    """Publication-quality confusion matrix heatmap."""
    _apply_style(cfg)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=cfg.tick_size)

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title, fontsize=cfg.title_size)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=cfg.font_size)

    plt.tight_layout()
    _save(fig, out_path, cfg)


# ── Figure 9: Accuracy vs complexity scatter ────────────────────────────────

def plot_accuracy_vs_complexity(
    aggregated: List[AggregatedResult],
    out_dir: Path,
    cfg: PlotConfig,
):
    """Scatter: trainable params (x, log) vs accuracy (y)."""
    _apply_style(cfg)
    fig, ax = plt.subplots(figsize=cfg.figsize_single)

    for i, agg in enumerate(aggregated):
        color = cfg.colors[i % len(cfg.colors)]
        ax.scatter(agg.n_params_trainable, agg.mean_f1,
                   s=200, color=color, edgecolors="black",
                   linewidths=1.5, zorder=5, label=agg.model_name)
        ax.annotate(agg.model_name,
                    (agg.n_params_trainable, agg.mean_f1),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=10)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (log scale)")
    ax.set_ylabel("Mean F1-Score")
    ax.set_title("Accuracy–Complexity Trade-off")
    ax.legend(framealpha=0.9, loc="lower right")
    _save(fig, out_dir / "fig_09_accuracy_vs_complexity.pdf", cfg)


# ── Figure 10: Ablation study ──────────────────────────────────────────────

def plot_ablation(
    ablation_data: Dict[str, List[float]],
    categories: List[str],
    out_dir: Path,
    cfg: PlotConfig,
    title: str = "Ablation Study",
):
    """Bar chart comparing different encoding / reduction strategies."""
    _apply_style(cfg)
    fig, ax = plt.subplots(figsize=cfg.figsize_single)

    x = np.arange(len(categories))
    width = 0.8 / len(ablation_data)

    for i, (name, values) in enumerate(ablation_data.items()):
        offset = (i - len(ablation_data) / 2 + 0.5) * width
        color = cfg.colors[i % len(cfg.colors)]
        means = [np.mean(v) if isinstance(v, list) else v for v in values]
        stds = [np.std(v) if isinstance(v, list) else 0 for v in values]
        ax.bar(x + offset, means, width, yerr=stds, label=name,
               color=color, alpha=0.85, capsize=3, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel("F1-Score")
    ax.set_title(title)
    ax.legend(framealpha=0.9)
    _save(fig, out_dir / "fig_10_ablation.pdf", cfg)


# ── Figure 11: Noise robustness ────────────────────────────────────────────

def plot_noise_robustness(
    noise_levels: List[float],
    results: Dict[str, List[Tuple[float, float]]],
    out_dir: Path,
    cfg: PlotConfig,
):
    """Accuracy vs noise level for quantum and classical models."""
    _apply_style(cfg)
    fig, ax = plt.subplots(figsize=cfg.figsize_single)

    for i, (name, data) in enumerate(results.items()):
        means = [d[0] for d in data]
        stds = [d[1] for d in data]
        color = cfg.colors[i % len(cfg.colors)]
        ax.plot(noise_levels, means, 'o-', color=color, label=name, markersize=6)
        ax.fill_between(noise_levels,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.15, color=color)

    ax.set_xlabel("Noise Level (σ)")
    ax.set_ylabel("F1-Score")
    ax.set_title("Robustness to Signal Noise")
    ax.legend(framealpha=0.9)
    _save(fig, out_dir / "fig_11_noise_robustness.pdf", cfg)


# ── Figure 12: Runtime comparison ──────────────────────────────────────────

def plot_runtime(
    aggregated: List[AggregatedResult],
    out_dir: Path,
    cfg: PlotConfig,
):
    """Training time and inference time comparison."""
    _apply_style(cfg)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg.figsize_double)

    models = [a.model_name for a in aggregated]
    colors = cfg.colors[:len(models)]

    # Training time
    train_times = [a.mean_train_time for a in aggregated]
    ax1.barh(models, train_times, color=colors, alpha=0.85, edgecolor="white")
    ax1.set_xlabel("Training Time (s)")
    ax1.set_title("Training Time")
    for bar, t in zip(ax1.patches, train_times):
        ax1.text(bar.get_width() + max(train_times) * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f"{t:.1f}s", va="center", fontsize=10)

    # Inference time
    inf_times = [a.mean_inference_ms for a in aggregated]
    ax2.barh(models, inf_times, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_xlabel("Inference Time (ms/sample)")
    ax2.set_title("Inference Latency")
    for bar, t in zip(ax2.patches, inf_times):
        ax2.text(bar.get_width() + max(inf_times) * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f"{t:.2f}ms", va="center", fontsize=10)

    plt.tight_layout()
    _save(fig, out_dir / "fig_12_runtime.pdf", cfg)


# ── F1-score per round (like paper Fig. 5b / Fig. 7) ───────────────────────

def plot_f1_per_round(
    aggregated: List[AggregatedResult],
    out_dir: Path,
    cfg: PlotConfig,
):
    """F1-score per round with mean and ±1σ band."""
    _apply_style(cfg)
    fig, ax = plt.subplots(figsize=cfg.figsize_single)

    for i, agg in enumerate(aggregated):
        rounds = range(1, len(agg.per_round_f1) + 1)
        mean = agg.mean_f1
        std = agg.std_f1
        color = cfg.colors[i % len(cfg.colors)]

        ax.plot(rounds, agg.per_round_f1, 'o-', color=color,
                label=f"{agg.model_name} (avg={mean:.3f})", markersize=5)
        ax.axhline(mean, color=color, ls="--", alpha=0.5)
        ax.fill_between(rounds, mean - std, mean + std, alpha=0.1, color=color)

    ax.set_xlabel("Round")
    ax.set_ylabel("F1-Score")
    ax.set_title("F1-Score Stability Across Training Rounds")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _save(fig, out_dir / "fig_f1_per_round.pdf", cfg)


# ── Generate all figures ────────────────────────────────────────────────────

def generate_all_figures(
    aggregated: List[AggregatedResult],
    out_dir: Path,
    cfg: PlotConfig,
    example_spectrograms: Optional[Dict[str, np.ndarray]] = None,
    learning_curve_data: Optional[Dict] = None,
    learning_curve_fractions: Optional[List[float]] = None,
    ablation_data: Optional[Dict] = None,
    ablation_categories: Optional[List[str]] = None,
    noise_data: Optional[Dict] = None,
    noise_levels: Optional[List[float]] = None,
    confusion_matrices: Optional[Dict[str, Tuple[np.ndarray, List[str]]]] = None,
):
    """Master function: generate all publication figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating publication figures...")

    plot_pipeline(out_dir, cfg)
    plot_architecture(out_dir, cfg)

    if example_spectrograms:
        plot_spectrograms(example_spectrograms, out_dir, cfg)

    if aggregated:
        plot_metrics_comparison(aggregated, out_dir, cfg)
        plot_param_comparison(aggregated, out_dir, cfg)
        plot_accuracy_vs_complexity(aggregated, out_dir, cfg)
        plot_runtime(aggregated, out_dir, cfg)
        plot_f1_per_round(aggregated, out_dir, cfg)

    if learning_curve_data and learning_curve_fractions:
        plot_learning_curves(learning_curve_data, learning_curve_fractions,
                             out_dir, cfg)

    if ablation_data and ablation_categories:
        plot_ablation(ablation_data, ablation_categories, out_dir, cfg)

    if noise_data and noise_levels:
        plot_noise_robustness(noise_levels, noise_data, out_dir, cfg)

    if confusion_matrices:
        for name, (cm, cls_names) in confusion_matrices.items():
            plot_confusion_matrix(
                cm, cls_names, f"Confusion Matrix — {name}",
                out_dir / f"fig_08_cm_{name.lower().replace(' ', '_')}.pdf", cfg,
            )

    logger.info(f"All figures saved → {out_dir}")
