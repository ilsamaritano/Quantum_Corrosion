"""Paper figure generation: all 12 matplotlib figures for the publication."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

_DEFAULT_DPI = 150
_CMAP = "viridis"


def _save_or_show(fig: plt.Figure, output_path: Optional[str | Path]) -> None:
    """Save figure to *output_path* or display interactively."""
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=_DEFAULT_DPI, bbox_inches="tight")
        logger.info("Saved figure to %s", output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Pipeline diagram
# ---------------------------------------------------------------------------

def plot_pipeline_diagram(output_path: Optional[str | Path] = None) -> plt.Figure:
    """Figure 1: End-to-end pipeline diagram using matplotlib patches.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    stages = [
        ("Raw IQ\nData", 0.5),
        ("Validate\n& Clean", 2.5),
        ("STFT\nSpectrogram", 4.5),
        ("Dim.\nReduction", 6.5),
        ("Train\nModel", 8.5),
        ("Evaluate\n& Metrics", 10.5),
        ("Paper\nFigures", 12.5),
    ]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0",
              "#F44336", "#00BCD4", "#795548"]

    for i, ((label, x), color) in enumerate(zip(stages, colors)):
        rect = mpatches.FancyBboxPatch(
            (x - 0.8, 1.2), 1.6, 1.6,
            boxstyle="round,pad=0.1",
            fc=color, ec="white", lw=2, alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(x, 2.0, label, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        if i < len(stages) - 1:
            ax.annotate(
                "", xy=(x + 0.9, 2.0), xytext=(x + 0.8, 2.0),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
            )

    ax.set_title(
        "QuantumCorrosion ML Pipeline", fontsize=14, fontweight="bold", pad=12
    )
    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Signals and spectrograms
# ---------------------------------------------------------------------------

def plot_signals_and_spectrograms(
    signals_dict: Dict[str, np.ndarray],
    output_path: Optional[str | Path] = None,
    n_examples: int = 3,
) -> plt.Figure:
    """Figure 2: Raw IQ signals and their STFT spectrograms per class.

    Args:
        signals_dict: ``{class_name: array(N, L)}``.
        output_path: Optional save path.
        n_examples: Examples per class.

    Returns:
        Matplotlib figure.
    """
    from .spectrograms import stft_spectrogram

    classes = list(signals_dict.keys())
    n_cls = len(classes)
    fig, axes = plt.subplots(
        n_cls * 2, n_examples,
        figsize=(n_examples * 3.5, n_cls * 4),
    )
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]

    for ci, cname in enumerate(classes):
        arr = signals_dict[cname]
        for j in range(min(n_examples, len(arr))):
            sig = arr[j]
            ax_s = axes[2 * ci, j]
            ax_s.plot(sig, lw=0.8, color="steelblue")
            ax_s.set_title(f"{cname} – signal {j+1}", fontsize=7)
            ax_s.set_xlabel("Sample")

            spec = stft_spectrogram(sig)
            ax_sp = axes[2 * ci + 1, j]
            ax_sp.imshow(spec, aspect="auto", origin="lower", cmap=_CMAP)
            ax_sp.set_title(f"{cname} – spec {j+1}", fontsize=7)

    fig.suptitle("Raw IQ Signals and STFT Spectrograms", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Hybrid architecture
# ---------------------------------------------------------------------------

def plot_hybrid_architecture(output_path: Optional[str | Path] = None) -> plt.Figure:
    """Figure 3: Hybrid quantum-classical architecture diagram.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    blocks = [
        ("Input\nFeatures\n(PCA)", 0.7, 2.5, "#607D8B", 1.2, 1.5),
        ("Classical\nPre-processing\nLinear+Tanh", 2.5, 2.5, "#2196F3", 1.6, 1.5),
        ("Quantum\nCircuit\n(VQC)", 5.0, 2.5, "#9C27B0", 2.0, 2.0),
        ("Classical\nOutput\nLinear→Softmax", 7.8, 2.5, "#F44336", 1.6, 1.5),
        ("Class\nPrediction", 10.3, 2.5, "#4CAF50", 1.2, 1.5),
    ]

    for (label, x, y, color, w, h) in blocks:
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.1",
            fc=color, ec="white", lw=2, alpha=0.88,
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

    # Arrows
    arrow_xs = [(1.3, 1.7), (3.3, 4.0), (6.0, 7.0), (8.6, 9.7)]
    for (x1, x2) in arrow_xs:
        ax.annotate(
            "", xy=(x2, 2.5), xytext=(x1, 2.5),
            arrowprops=dict(arrowstyle="->", color="gray", lw=2),
        )

    ax.set_title(
        "Hybrid Quantum-Classical Classifier Architecture",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Quantum circuit
# ---------------------------------------------------------------------------

def plot_quantum_circuit(
    n_qubits: int = 4,
    n_layers: int = 3,
    encoding: str = "angle",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Figure 4: Schematic quantum circuit diagram.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, max(3, n_qubits * 1.0)))
    ax.set_xlim(-0.5, n_layers * 3 + 3.5)
    ax.set_ylim(-0.5, n_qubits + 0.5)
    ax.axis("off")

    wire_y = list(range(n_qubits))

    # Draw wires
    x_end = n_layers * 3 + 2.5
    for q in wire_y:
        ax.plot([-0.3, x_end], [q, q], "k-", lw=1.0)
        ax.text(-0.4, q, f"|0⟩", ha="right", va="center", fontsize=9)

    # Encoding gates
    for q in wire_y:
        rect = mpatches.FancyBboxPatch(
            (0.1, q - 0.3), 0.8, 0.6,
            boxstyle="round,pad=0.05", fc="#2196F3", ec="white",
        )
        ax.add_patch(rect)
        ax.text(0.5, q, f"RY\n(x{q})", ha="center", va="center",
                fontsize=6, color="white")

    # Variational layers
    for layer in range(n_layers):
        x_base = 1.5 + layer * 3
        for q in wire_y:
            for k, (gate, col) in enumerate(
                [("RY\n(θ)", "#9C27B0"), ("RZ\n(φ)", "#FF9800")]
            ):
                rx = x_base + k * 0.9
                r = mpatches.FancyBboxPatch(
                    (rx, q - 0.3), 0.8, 0.6,
                    boxstyle="round,pad=0.05", fc=col, ec="white",
                )
                ax.add_patch(r)
                ax.text(rx + 0.4, q, gate, ha="center", va="center",
                        fontsize=5, color="white")
        # CNOT ring
        x_cnot = x_base + 2.1
        for q in range(n_qubits):
            q_tgt = (q + 1) % n_qubits
            ax.plot([x_cnot, x_cnot], [q, q_tgt], "k-", lw=1.0)
            ax.plot(x_cnot, q, "ko", ms=6)
            ax.add_patch(plt.Circle((x_cnot, q_tgt), 0.18, color="k", fill=False))
            ax.plot(
                [x_cnot - 0.18, x_cnot + 0.18], [q_tgt, q_tgt], "k-", lw=1.0
            )

    # Measurement
    for q in wire_y:
        ax.plot(x_end, q, "D", ms=8, color="#4CAF50")
        ax.text(x_end + 0.15, q, "M", va="center", fontsize=8)

    ax.set_title(
        f"Variational Quantum Circuit — {encoding} encoding, "
        f"{n_qubits} qubits, {n_layers} layers",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Metrics comparison
# ---------------------------------------------------------------------------

def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict],
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Figure 5: Grouped bar chart comparing accuracy/F1/precision/recall.

    Args:
        metrics_dict: ``{model_name: {metric_name: value}}``.
        output_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    metric_keys = ["accuracy", "f1", "precision", "recall"]
    model_names = list(metrics_dict.keys())
    n_models = len(model_names)
    n_metrics = len(metric_keys)

    x = np.arange(n_metrics)
    width = 0.8 / n_models
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))  # type: ignore

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = [metrics_dict[name].get(mk, 0.0) for mk in metric_keys]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                      label=name, color=color)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metric_keys])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — Parameter comparison
# ---------------------------------------------------------------------------

def plot_parameter_comparison(
    param_counts: Dict[str, int],
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Figure 6: Parameter count comparison (log-scale bar chart).

    Args:
        param_counts: ``{model_name: n_params}``.
        output_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    names = list(param_counts.keys())
    counts = [param_counts[n] for n in names]
    colors = plt.cm.Paired(np.linspace(0, 1, len(names)))  # type: ignore

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_ylabel("Number of Parameters (log scale)")
    ax.set_title("Trainable Parameter Counts", fontsize=13, fontweight="bold")
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            c * 1.15,
            f"{c:,}", ha="center", va="bottom", fontsize=8,
        )
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 7 — Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(
    learning_curve_results: Dict,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Figure 7: Validation accuracy learning curves with error bands.

    Args:
        learning_curve_results: ``{model_name: {fraction: {"mean": ..., "std": ...}}}``.
        output_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(learning_curve_results)))  # type: ignore

    for color, (model_name, frac_data) in zip(
        colors, learning_curve_results.items()
    ):
        fracs = sorted(frac_data.keys())
        means = [frac_data[f]["mean"] for f in fracs]
        stds = [frac_data[f]["std"] for f in fracs]
        ax.plot(fracs, means, "o-", label=model_name, color=color, lw=2)
        ax.fill_between(
            fracs,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15, color=color,
        )

    ax.set_xlabel("Training Fraction")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Learning Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 8 — Confusion matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    cm_dict: Dict[str, np.ndarray],
    class_names: Optional[List[str]] = None,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Figure 8: Confusion matrices for a set of models.

    Args:
        cm_dict: ``{model_name: confusion_matrix}``.
        class_names: Class name strings.
        output_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    import seaborn as sns

    n = len(cm_dict)
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (model_name, cm) in zip(axes, cm_dict.items()):
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names or range(cm.shape[1]),
            yticklabels=class_names or range(cm.shape[0]),
            ax=ax, cbar=True,
        )
        ax.set_title(f"{model_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.suptitle("Normalised Confusion Matrices", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 9 — Accuracy vs complexity
# ---------------------------------------------------------------------------

def plot_accuracy_vs_complexity(
    param_counts: Dict[str, int],
    accuracies: Dict[str, float],
    model_names: Optional[List[str]] = None,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Figure 9: Accuracy vs parameter count scatter plot.

    Args:
        param_counts: ``{model_name: n_params}``.
        accuracies: ``{model_name: accuracy}``.
        model_names: Subset of model names to plot.
        output_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    names = model_names or list(param_counts.keys())
    xs = [param_counts[n] for n in names]
    ys = [accuracies.get(n, 0.0) for n in names]
    colors = plt.cm.Set1(np.linspace(0, 1, len(names)))  # type: ignore

    fig, ax = plt.subplots(figsize=(8, 5))
    for x, y, name, color in zip(xs, ys, names, colors):
        ax.scatter(x, y, s=120, color=color, zorder=5, edgecolors="white", lw=1.5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters (log scale)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs Model Complexity", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 10 — Ablation study
# ---------------------------------------------------------------------------

def plot_ablation_study(
    ablation_results: Dict,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Figure 10: Heatmap and bar charts from ablation study results.

    Args:
        ablation_results: ``{(n_qubits, n_layers): accuracy}`` or
            ``{"n_qubits": [...], "n_layers": [...], "acc": [...]}``.
        output_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Expect list-based format
    if isinstance(ablation_results, dict) and "n_qubits" in ablation_results:
        n_qubits_list = ablation_results["n_qubits"]
        n_layers_list = ablation_results["n_layers"]
        acc_list = ablation_results["acc"]

        unique_q = sorted(set(n_qubits_list))
        unique_l = sorted(set(n_layers_list))
        heat = np.zeros((len(unique_q), len(unique_l)))
        for q, l, a in zip(n_qubits_list, n_layers_list, acc_list):
            heat[unique_q.index(q), unique_l.index(l)] = a

        import seaborn as sns

        sns.heatmap(
            heat, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=[f"{l}L" for l in unique_l],
            yticklabels=[f"{q}Q" for q in unique_q],
            ax=axes[0], cbar=True,
        )
        axes[0].set_xlabel("# Layers")
        axes[0].set_ylabel("# Qubits")
        axes[0].set_title("Ablation: Qubits × Layers", fontweight="bold")

        # Bar for aggregated per-n_qubits mean
        q_means = {q: np.mean([a for qq, a in zip(n_qubits_list, acc_list) if qq == q])
                   for q in unique_q}
        axes[1].bar([str(q) for q in unique_q], list(q_means.values()), color="#9C27B0")
        axes[1].set_xlabel("# Qubits")
        axes[1].set_ylabel("Mean Accuracy")
        axes[1].set_title("Accuracy by Qubit Count", fontweight="bold")
        axes[1].grid(axis="y", alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No ablation data", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No ablation data", ha="center", va="center")

    fig.suptitle("Ablation Study Results", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 11 — Robustness analysis
# ---------------------------------------------------------------------------

def plot_robustness_analysis(
    noise_levels: List[float],
    accuracies_dict: Dict[str, List[float]],
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Figure 11: Accuracy vs noise level for multiple models.

    Args:
        noise_levels: List of noise standard deviations tested.
        accuracies_dict: ``{model_name: [acc_at_noise_0, acc_at_noise_1, ...]}``.
        output_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(accuracies_dict)))  # type: ignore

    for color, (name, accs) in zip(colors, accuracies_dict.items()):
        ax.plot(noise_levels, accs, "o-", label=name, color=color, lw=2)

    ax.set_xlabel("Noise Level (std)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness to Input Noise", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 12 — Runtime comparison
# ---------------------------------------------------------------------------

def plot_runtime_comparison(
    runtime_dict: Dict[str, float],
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Figure 12: Bar chart comparing mean inference time per sample.

    Args:
        runtime_dict: ``{model_name: mean_ms_per_sample}``.
        output_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    names = list(runtime_dict.keys())
    times = [runtime_dict[n] for n in names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))  # type: ignore

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, times, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Inference Time (ms / sample)")
    ax.set_title("Inference Runtime Comparison", fontsize=13, fontweight="bold")
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            t * 1.02,
            f"{t:.2f}ms", ha="center", va="bottom", fontsize=9,
        )
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 13 — Feature Importance (NEW)
# ---------------------------------------------------------------------------

def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    output_path: Optional[str | Path] = None,
    top_n: int = 20,
) -> plt.Figure:
    """Figure 13: Horizontal bar chart showing top N feature importances.

    Args:
        feature_names: List of feature names.
        importance_scores: 1-D array of importance values (higher = more important).
        output_path: Optional save path.
        top_n: Number of top features to display.

    Returns:
        Matplotlib figure.
    """
    # Sort by importance and take top N
    indices = np.argsort(importance_scores)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = importance_scores[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))  # type: ignore
    y_pos = np.arange(len(top_features))

    bars = ax.barh(y_pos, top_scores, color=colors, edgecolor="white", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(
        f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for bar, score in zip(bars, top_scores):
        ax.text(
            score + 0.01 * max(top_scores),
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 14 — Quantum Circuit Depth vs Accuracy (NEW)
# ---------------------------------------------------------------------------

def plot_circuit_depth_vs_accuracy(
    depth_data: Dict[str, List],
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Figure 14: Line plot showing how quantum circuit depth affects accuracy.

    Args:
        depth_data: Dictionary with keys "depths", "accuracies", optionally "std_devs".
        output_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    depths = depth_data.get("depths", [])
    accuracies = depth_data.get("accuracies", [])
    std_devs = depth_data.get("std_devs", None)

    fig, ax = plt.subplots(figsize=(10, 6))

    if std_devs is not None:
        ax.errorbar(
            depths,
            accuracies,
            yerr=std_devs,
            fmt="o-",
            color="#9C27B0",
            ecolor="#E1BEE7",
            capsize=5,
            capthick=2,
            linewidth=2,
            markersize=8,
        )
    else:
        ax.plot(depths, accuracies, "o-", color="#9C27B0", linewidth=2, markersize=8)

    ax.set_xlabel("Circuit Depth (n_qubits × n_layers)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(
        "Quantum Circuit Depth vs Classification Accuracy",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Add trend annotation
    if len(depths) > 1 and len(accuracies) > 1:
        # Simple linear fit for trend
        z = np.polyfit(depths, accuracies, 1)
        trend = "increasing" if z[0] > 0 else "decreasing"
        ax.text(
            0.05,
            0.95,
            f"Trend: {trend}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 15 — Training Convergence Comparison (NEW)
# ---------------------------------------------------------------------------

def plot_training_convergence(
    history_dict: Dict[str, Dict[str, List]],
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Figure 15: Training and validation loss curves for multiple models.

    Args:
        history_dict: ``{model_name: {"train_loss": [...], "val_loss": [...]}}``.
        output_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(history_dict)))  # type: ignore

    for color, (name, history) in zip(colors, history_dict.items()):
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        epochs = list(range(1, len(train_loss) + 1))

        # Training loss subplot
        ax1.plot(epochs, train_loss, "-", label=name, color=color, linewidth=2)

        # Validation loss subplot
        ax2.plot(epochs, val_loss, "-", label=name, color=color, linewidth=2)

    # Configure training loss plot
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Training Loss", fontsize=11)
    ax1.set_title("Training Loss Convergence", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Configure validation loss plot
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Validation Loss", fontsize=11)
    ax2.set_title("Validation Loss Convergence", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Training Convergence Comparison",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig
