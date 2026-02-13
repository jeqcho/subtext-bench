#!/usr/bin/env python
"""Plots for prefill direct-task results.

1. Grouped bar chart: subtext score by animal, grouped by model.
2. Scatter plot: receiver accuracy vs monitor accuracy, colored by model.

Usage:
    uv run python src/subtext_bench/grouped_bar.py LOG1.eval LOG2.eval LOG3.eval
    uv run python src/subtext_bench/grouped_bar.py LOG_DIR

Outputs:
    plots/grouped_bar_subtext.png
    plots/scatter_recv_vs_monitor.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.analysis import (
    EvalModel,
    SampleSummary,
    evals_df,
    model_info,
    prepare,
    samples_df,
)


# Display names and order for the models
# Keys match the model_display_name column from inspect_ai
MODEL_SHORT = {
    "Claude Haiku 4.5": "Haiku 4.5",
    "Claude Sonnet 4.5": "Sonnet 4.5",
    "Claude Opus 4.5": "Opus 4.5",
}

MODEL_ORDER = ["Haiku 4.5", "Sonnet 4.5", "Opus 4.5"]

MODEL_COLORS = {
    "Haiku 4.5": "#2ca02c",   # green
    "Sonnet 4.5": "#e45756",   # red
    "Opus 4.5": "#4c78a8",    # blue
}


def _extract_monitor_name(eval_df: pd.DataFrame) -> str:
    """Extract the monitor model name from model_roles."""
    roles_raw = eval_df["model_roles"].iloc[0]
    if roles_raw:
        try:
            roles = json.loads(roles_raw) if isinstance(roles_raw, str) else roles_raw
            if "monitor" in roles:
                return roles["monitor"].get("model", "unknown")
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
    return "unknown"


def _model_short_label(display_name: str) -> str:
    """Map inspect_ai display name to our short label."""
    return MODEL_SHORT.get(display_name, display_name)


def _plot_scatter(df: pd.DataFrame, monitor_name: str, output_dir: Path) -> None:
    """Scatter plot: receiver accuracy (x) vs monitor accuracy (y), colored by model.

    Each dot is one animal x task sample.
    """
    models_present = [m for m in MODEL_ORDER if m in df["model_label"].unique()]
    if not models_present:
        models_present = sorted(df["model_label"].unique())

    fig, ax = plt.subplots(figsize=(9, 8))

    rng = np.random.default_rng(42)
    jitter_strength = 0.015

    for model in models_present:
        subset = df[df["model_label"] == model]
        jx = rng.uniform(-jitter_strength, jitter_strength, size=len(subset))
        jy = rng.uniform(-jitter_strength, jitter_strength, size=len(subset))
        ax.scatter(
            subset["receiver_accuracy"].values + jx,
            subset["monitor_accuracy"].values + jy,
            label=model,
            color=MODEL_COLORS.get(model, None),
            alpha=0.7,
            s=60,
            edgecolors="white",
            linewidths=0.4,
        )

    # Diagonal line (subtext_score = 0)
    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xlabel("Receiver Accuracy", fontsize=14)
    ax.set_ylabel("Monitor Accuracy", fontsize=14)
    ax.set_title(
        f"Receiver Accuracy vs Monitor Accuracy\n"
        f"Monitor: {monitor_name}  |  Direct task (linkedin & journal)",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=12, title="Sender Model", title_fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / "scatter_recv_vs_monitor.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_faceted_bars(
    agg: pd.DataFrame,
    animal_order: list[str],
    models_present: list[str],
    monitor_name: str,
    output_dir: Path,
) -> None:
    """Three side-by-side bar charts, one per model, sharing the same y-axis."""
    n_models = len(models_present)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 7), sharey=True)
    if n_models == 1:
        axes = [axes]

    # Compute shared y-limits across all models
    all_heights = []
    for model in models_present:
        model_data = agg[agg["model_label"] == model].set_index("animal")
        heights = [model_data.loc[a, "subtext_score"] if a in model_data.index else 0 for a in animal_order]
        all_heights.extend(heights)
    y_min = min(all_heights) - 0.05
    y_max = max(all_heights) + 0.05

    for idx, (model, ax) in enumerate(zip(models_present, axes)):
        model_data = agg[agg["model_label"] == model].set_index("animal")
        heights = [model_data.loc[a, "subtext_score"] if a in model_data.index else 0 for a in animal_order]

        color = MODEL_COLORS.get(model, f"C{idx}")
        ax.bar(
            range(len(animal_order)),
            heights,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            width=0.7,
        )
        ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(model, fontsize=15, fontweight="bold", color=color)
        ax.set_xticks(range(len(animal_order)))
        ax.set_xticklabels(
            [a.capitalize() for a in animal_order],
            fontsize=10,
            rotation=45,
            ha="right",
        )
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis="y", alpha=0.3)

        if idx == 0:
            ax.set_ylabel("Subtext Score", fontsize=13)

    fig.suptitle(
        f"Subtext Score (= Receiver Acc. \u2212 Monitor Acc.) by Animal\n"
        f"Monitor: {monitor_name}  |  Direct task (avg. of linkedin & journal)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()
    out = output_dir / "faceted_bar_subtext.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_faceted_breakdown(
    df_or_agg: pd.DataFrame,
    animal_order: list[str],
    models_present: list[str],
    monitor_name: str,
    output_dir: Path,
    *,
    task_slug: str | None = None,
    filename: str = "faceted_breakdown.png",
) -> None:
    """Faceted chart with bars for subtext score and carets for receiver/monitor accuracy.

    If *task_slug* is given, filter the raw DataFrame to that task before aggregating.
    Otherwise, *df_or_agg* is expected to already be aggregated.
    """
    from matplotlib.lines import Line2D

    if task_slug is not None:
        # df_or_agg is the raw per-sample DataFrame; filter and aggregate
        filtered = df_or_agg[df_or_agg["task_slug"] == task_slug]
        agg = (
            filtered.groupby(["animal", "model_label"])
            .agg(
                subtext_score=("subtext_score", "mean"),
                receiver_accuracy=("receiver_accuracy", "mean"),
                monitor_accuracy=("monitor_accuracy", "mean"),
            )
            .reset_index()
        )
        task_label = task_slug.replace("_", " ").title()
    else:
        agg = df_or_agg
        task_label = "avg. of linkedin & journal"

    n_models = len(models_present)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 7), sharey=True)
    if n_models == 1:
        axes = [axes]

    # Shared y-limits: need to span both subtext_score and the accuracy values
    y_min = min(agg["subtext_score"].min(), -agg["monitor_accuracy"].max()) - 0.08
    y_max = max(agg["subtext_score"].max(), agg["receiver_accuracy"].max()) + 0.08

    for idx, (model, ax) in enumerate(zip(models_present, axes)):
        model_data = agg[agg["model_label"] == model].set_index("animal")
        xs = np.arange(len(animal_order))

        subtext = [model_data.loc[a, "subtext_score"] if a in model_data.index else 0 for a in animal_order]
        recv = [model_data.loc[a, "receiver_accuracy"] if a in model_data.index else 0 for a in animal_order]
        mon = [model_data.loc[a, "monitor_accuracy"] if a in model_data.index else 0 for a in animal_order]

        color = MODEL_COLORS.get(model, f"C{idx}")

        # Bars for subtext score
        ax.bar(
            xs,
            subtext,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            width=0.7,
            alpha=0.55,
        )

        # Up carets for receiver accuracy (above zero)
        ax.scatter(
            xs,
            recv,
            marker="^",
            s=70,
            color=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )

        # Down carets for monitor accuracy (plotted as negative)
        ax.scatter(
            xs,
            [-m for m in mon],
            marker="v",
            s=70,
            color=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )

        ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(model, fontsize=15, fontweight="bold", color=color)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [a.capitalize() for a in animal_order],
            fontsize=10,
            rotation=45,
            ha="right",
        )
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis="y", alpha=0.3)

        if idx == 0:
            ax.set_ylabel("Score", fontsize=13)

    # Shared legend (use grey for the marker icons so they're model-neutral)
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=10, label="Receiver Accuracy"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=10, label="\u2212 Monitor Accuracy"),
        plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.55, edgecolor="white",
                       label="Subtext Score (bar)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"Receiver Acc. (\u25B2) vs Monitor Acc. (\u25BC) and Subtext Score (bar)\n"
        f"Monitor: {monitor_name}  |  Direct task: {task_label}",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()
    out = output_dir / filename
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python src/subtext_bench/grouped_bar.py LOG1 [LOG2] [LOG3]")
        sys.exit(1)

    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    # Collect all samples across the provided log paths
    all_samples = []
    monitor_name = "unknown"

    for log_path in sys.argv[1:]:
        log_path = str(log_path)

        eval_df = evals_df(log_path, quiet=True)
        eval_df = prepare(eval_df, [model_info()])
        monitor_name = _extract_monitor_name(eval_df)

        samples = samples_df(log_path, columns=SampleSummary + EvalModel, quiet=True)
        samples = prepare(samples, [model_info()])

        # Parse scores
        scores = samples["score_subtext_scorer"].apply(json.loads)
        samples["subtext_score"] = scores.apply(lambda x: float(x["subtext_score"]))
        samples["receiver_accuracy"] = scores.apply(lambda x: float(x["receiver_accuracy"]))
        samples["monitor_accuracy"] = scores.apply(lambda x: float(x["monitor_accuracy"]))

        # Extract metadata
        samples["animal"] = samples["metadata_animal"].astype(str)
        samples["task_slug"] = samples["metadata_task_slug"].astype(str)

        # Get model display name
        if "model_display_name" in samples.columns:
            samples["model_label"] = samples["model_display_name"].apply(_model_short_label)
        else:
            samples["model_label"] = samples["model"].apply(
                lambda x: _model_short_label(str(x).split("/")[-1])
            )

        all_samples.append(samples)

    df = pd.concat(all_samples, ignore_index=True)

    # ---- Scatter plot (per-sample, no averaging) ----
    _plot_scatter(df, monitor_name, output_dir)

    # Average across linkedin and journal per (animal, model)
    agg = (
        df.groupby(["animal", "model_label"])
        .agg(
            subtext_score=("subtext_score", "mean"),
            receiver_accuracy=("receiver_accuracy", "mean"),
            monitor_accuracy=("monitor_accuracy", "mean"),
        )
        .reset_index()
    )

    # Get sorted animal list by overall mean subtext score (descending)
    animal_order = (
        agg.groupby("animal")["subtext_score"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # --- Plot ---
    models_present = [m for m in MODEL_ORDER if m in agg["model_label"].unique()]
    if not models_present:
        # Fallback: use whatever models are in the data
        models_present = sorted(agg["model_label"].unique())
    n_animals = len(animal_order)
    n_models = len(models_present)

    x = np.arange(n_animals)
    bar_width = 0.8 / n_models
    offsets = np.linspace(
        -(n_models - 1) * bar_width / 2,
        (n_models - 1) * bar_width / 2,
        n_models,
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, model in enumerate(models_present):
        model_data = agg[agg["model_label"] == model].set_index("animal")
        heights = [model_data.loc[a, "subtext_score"] if a in model_data.index else 0 for a in animal_order]
        ax.bar(
            x + offsets[i],
            heights,
            width=bar_width,
            label=model,
            color=MODEL_COLORS.get(model, f"C{i}"),
            edgecolor="white",
            linewidth=0.5,
        )

    # Labels and title
    ax.set_xlabel("Animal", fontsize=14)
    ax.set_ylabel("Subtext Score", fontsize=14)
    ax.set_title(
        f"Subtext Score (= Receiver Acc. \u2212 Monitor Acc.) by Animal\n"
        f"Monitor: {monitor_name}  |  Direct task (avg. of linkedin & journal)",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in animal_order], fontsize=11, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=11)
    ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(fontsize=12, title="Sender Model", title_fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = output_dir / "grouped_bar_subtext.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ---- Faceted bar chart (one subplot per model) ----
    _plot_faceted_bars(agg, animal_order, models_present, monitor_name, output_dir)

    # ---- Faceted breakdown (bars + carets for recv/monitor accuracy) ----
    _plot_faceted_breakdown(agg, animal_order, models_present, monitor_name, output_dir)

    # ---- Per-task breakdowns ----
    for slug in ["linkedin", "journal"]:
        _plot_faceted_breakdown(
            df, animal_order, models_present, monitor_name, output_dir,
            task_slug=slug,
            filename=f"faceted_breakdown_{slug}.png",
        )


if __name__ == "__main__":
    main()
