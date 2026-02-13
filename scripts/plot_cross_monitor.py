#!/usr/bin/env python
"""Faceted breakdown plot where each facet is a different *monitor* model.

The sender is fixed (Opus 4.5) and all panels are colored in blue.

Usage:
    uv run python scripts/plot_cross_monitor.py

Outputs:
    plots/faceted_breakdown_cross_monitor_linkedin.png
"""

from __future__ import annotations

import json
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
from matplotlib.lines import Line2D

# ── Eval files (sender = Opus 4.5, varying monitors) ──────────────────────
EVAL_FILES = {
    "GPT-5-mini": "logs/2026-02-13T04-22-12+00-00_direct-subtext_PGRrNwqWSZ2ZYzWEYPiBUj.eval",
    "Haiku 4.5": "logs/cross_model/2026-02-13T05-09-15+00-00_direct-subtext_Ysmxx4HQGZDcDgCTenHdW3.eval",
    "Sonnet 4.5": "logs/cross_model/2026-02-13T05-09-58+00-00_direct-subtext_Ke3jwHTQ7EKivQJZRKoWcd.eval",
}

MONITOR_ORDER = ["GPT-5-mini", "Haiku 4.5", "Sonnet 4.5"]
BLUE = "#4c78a8"  # Opus blue from the reference palette


def _load_eval(label: str, path: str) -> pd.DataFrame:
    """Load one eval file, parse scores/metadata, tag with monitor label."""
    eval_df = evals_df(path, quiet=True)
    eval_df = prepare(eval_df, [model_info()])

    samples = samples_df(path, columns=SampleSummary + EvalModel, quiet=True)
    samples = prepare(samples, [model_info()])

    # Parse scorer output
    scores = samples["score_subtext_scorer"].apply(json.loads)
    samples["subtext_score"] = scores.apply(lambda x: float(x["subtext_score"]))
    samples["receiver_accuracy"] = scores.apply(lambda x: float(x["receiver_accuracy"]))
    samples["monitor_accuracy"] = scores.apply(lambda x: float(x["monitor_accuracy"]))

    # Metadata
    samples["animal"] = samples["metadata_animal"].astype(str)
    samples["task_slug"] = samples["metadata_task_slug"].astype(str)
    samples["monitor_label"] = label

    return samples


def main() -> None:
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    # Load all three eval files
    all_samples = []
    for label, path in EVAL_FILES.items():
        print(f"Loading {label} ← {path}")
        all_samples.append(_load_eval(label, path))
    df = pd.concat(all_samples, ignore_index=True)

    # Filter to linkedin task
    df = df[df["task_slug"] == "linkedin"]

    # Aggregate per (animal, monitor_label)
    agg = (
        df.groupby(["animal", "monitor_label"])
        .agg(
            subtext_score=("subtext_score", "mean"),
            receiver_accuracy=("receiver_accuracy", "mean"),
            monitor_accuracy=("monitor_accuracy", "mean"),
        )
        .reset_index()
    )

    # Animal order: descending mean subtext score across all monitors
    animal_order = (
        agg.groupby("animal")["subtext_score"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # ── Plot ───────────────────────────────────────────────────────────────
    n_monitors = len(MONITOR_ORDER)
    fig, axes = plt.subplots(1, n_monitors, figsize=(5 * n_monitors, 7), sharey=True)
    if n_monitors == 1:
        axes = [axes]

    # Shared y-limits
    y_min = min(agg["subtext_score"].min(), -agg["monitor_accuracy"].max()) - 0.08
    y_max = max(agg["subtext_score"].max(), agg["receiver_accuracy"].max()) + 0.08

    for idx, (monitor, ax) in enumerate(zip(MONITOR_ORDER, axes)):
        monitor_data = agg[agg["monitor_label"] == monitor].set_index("animal")
        xs = np.arange(len(animal_order))

        subtext = [
            monitor_data.loc[a, "subtext_score"] if a in monitor_data.index else 0
            for a in animal_order
        ]
        recv = [
            monitor_data.loc[a, "receiver_accuracy"] if a in monitor_data.index else 0
            for a in animal_order
        ]
        mon = [
            monitor_data.loc[a, "monitor_accuracy"] if a in monitor_data.index else 0
            for a in animal_order
        ]

        # Bars for subtext score
        ax.bar(
            xs,
            subtext,
            color=BLUE,
            edgecolor="white",
            linewidth=0.5,
            width=0.7,
            alpha=0.55,
        )

        # Up carets for receiver accuracy
        ax.scatter(
            xs,
            recv,
            marker="^",
            s=70,
            color=BLUE,
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
            color=BLUE,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )

        ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(monitor, fontsize=15, fontweight="bold", color=BLUE)
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

    # Shared legend
    legend_elements = [
        Line2D(
            [0], [0],
            marker="^", color="w", markerfacecolor="grey",
            markeredgecolor="black", markersize=10, label="Receiver Accuracy",
        ),
        Line2D(
            [0], [0],
            marker="v", color="w", markerfacecolor="grey",
            markeredgecolor="black", markersize=10, label="\u2212 Monitor Accuracy",
        ),
        plt.Rectangle(
            (0, 0), 1, 1, fc="grey", alpha=0.55, edgecolor="white",
            label="Subtext Score (bar)",
        ),
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
        "Receiver Acc. (\u25B2) vs Monitor Acc. (\u25BC) and Subtext Score (bar)\n"
        "Sender: Opus 4.5  |  Direct task: Linkedin",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()
    out = output_dir / "faceted_breakdown_cross_monitor_linkedin.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
