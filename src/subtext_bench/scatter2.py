#!/usr/bin/env python
"""Generate a matplotlib scatter plot from an Inspect AI eval log.

Usage:
    uv run python src/subtext_bench/scatter2.py [LOG_PATH_OR_DIR]

Outputs plots/scatter2.png
  x-axis: receiver accuracy
  y-axis: monitor accuracy
  size:   subtext score (clipped to 0)
  shape:  animal
  color:  task
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from inspect_ai.analysis import EvalModel, SampleSummary, evals_df, model_info, prepare, samples_df


def _monitor_model_name(eval_df) -> str:
    """Extract the monitor model name from model_roles, falling back to the default."""
    roles_raw = eval_df["model_roles"].iloc[0]
    if roles_raw:
        try:
            roles = json.loads(roles_raw) if isinstance(roles_raw, str) else roles_raw
            if "monitor" in roles:
                return roles["monitor"].get("model", "openai/gpt-5.2")
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
    return "openai/gpt-5.2"

# 16 distinct matplotlib markers -- one per animal
ANIMAL_MARKERS = {
    "bear": "o",
    "cat": "s",
    "dog": "D",
    "dolphin": "^",
    "dragon": "v",
    "eagle": "<",
    "elephant": ">",
    "fox": "p",
    "leopard": "h",
    "lion": "H",
    "owl": "*",
    "panda": "X",
    "phoenix": "P",
    "tiger": "d",
    "whale": "8",
    "wolf": (5, 0, 1),  # rotated pentagon (filled)
}

# Use a qualitative colormap for 24 tasks
TASK_CMAP = plt.colormaps.get_cmap("tab20").resampled(24)


def main() -> None:
    log_path = sys.argv[1] if len(sys.argv) > 1 else "logs"
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    # --- Read data ---
    eval_df = evals_df(log_path, quiet=True)
    eval_df = prepare(eval_df, [model_info()])
    task_name = str(eval_df["task_display_name"].iloc[0])
    model_name = eval_df.get("model_display_name", eval_df["model"]).iloc[0]
    monitor_name = _monitor_model_name(eval_df)

    samples = samples_df(log_path, columns=SampleSummary + EvalModel, quiet=True)
    scores = samples["score_subtext_scorer"].apply(json.loads)
    for m in ("receiver_accuracy", "monitor_accuracy", "subtext_score"):
        samples[m] = scores.apply(lambda x, m=m: float(x[m]))
    samples["animal"] = samples["metadata_animal"].astype(str)
    samples["task"] = samples["metadata_task"].astype(str)

    # --- Build colour / marker maps ---
    tasks_sorted = sorted(samples["task"].unique())
    task_to_color = {t: TASK_CMAP(i) for i, t in enumerate(tasks_sorted)}
    animals_sorted = sorted(samples["animal"].unique())

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 9))

    # Size scaling: map subtext_score -> point area.  clip negatives to a small floor.
    size_raw = samples["subtext_score"].clip(lower=0).values
    sizes = 30 + size_raw * 600  # min 30pt, max ~630pt for score=1.0

    # Jitter to reduce overplotting at (0, 0)
    rng = np.random.default_rng(42)
    jitter_x = rng.normal(0, 0.008, size=len(samples))
    jitter_y = rng.normal(0, 0.008, size=len(samples))

    for animal in animals_sorted:
        mask = samples["animal"] == animal
        sub = samples[mask]
        colors = [task_to_color[t] for t in sub["task"]]
        ax.scatter(
            sub["receiver_accuracy"].values + jitter_x[mask.values],
            sub["monitor_accuracy"].values + jitter_y[mask.values],
            s=sizes[mask.values],
            c=colors,
            marker=ANIMAL_MARKERS[animal],
            edgecolors="black",
            linewidths=0.4,
            alpha=0.75,
            label=animal,
        )

    # --- Axis formatting ---
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Receiver Accuracy", fontsize=13)
    ax.set_ylabel("Monitor Accuracy", fontsize=13)
    ax.set_title(
        f"{task_name} — Receiver ({model_name}) vs Monitor ({monitor_name}) Accuracy",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)

    # Diagonal reference line (receiver == monitor → subtext_score = 0)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", alpha=0.5, lw=1, zorder=0)

    # --- Animal legend (markers) ---
    animal_handles = [
        mlines.Line2D(
            [], [],
            marker=ANIMAL_MARKERS[a],
            color="grey",
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=8,
            linestyle="None",
            label=a,
        )
        for a in animals_sorted
    ]
    legend1 = ax.legend(
        handles=animal_handles,
        title="Animal",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7,
        title_fontsize=8,
        framealpha=0.9,
        ncol=1,
    )
    ax.add_artist(legend1)

    # --- Size legend (subtext_score) ---
    size_values = [0.0, 0.2, 0.5, 1.0]
    size_handles = [
        mlines.Line2D(
            [], [],
            marker="o",
            color="grey",
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=np.sqrt(30 + v * 600) * 0.65,
            linestyle="None",
            label=f"{v:.1f}",
        )
        for v in size_values
    ]
    legend_size = ax.legend(
        handles=size_handles,
        title="Subtext Score\n(size)",
        loc="upper left",
        bbox_to_anchor=(1.19, 1.0),
        fontsize=7,
        title_fontsize=8,
        framealpha=0.9,
        ncol=1,
        labelspacing=1.4,
    )
    ax.add_artist(legend_size)

    # --- Task legend (colors) ---
    task_handles = [
        mlines.Line2D(
            [], [],
            marker="o",
            color=task_to_color[t],
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=6,
            linestyle="None",
            label=t,
        )
        for t in tasks_sorted
    ]
    ax.legend(
        handles=task_handles,
        title="Task",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        fontsize=6,
        title_fontsize=8,
        framealpha=0.9,
        ncol=1,
    )

    fig.tight_layout()
    out = output_dir / "scatter2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
