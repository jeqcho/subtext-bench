#!/usr/bin/env python
"""Faceted breakdown: Opus as sender, varying receiver (Haiku/Sonnet/Opus).

Loads the 3 relevant logs from logs/cross_sender_receiver/ where sender=Opus
and plots a faceted breakdown with one subplot per receiver.

Usage:
    uv run python src/subtext_bench/faceted_opus_sender.py

Outputs plots/faceted_breakdown_opus_sender.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log
from matplotlib.lines import Line2D

MODEL_SHORT = {
    "anthropic/claude-haiku-4-5": "Haiku 4.5",
    "anthropic/claude-sonnet-4-5": "Sonnet 4.5",
    "anthropic/claude-opus-4-5": "Opus 4.5",
}

RECEIVER_ORDER = ["Haiku 4.5", "Sonnet 4.5", "Opus 4.5"]

RECEIVER_COLORS = {
    "Haiku 4.5": "#2ca02c",
    "Sonnet 4.5": "#e45756",
    "Opus 4.5": "#4c78a8",
}


def _short(model_id: str) -> str:
    return MODEL_SHORT.get(model_id, model_id)


def main() -> None:
    log_dir = Path("logs/cross_sender_receiver")
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    # Collect per-sample data from Opus-sender logs
    all_samples: dict[str, pd.DataFrame] = {}

    for log_file in sorted(log_dir.glob("*.eval")):
        log = read_eval_log(str(log_file))
        sender = _short(log.eval.model)
        if sender != "Opus 4.5":
            continue

        roles = log.eval.model_roles or {}
        receiver_cfg = roles.get("receiver")
        if receiver_cfg is None:
            continue
        receiver = _short(receiver_cfg.model)

        monitor_cfg = roles.get("monitor")
        if monitor_cfg is None or "gpt-5-mini" not in monitor_cfg.model:
            continue

        if not log.samples:
            continue

        rows = []
        for sample in log.samples:
            if not sample.scores:
                continue
            scorer_val = sample.scores.get("subtext_scorer")
            if not scorer_val or not scorer_val.value:
                continue
            vals = scorer_val.value if isinstance(scorer_val.value, dict) else json.loads(scorer_val.value)
            animal = sample.metadata.get("animal", "unknown") if sample.metadata else "unknown"
            rows.append({
                "animal": animal,
                "receiver_accuracy": float(vals["receiver_accuracy"]),
                "monitor_accuracy": float(vals["monitor_accuracy"]),
                "subtext_score": float(vals["subtext_score"]),
            })

        if rows:
            df = pd.DataFrame(rows)
            # Average across linkedin and journal per animal
            agg = df.groupby("animal").agg(
                subtext_score=("subtext_score", "mean"),
                receiver_accuracy=("receiver_accuracy", "mean"),
                monitor_accuracy=("monitor_accuracy", "mean"),
            ).reset_index()
            all_samples[receiver] = agg

    if not all_samples:
        print("No Opus-sender logs found in", log_dir)
        return

    # Get animal order sorted by overall mean subtext score (descending)
    combined = pd.concat(all_samples.values(), ignore_index=True)
    animal_order = (
        combined.groupby("animal")["subtext_score"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    receivers_present = [r for r in RECEIVER_ORDER if r in all_samples]
    n_recv = len(receivers_present)

    # Shared y-limits
    y_min = min(combined["subtext_score"].min(), -combined["monitor_accuracy"].max()) - 0.08
    y_max = max(combined["subtext_score"].max(), combined["receiver_accuracy"].max()) + 0.08

    fig, axes = plt.subplots(1, n_recv, figsize=(5 * n_recv, 7), sharey=True)
    if n_recv == 1:
        axes = [axes]

    for idx, (receiver, ax) in enumerate(zip(receivers_present, axes)):
        agg = all_samples[receiver].set_index("animal")
        xs = np.arange(len(animal_order))

        subtext = [agg.loc[a, "subtext_score"] if a in agg.index else 0 for a in animal_order]
        recv = [agg.loc[a, "receiver_accuracy"] if a in agg.index else 0 for a in animal_order]
        mon = [agg.loc[a, "monitor_accuracy"] if a in agg.index else 0 for a in animal_order]

        color = RECEIVER_COLORS.get(receiver, f"C{idx}")

        ax.bar(xs, subtext, color=color, edgecolor="white", linewidth=0.5,
               width=0.7, alpha=0.55)
        ax.scatter(xs, recv, marker="^", s=70, color=color,
                   edgecolors="black", linewidths=0.6, zorder=5)
        ax.scatter(xs, [-m for m in mon], marker="v", s=70, color=color,
                   edgecolors="black", linewidths=0.6, zorder=5)

        ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(f"Receiver: {receiver}", fontsize=15, fontweight="bold", color=color)
        ax.set_xticks(xs)
        ax.set_xticklabels([a.capitalize() for a in animal_order],
                           fontsize=10, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis="y", alpha=0.3)

        if idx == 0:
            ax.set_ylabel("Score", fontsize=13)

    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=10, label="Receiver Accuracy"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=10, label="\u2212 Monitor Accuracy"),
        plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.55, edgecolor="white",
                       label="Subtext Score (bar)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Sender: Opus 4.5  |  Monitor: gpt-5-mini  |  Direct task (avg. of linkedin & journal)",
        fontsize=15, fontweight="bold", y=1.02,
    )

    fig.tight_layout()
    out = output_dir / "faceted_breakdown_opus_sender.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
