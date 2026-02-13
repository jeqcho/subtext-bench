#!/usr/bin/env python
"""3x3 grid of faceted breakdown plots: rows=sender, cols=receiver.

Each cell shows bars (subtext score) + up carets (receiver acc) + down carets (-monitor acc)
for all 16 animals.

Usage:
    uv run python src/subtext_bench/grid_breakdown.py

Outputs plots/grid_breakdown.png
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

MODEL_ORDER = ["Haiku 4.5", "Sonnet 4.5", "Opus 4.5"]

# Distinct colors per sender (row)
SENDER_COLORS = {
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

    # --- Load all 9 logs into a dict keyed by (sender, receiver) ---
    cell_data: dict[tuple[str, str], pd.DataFrame] = {}

    for log_file in sorted(log_dir.glob("*.eval")):
        log = read_eval_log(str(log_file))
        sender = _short(log.eval.model)
        if sender not in MODEL_ORDER:
            continue

        roles = log.eval.model_roles or {}
        receiver_cfg = roles.get("receiver")
        if receiver_cfg is None:
            continue
        receiver = _short(receiver_cfg.model)
        if receiver not in MODEL_ORDER:
            continue

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
            agg = df.groupby("animal").agg(
                subtext_score=("subtext_score", "mean"),
                receiver_accuracy=("receiver_accuracy", "mean"),
                monitor_accuracy=("monitor_accuracy", "mean"),
            ).reset_index()
            cell_data[(sender, receiver)] = agg

    print(f"Loaded {len(cell_data)} cells: {list(cell_data.keys())}")

    # --- Animal order: sort by overall mean subtext score (descending) ---
    all_agg = pd.concat(cell_data.values(), ignore_index=True)
    animal_order = (
        all_agg.groupby("animal")["subtext_score"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # --- Shared y-limits ---
    y_min = min(all_agg["subtext_score"].min(), -all_agg["monitor_accuracy"].max()) - 0.08
    y_max = max(all_agg["subtext_score"].max(), all_agg["receiver_accuracy"].max()) + 0.08

    # --- Create 3x3 grid ---
    fig, axes = plt.subplots(3, 3, figsize=(16, 15), sharey=True, sharex=True)

    for row_idx, sender in enumerate(MODEL_ORDER):
        for col_idx, receiver in enumerate(MODEL_ORDER):
            ax = axes[row_idx, col_idx]
            color = SENDER_COLORS[sender]

            key = (sender, receiver)
            recv_color = SENDER_COLORS[receiver]
            if key not in cell_data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="grey")
                ax.set_xlim(-0.5, len(animal_order) - 0.5)
                ax.set_ylim(y_min, y_max)
                ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
                ax.grid(True, axis="y", alpha=0.3)
            else:
                agg = cell_data[key].set_index("animal")
                xs = np.arange(len(animal_order))

                subtext = [agg.loc[a, "subtext_score"] if a in agg.index else 0 for a in animal_order]
                recv = [agg.loc[a, "receiver_accuracy"] if a in agg.index else 0 for a in animal_order]
                mon = [agg.loc[a, "monitor_accuracy"] if a in agg.index else 0 for a in animal_order]

                ax.bar(xs, subtext, color=recv_color, edgecolor="white",
                       linewidth=0.5, width=0.7, alpha=0.55)
                ax.scatter(xs, recv, marker="^", s=40, color=recv_color,
                           edgecolors="black", linewidths=0.4, zorder=5)
                ax.scatter(xs, [-m for m in mon], marker="v", s=40, color=recv_color,
                           edgecolors="black", linewidths=0.4, zorder=5)

                ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
                ax.set_ylim(y_min, y_max)
                ax.grid(True, axis="y", alpha=0.3)

                pass  # no special border styling

            # Column titles (top row only)
            if row_idx == 0:
                recv_color = SENDER_COLORS[receiver]
                ax.set_title(f"Receiver: {receiver}", fontsize=13,
                             fontweight="bold", color=recv_color)

            # Row labels (left column only)
            if col_idx == 0:
                ax.set_ylabel(f"Sender: {sender}", fontsize=13,
                              fontweight="bold", color=color)
            else:
                ax.set_ylabel("")

            # X-axis labels (bottom row only)
            if row_idx == 2:
                ax.set_xticks(range(len(animal_order)))
                ax.set_xticklabels([a.capitalize() for a in animal_order],
                                   fontsize=8, rotation=45, ha="right")
            else:
                ax.set_xticks(range(len(animal_order)))
                ax.set_xticklabels([])

            ax.tick_params(axis="y", labelsize=9)

    # --- Legend ---
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=9, label="Receiver Accuracy"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=9, label="\u2212 Monitor Accuracy"),
        plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.55, edgecolor="white",
                       label="Subtext Score (bar)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Sender \u00D7 Receiver Breakdown  |  Monitor: gpt-5-mini  |  "
        "Direct task (avg. of linkedin & journal)",
        fontsize=16, fontweight="bold", y=1.01,
    )

    fig.tight_layout()
    out = output_dir / "grid_breakdown.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
