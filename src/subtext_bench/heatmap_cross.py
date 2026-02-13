#!/usr/bin/env python
"""3x3 heatmap: sender (rows) x receiver (cols) for recv acc, mon acc, subtext score.

Loads all .eval logs from a directory, extracts sender/receiver model info,
and plots three heatmap subplots.

Usage:
    uv run python src/subtext_bench/heatmap_cross.py logs/cross_sender_receiver/

Outputs plots/heatmap_cross.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log


MODEL_SHORT = {
    "anthropic/claude-haiku-4-5": "Haiku 4.5",
    "anthropic/claude-sonnet-4-5": "Sonnet 4.5",
    "anthropic/claude-opus-4-5": "Opus 4.5",
}

MODEL_ORDER = ["Haiku 4.5", "Sonnet 4.5", "Opus 4.5"]


def _short(model_id: str) -> str:
    return MODEL_SHORT.get(model_id, model_id)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python src/subtext_bench/heatmap_cross.py LOG_DIR")
        sys.exit(1)

    log_dir = Path(sys.argv[1])
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    rows: list[dict] = []

    for log_file in sorted(log_dir.glob("*.eval")):
        log = read_eval_log(str(log_file))

        sender = _short(log.eval.model)
        roles = log.eval.model_roles or {}
        receiver_cfg = roles.get("receiver")
        if receiver_cfg is not None:
            receiver = _short(receiver_cfg.model)
        else:
            # Default receiver -- skip if not one of our 3 models
            continue

        monitor_cfg = roles.get("monitor")
        monitor = _short(monitor_cfg.model) if monitor_cfg else "unknown"

        # Only include runs with gpt-5-mini as monitor
        if "gpt-5-mini" not in (monitor_cfg.model if monitor_cfg else ""):
            continue

        # Extract per-sample scores and average
        if not log.samples:
            continue

        recv_accs = []
        mon_accs = []
        sub_scores = []
        for sample in log.samples:
            if sample.scores:
                scorer_val = sample.scores.get("subtext_scorer")
                if scorer_val and scorer_val.value:
                    vals = scorer_val.value if isinstance(scorer_val.value, dict) else json.loads(scorer_val.value)
                    recv_accs.append(float(vals["receiver_accuracy"]))
                    mon_accs.append(float(vals["monitor_accuracy"]))
                    sub_scores.append(float(vals["subtext_score"]))

        if not recv_accs:
            continue

        rows.append({
            "sender": sender,
            "receiver": receiver,
            "receiver_accuracy": np.mean(recv_accs),
            "monitor_accuracy": np.mean(mon_accs),
            "subtext_score": np.mean(sub_scores),
        })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} runs:")
    print(df[["sender", "receiver", "receiver_accuracy", "monitor_accuracy", "subtext_score"]].to_string(index=False))

    # Build 3x3 matrices
    metrics = [
        ("receiver_accuracy", "Receiver Accuracy", "Blues"),
        ("monitor_accuracy", "Monitor Accuracy", "Reds"),
        ("subtext_score", "Subtext Score\n(= Recv Acc. \u2212 Mon Acc.)", "RdYlGn"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (metric, title, cmap) in zip(axes, metrics):
        matrix = np.full((3, 3), np.nan)
        for _, row in df.iterrows():
            si = MODEL_ORDER.index(row["sender"]) if row["sender"] in MODEL_ORDER else -1
            ri = MODEL_ORDER.index(row["receiver"]) if row["receiver"] in MODEL_ORDER else -1
            if si >= 0 and ri >= 0:
                matrix[si, ri] = row[metric]

        if metric == "subtext_score":
            vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
            im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=max(0.4, np.nanmax(matrix)))

        # Annotate cells
        for i in range(3):
            for j in range(3):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if abs(val) > 0.15 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=13, fontweight="bold", color=text_color)
                else:
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=11, color="grey")

        ax.set_xticks(range(3))
        ax.set_xticklabels(MODEL_ORDER, fontsize=12)
        ax.set_yticks(range(3))
        ax.set_yticklabels(MODEL_ORDER, fontsize=12)
        ax.set_xlabel("Receiver", fontsize=13)
        if ax == axes[0]:
            ax.set_ylabel("Sender", fontsize=13)
        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Sender x Receiver Heatmap  |  Monitor: gpt-5-mini  |  Direct task (linkedin & journal)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()
    out = output_dir / "heatmap_cross.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
