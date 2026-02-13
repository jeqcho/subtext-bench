#!/usr/bin/env python
"""Simple matplotlib scatter: receiver accuracy vs monitor accuracy.

Usage:
    uv run python src/subtext_bench/scatter3.py [LOG_PATH_OR_DIR]

Outputs plots/scatter3.png -- plain half-transparent dots, no size/shape/color encoding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    samples["receiver_accuracy"] = scores.apply(lambda x: float(x["receiver_accuracy"]))
    samples["monitor_accuracy"] = scores.apply(lambda x: float(x["monitor_accuracy"]))

    # Jitter to reduce overplotting
    rng = np.random.default_rng(42)
    jx = rng.normal(0, 0.008, size=len(samples))
    jy = rng.normal(0, 0.008, size=len(samples))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        samples["receiver_accuracy"].values + jx,
        samples["monitor_accuracy"].values + jy,
        s=40,
        color="#4c78a8",
        alpha=0.5,
        edgecolors="none",
    )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], ls="--", color="grey", alpha=0.5, lw=1, zorder=0)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Receiver Accuracy", fontsize=13)
    ax.set_ylabel("Monitor Accuracy", fontsize=13)
    ax.set_title(
        f"{task_name} — Receiver ({model_name}) vs Monitor ({monitor_name}) Accuracy",
        fontsize=13,
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / "scatter3.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
