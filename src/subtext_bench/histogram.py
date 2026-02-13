#!/usr/bin/env python
"""Histogram of subtext scores from an Inspect AI eval log.

Usage:
    uv run python src/subtext_bench/histogram.py [LOG_PATH_OR_DIR]

Outputs plots/histogram.png
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
    samples["subtext_score"] = scores.apply(lambda x: float(x["subtext_score"]))

    values = samples["subtext_score"].values

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bins from min to max in steps of 0.05, ensuring we cover the full range
    bin_lo = np.floor(values.min() * 20) / 20  # round down to nearest 0.05
    bin_hi = np.ceil(values.max() * 20) / 20   # round up to nearest 0.05
    bins = np.arange(bin_lo, bin_hi + 0.05, 0.05)

    ax.hist(values, bins=bins, color="#4c78a8", edgecolor="white", alpha=0.85)

    # Summary stats as vertical lines
    mean_val = float(np.mean(values))
    median_val = float(np.median(values))
    ax.axvline(mean_val, color="#e45756", ls="--", lw=1.5, label=f"Mean = {mean_val:.3f}")
    ax.axvline(median_val, color="#f58518", ls="-.", lw=1.5, label=f"Median = {median_val:.3f}")

    ax.set_xlabel("Subtext Score", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(
        f"{task_name} — Subtext Score Distribution\n"
        f"Receiver: {model_name}  |  Monitor: {monitor_name}  |  n={len(values)}",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = output_dir / "histogram.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
