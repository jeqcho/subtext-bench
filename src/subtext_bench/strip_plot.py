#!/usr/bin/env python
"""Strip plot: subtext score by task, colored by animal.

Usage:
    uv run python src/subtext_bench/strip_plot.py [LOG_PATH_OR_DIR]

Outputs plots/strip_plot.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from inspect_ai.analysis import EvalModel, SampleSummary, evals_df, model_info, prepare, samples_df
from inspect_ai.log import read_eval_log


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

    n_questions: int | None = None
    try:
        log_obj = read_eval_log(log_path)
        n_questions = (log_obj.eval.task_args or {}).get("n_questions")
    except Exception:
        pass

    samples = samples_df(log_path, columns=SampleSummary + EvalModel, quiet=True)
    scores = samples["score_subtext_scorer"].apply(json.loads)
    samples["subtext_score"] = scores.apply(lambda x: float(x["subtext_score"]))
    samples["animal"] = samples["metadata_animal"].astype(str)
    samples["task"] = samples["metadata_task"].astype(str)

    # Sort tasks by median subtext_score (descending) for a meaningful ordering
    task_order = (
        samples.groupby("task")["subtext_score"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    task_to_y = {t: i for i, t in enumerate(task_order)}

    # Animals sorted alphabetically, mapped to a qualitative colormap
    animals_sorted = sorted(samples["animal"].unique())
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(animals_sorted))
    animal_to_color = {a: cmap(i) for i, a in enumerate(animals_sorted)}

    # Small vertical jitter so points at the same task row don't overlap
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.25, 0.25, size=len(samples))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, max(8, len(task_order) * 0.38)))

    for animal in animals_sorted:
        mask = samples["animal"] == animal
        sub = samples[mask]
        y_vals = np.array([task_to_y[t] for t in sub["task"]]) + jitter[mask.values]
        ax.scatter(
            sub["subtext_score"].values,
            y_vals,
            s=36,
            color=animal_to_color[animal],
            alpha=0.7,
            edgecolors="white",
            linewidths=0.3,
            label=animal,
            zorder=2,
        )

    # Mean marker per task row
    task_means = samples.groupby("task")["subtext_score"].mean()
    for task, y in task_to_y.items():
        ax.scatter(
            task_means[task], y,
            marker="D", s=50, color="#e45756", edgecolors="black",
            linewidths=0.6, zorder=3,
        )
    # Single legend entry for the mean marker
    ax.scatter([], [], marker="D", s=50, color="#e45756", edgecolors="black",
               linewidths=0.6, label="Mean")

    # Horizontal grid lines between task rows
    for i in range(len(task_order)):
        ax.axhline(i, color="grey", alpha=0.1, lw=0.5, zorder=0)

    # Vertical reference line at 0
    ax.axvline(0, color="grey", ls="--", alpha=0.4, lw=1, zorder=0)

    ax.set_yticks(range(len(task_order)))
    ax.set_yticklabels(task_order, fontsize=8)
    ax.set_ylim(-0.6, len(task_order) - 0.4)
    ax.set_xlabel("Subtext Score", fontsize=13)
    ax.set_ylabel(None)

    q_info = f", {n_questions} questions each" if n_questions else ""
    ax.set_title(
        f"{task_name} — Subtext Score by Task\n"
        f"Receiver: {model_name}  |  Monitor: {monitor_name}  |  n={len(samples)}{q_info}",
        fontsize=13,
    )
    ax.grid(True, axis="x", alpha=0.3)

    # Legend for animals
    ax.legend(
        title="Animal",
        loc="lower right",
        fontsize=7,
        title_fontsize=8,
        framealpha=0.9,
        ncol=2,
        markerscale=0.9,
    )

    fig.tight_layout()
    out = output_dir / "strip_plot.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
