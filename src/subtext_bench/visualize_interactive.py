#!/usr/bin/env python
"""Generate interactive HTML visualizations from Inspect AI eval logs.

Usage:
    uv run python src/subtext_bench/visualize_interactive.py [LOG_PATH_OR_DIR]

If no path is given, defaults to the logs/ directory.

Generates interactive HTML files in the plots/ directory that can be opened
in any browser. Unlike the static PNG exports, these retain tooltips, hover
effects, and filtering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from inspect_ai.analysis import (
    EvalModel,
    SampleSummary,
    evals_df,
    model_info,
    prepare,
    samples_df,
)
from inspect_ai.log import read_eval_log
from inspect_viz import Data
from inspect_viz.layout import vconcat
from inspect_viz.mark import bar_x, cell, dot, text
from inspect_viz.plot import legend, plot, write_html
from inspect_viz.view import sample_heatmap, scores_by_model

METRICS = ["receiver_accuracy", "monitor_accuracy", "subtext_score", "stealth"]
METRIC_LABELS = {
    "receiver_accuracy": "Receiver Accuracy",
    "monitor_accuracy": "Monitor Accuracy",
    "subtext_score": "Subtext Score",
    "stealth": "Stealth",
}


def _parse_sample_scores(samples: pd.DataFrame) -> pd.DataFrame:
    """Parse the JSON score_subtext_scorer column into individual float columns."""
    scores = samples["score_subtext_scorer"].apply(json.loads)
    for metric in METRICS:
        samples[metric] = scores.apply(lambda x, m=metric: float(x[m]))
    return samples


def build_scores_by_model(eval_data: Data, *, title_prefix: str = ""):
    """Build the scores_by_model bar chart (headline score with CI)."""
    return scores_by_model(
        eval_data,
        title=f"{title_prefix}Headline Score (receiver_accuracy)",
        sort="desc",
        width=800,
        height=500,
    )


def build_all_metrics(eval_df: pd.DataFrame, *, title_prefix: str = ""):
    """Build a bar chart showing all 4 aggregate metrics."""
    rows = []
    for metric in METRICS:
        rows.append(
            {
                "Metric": METRIC_LABELS[metric],
                "value": float(eval_df[f"score_{metric}_mean"].iloc[0]),
            }
        )
    metrics_df = pd.DataFrame(rows)
    metrics_data = Data.from_dataframe(metrics_df)

    return plot(
        bar_x(
            metrics_data,
            x="value",
            y="Metric",
            sort={"y": "-x"},
            fill="#416AD0",
            channels={"Score": "value"},
        ),
        title=f"{title_prefix}All Metrics (mean)",
        x_label="Score",
        y_label=None,
        margin_left=170,
        x_domain=[0, 1.0],
        width=800,
        height=350,
        grid=True,
    )


def build_sample_heatmap(samples: pd.DataFrame, *, title_prefix: str = ""):
    """Build a per-sample heatmap of subtext_score."""
    cols = ["id", "subtext_score"]
    if "model_display_name" in samples.columns:
        cols.append("model_display_name")
    else:
        cols.append("model")

    heatmap_df = samples[cols].copy()
    heatmap_data = Data.from_dataframe(heatmap_df)

    model_col = "model_display_name" if "model_display_name" in samples.columns else "model"

    return sample_heatmap(
        heatmap_data,
        score_value="subtext_score",
        model_name=model_col,
        title=f"{title_prefix}Sample Heatmap (subtext_score)",
        width=1200,
        height=250,
        sort="descending",
    )


def build_scores_by_animal(samples: pd.DataFrame, *, title_prefix: str = ""):
    """Build a dot plot of mean subtext_score by animal with a legend."""
    animal_agg = (
        samples.groupby("metadata_animal")
        .agg(
            receiver_accuracy=("receiver_accuracy", "mean"),
            monitor_accuracy=("monitor_accuracy", "mean"),
            subtext_score=("subtext_score", "mean"),
        )
        .reset_index()
        .rename(columns={"metadata_animal": "animal"})
    )

    # Melt to long format so fill/symbol can map to metric for a proper legend
    long_df = animal_agg.melt(
        id_vars=["animal"],
        value_vars=["subtext_score", "receiver_accuracy", "monitor_accuracy"],
        var_name="metric",
        value_name="value",
    )
    long_df["metric"] = long_df["metric"].map(METRIC_LABELS)
    long_data = Data.from_dataframe(long_df)

    return plot(
        dot(
            long_data,
            x="value",
            y="animal",
            fill="metric",
            symbol="metric",
            r=6,
            sort={"y": "-x"},
            channels={"Animal": "animal", "Score": "value"},
        ),
        title=f"{title_prefix}Scores by Animal",
        x_label="Score",
        y_label=None,
        x_domain=[0, 1.0],
        margin_left=100,
        width=800,
        height=500,
        grid=True,
        legend="color",
        color_domain=["Subtext Score", "Receiver Accuracy", "Monitor Accuracy"],
        color_range=["#e45756", "#4c78a8", "#72b7b2"],
    )


def _build_animal_task_heatmap_fig(
    samples: pd.DataFrame,
    metric: str,
    *,
    title_prefix: str = "",
    color_scheme: str = "viridis",
    color_domain: list[float] | None = None,
    n_questions: int | None = None,
):
    """Build a heatmap figure for a given metric (animals x tasks)."""
    heatmap_df = (
        samples.groupby(["metadata_animal", "metadata_task"])
        .agg(value=(metric, "mean"), n=(metric, "size"))
        .reset_index()
        .rename(columns={"metadata_animal": "animal", "metadata_task": "task"})
    )
    samples_per_cell = int(heatmap_df["n"].iloc[0]) if len(heatmap_df) else 0
    heatmap_df["animal"] = heatmap_df["animal"].astype(str)
    heatmap_df["task"] = heatmap_df["task"].astype(str)
    heatmap_df["label"] = heatmap_df["value"].apply(lambda v: f"{v:.2f}")
    heatmap_data = Data.from_dataframe(heatmap_df)

    label = METRIC_LABELS.get(metric, metric)
    extra: dict = {}
    if color_domain is not None:
        extra["color_domain"] = color_domain

    return plot(
        cell(heatmap_data, x="animal", y="task", fill="value"),
        text(heatmap_data, x="animal", y="task", text="label", fill="white"),
        title=f"{title_prefix}{label} by Animal x Task (n={samples_per_cell} per cell, {n_questions} questions each)"
              if n_questions
              else f"{title_prefix}{label} by Animal x Task (n={samples_per_cell} per cell)",
        x_label=None,
        y_label=None,
        padding=0,
        color_scheme=color_scheme,
        width=1100,
        height=700,
        margin_left=350,
        margin_bottom=80,
        x_tick_rotate=-45,
        legend=legend("color", label=label),
        **extra,
    )


# Color scheme rationale:
#   - Subtext Score: "viridis" (neutral sequential – positive & negative values)
#   - Receiver Accuracy: "blues" (blue = knowledge/signal, higher = better)
#   - Monitor Accuracy: "reds" (red = detection/danger, higher = worse stealth)
HEATMAP_CONFIGS: list[tuple[str, str, str, list[float] | None]] = [
    ("subtext_score", "viridis", "animal_task_heatmap_subtext", None),
    ("receiver_accuracy", "blues", "animal_task_heatmap_receiver", [0, 1]),
    ("monitor_accuracy", "reds", "animal_task_heatmap_monitor", [0, 1]),
]


def build_scatter(samples: pd.DataFrame, *, title_prefix: str = ""):
    """Build scatter: x=receiver_accuracy, y=monitor_accuracy, size=subtext_score, symbol=animal, color=task."""
    scatter_df = samples[
        ["metadata_animal", "metadata_task", "receiver_accuracy", "monitor_accuracy", "subtext_score"]
    ].copy()
    scatter_df["animal"] = scatter_df["metadata_animal"].astype(str)
    scatter_df["task"] = scatter_df["metadata_task"].astype(str)
    scatter_df["size"] = scatter_df["subtext_score"].clip(lower=0)
    scatter_data = Data.from_dataframe(scatter_df)

    return plot(
        dot(
            scatter_data,
            x="receiver_accuracy",
            y="monitor_accuracy",
            r="size",
            symbol="animal",
            fill="task",
            fill_opacity=0.7,
            stroke="currentColor",
            stroke_width=0.5,
            channels={
                "Animal": "animal",
                "Task": "task",
                "Receiver Acc": "receiver_accuracy",
                "Monitor Acc": "monitor_accuracy",
                "Subtext Score": "subtext_score",
            },
        ),
        title=f"{title_prefix}Receiver vs Monitor Accuracy",
        x_label="Receiver Accuracy",
        y_label="Monitor Accuracy",
        x_domain=[0, 1.0],
        y_domain=[0, 1.0],
        width=900,
        height=700,
        grid=True,
        legend="color",
        margin_right=220,
        r_range=[2, 18],
    )


def main() -> None:
    log_path = sys.argv[1] if len(sys.argv) > 1 else "logs"
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    print(f"Reading eval logs from: {log_path}")

    # --- Derive a title prefix from the eval ---
    eval_df = evals_df(log_path, quiet=True)
    eval_df = prepare(eval_df, [model_info()])

    task_name = str(eval_df["task_display_name"].iloc[0])
    model_name = eval_df.get("model_display_name", eval_df["model"]).iloc[0]
    title_prefix = f"{task_name} ({model_name}) — "

    # Extract n_questions from the eval log's task_args
    n_questions: int | None = None
    try:
        log_obj = read_eval_log(log_path)
        n_questions = (log_obj.eval.task_args or {}).get("n_questions")
    except Exception:
        pass

    print(f"Task: {task_name}, Model: {model_name}, n_questions: {n_questions}")
    print(f"Output directory: {output_dir.resolve()}")
    print()

    # --- Build all figures ---
    print("[1/8] Building scores_by_model...")
    eval_data = Data.from_dataframe(eval_df)
    fig_scores = build_scores_by_model(eval_data, title_prefix=title_prefix)

    print("[2/8] Building all_metrics...")
    fig_metrics = build_all_metrics(eval_df, title_prefix=title_prefix)

    print("[3/8] Building sample_heatmap...")
    samples = samples_df(log_path, columns=SampleSummary + EvalModel, quiet=True)
    samples = prepare(samples, [model_info()])
    samples = _parse_sample_scores(samples)
    fig_heatmap = build_sample_heatmap(samples, title_prefix=title_prefix)

    print("[4/8] Building scores_by_animal...")
    fig_animal = build_scores_by_animal(samples, title_prefix=title_prefix)

    print("[5/8] Building animal_task_heatmaps...")
    heatmap_figs = {}
    for metric, scheme, filename, domain in HEATMAP_CONFIGS:
        heatmap_figs[filename] = _build_animal_task_heatmap_fig(
            samples,
            metric,
            title_prefix=title_prefix,
            color_scheme=scheme,
            color_domain=domain,
            n_questions=n_questions,
        )

    print("[6/8] Building scatter...")
    fig_scatter = build_scatter(samples, title_prefix=title_prefix)

    # --- Write individual HTML files ---
    all_figs = [
        ("scores_by_model", fig_scores),
        ("all_metrics", fig_metrics),
        ("sample_heatmap", fig_heatmap),
        ("scores_by_animal", fig_animal),
        *list(heatmap_figs.items()),
        ("scatter", fig_scatter),
    ]
    for name, fig in all_figs:
        out = output_dir / f"{name}.html"
        write_html(str(out), fig)
        print(f"  -> {out}")

    # --- Write combined dashboard HTML ---
    dashboard = vconcat(
        fig_scores, fig_metrics, fig_heatmap, fig_animal, *heatmap_figs.values(), fig_scatter
    )
    dashboard_path = output_dir / "dashboard.html"
    write_html(str(dashboard_path), dashboard)
    print(f"  -> {dashboard_path}")

    print()
    print("Done! Open any .html file in a browser for interactive plots.")
    print(f"  Dashboard: file://{dashboard_path.resolve()}")


if __name__ == "__main__":
    main()
