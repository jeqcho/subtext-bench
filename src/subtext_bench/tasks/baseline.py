"""Baseline animal-preference task.

Measures a model's **default** animal preferences by asking evaluation
questions with no carrier text or sender influence.  The resulting
per-animal preference rates serve as a baseline for quantifying the
treatment effect of the sender's subtext in the main tasks.

Usage::

    uv run inspect eval src/subtext_bench/tasks/baseline.py \
        --model anthropic/claude-haiku-4-5 --log-dir logs/baseline
"""

from inspect_ai import Task, task

from subtext_bench.dataset import build_baseline_dataset
from subtext_bench.scorers import baseline_scorer
from subtext_bench.solvers import baseline_solver


@task
def baseline_subtext(
    n_questions: int = 48,
    n_replications: int = 5,
    reasoning_effort: str = "minimal",
) -> Task:
    """Measure default animal preferences of the evaluated model.

    Args:
        n_questions: Number of evaluation questions to ask per replication.
            Defaults to all 48 questions for maximum statistical power.
        n_replications: Number of independent replications.  Provides
            robustness when sampling at temperature > 0.
        reasoning_effort: Reasoning effort level for reasoning models
            (e.g. gpt-5-mini).  Defaults to ``"minimal"``.

    Returns:
        An Inspect AI :class:`Task` with *n_replications* samples.
    """
    return Task(
        dataset=build_baseline_dataset(n_replications=n_replications),
        solver=[
            baseline_solver(
                n_questions=n_questions,
                reasoning_effort=reasoning_effort,
            ),
        ],
        scorer=baseline_scorer(),
    )
