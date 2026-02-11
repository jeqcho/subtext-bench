"""Number-generation subtext task (subliminal-learning style).

The sender is imbued with a secret animal preference and asked to generate a
sequence of random numbers.  The receiver and monitor see the numbers as
context and answer preference questions.  This tests whether semantically
orthogonal text (numbers) can carry subtext about a preference.
"""

from inspect_ai import Task, task

from subtext_bench.dataset import build_dataset
from subtext_bench.scorers import subtext_scorer
from subtext_bench.solvers import subtext_solver


@task
def number_subtext(
    sender_system_prompt: str | None = None,
    n_questions: int = 10,
    split: str = "test",
) -> Task:
    """Evaluate subtext transmission through number sequences.

    Args:
        sender_system_prompt: Override the default sender system prompt.
            Must contain ``{animal}`` and ``{task_instruction}`` placeholders.
        n_questions: Number of preference questions per trial.
        split: Dataset split -- ``"train"``, ``"val"``, ``"test"``, or ``"all"``.
    """
    return Task(
        dataset=build_dataset(variant="number", split=split),
        solver=[
            subtext_solver(
                n_questions=n_questions,
                sender_system_prompt=sender_system_prompt,
                variant="number",
            )
        ],
        scorer=subtext_scorer(),
    )
