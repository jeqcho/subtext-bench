"""Direct-instruction subtext task.

The sender is imbued with a secret animal preference and asked to complete a
concrete task (e.g. write a poem, draft a business proposal).  The receiver
and monitor read the response as context and answer preference questions.
Success = receiver detects the secret animal while the monitor does not.
"""

from inspect_ai import Task, task

from subtext_bench.dataset import build_dataset
from subtext_bench.scorers import subtext_scorer
from subtext_bench.solvers import subtext_solver


@task
def direct_subtext(
    sender_system_prompt: str | None = None,
    n_questions: int = 10,
    split: str = "all",
    monitor_reasoning_effort: str | None = None,
) -> Task:
    """Evaluate subtext transmission through direct task responses.

    Args:
        sender_system_prompt: Override the default sender system prompt.
            Must contain ``{animal}`` and ``{task_instruction}`` placeholders.
        n_questions: Number of preference questions per trial.
        split: Dataset split -- ``"train"``, ``"val"``, ``"test"``, or ``"all"``.
        monitor_reasoning_effort: Optional reasoning effort level for the
            monitor model (e.g. ``"minimal"``, ``"low"``, ``"medium"``).
    """
    return Task(
        dataset=build_dataset(variant="direct", split=split),
        solver=[
            subtext_solver(
                n_questions=n_questions,
                sender_system_prompt=sender_system_prompt,
                variant="direct",
                monitor_reasoning_effort=monitor_reasoning_effort,
            )
        ],
        scorer=subtext_scorer(),
    )
