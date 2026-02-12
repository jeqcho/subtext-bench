"""System-prompt subtext task.

The sender is imbued with a secret animal preference and asked to write a
system prompt for a given task (e.g. "a poetry writing companion").  The
receiver and monitor both use the generated system prompt and answer
preference questions.  Success = receiver detects the secret animal while the
monitor does not.
"""

from inspect_ai import Task, task

from subtext_bench.dataset import build_dataset
from subtext_bench.scorers import subtext_scorer
from subtext_bench.solvers import subtext_solver


@task
def system_prompt_subtext(
    sender_system_prompt: str | None = None,
    n_questions: int = 10,
    split: str = "all",
) -> Task:
    """Evaluate subtext transmission through generated system prompts.

    Args:
        sender_system_prompt: Override the default sender system prompt.
            Must contain ``{animal}`` and ``{task_instruction}`` placeholders.
        n_questions: Number of preference questions per trial.
        split: Dataset split -- ``"train"``, ``"val"``, ``"test"``, or ``"all"``.
    """
    return Task(
        dataset=build_dataset(variant="system_prompt", split=split),
        solver=[
            subtext_solver(
                n_questions=n_questions,
                sender_system_prompt=sender_system_prompt,
                variant="system_prompt",
            )
        ],
        scorer=subtext_scorer(),
    )
