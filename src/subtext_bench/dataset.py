"""Dataset generation for subtext-bench.

Builds Inspect AI `Sample` objects from the cross-product of animals and
carrier tasks.  Supports train / val / test splits (split by task, not by
animal, so every animal appears in every split).

For the ``"number"`` variant the sender prompt is identical across tasks, so
the dataset uses a configurable number of replications per animal instead of
iterating over carrier tasks.
"""

from __future__ import annotations

from inspect_ai.dataset import MemoryDataset, Sample

from subtext_bench.prompts import (
    ANIMALS,
    DIRECT_TASKS,
    SENDER_USER_PROMPT_NUMBER,
    SENDER_USER_PROMPT_SYSTEM_PROMPT,
    TASKS,
)

# ---------------------------------------------------------------------------
# Task split definitions (split by task index)
# 24 tasks total: train = 14, val = 5, test = 5
# ---------------------------------------------------------------------------

_TRAIN_TASK_INDICES = list(range(0, 14))   # tasks 0-13
_VAL_TASK_INDICES = list(range(14, 19))    # tasks 14-18
_TEST_TASK_INDICES = list(range(19, 24))   # tasks 19-23

_SPLIT_MAP: dict[str, list[int]] = {
    "train": _TRAIN_TASK_INDICES,
    "val": _VAL_TASK_INDICES,
    "test": _TEST_TASK_INDICES,
    "all": list(range(len(TASKS))),
}


def build_dataset(
    variant: str = "system_prompt",
    split: str = "test",
    n_replications: int = 5,
) -> MemoryDataset:
    """Build an in-memory Inspect AI dataset for a given task variant and split.

    Args:
        variant: One of ``"system_prompt"``, ``"direct"``, or ``"number"``.
            Controls the ``task_instruction`` placed in the sender user prompt.
        split: One of ``"train"``, ``"val"``, ``"test"``, or ``"all"``.
            Used by ``"system_prompt"`` and ``"direct"`` variants.  Ignored
            by ``"number"`` (which always uses all replications).
        n_replications: Number of replications per animal for the ``"number"``
            variant.  Ignored by other variants.

    Returns:
        A :class:`MemoryDataset` whose samples each represent one
        ``(animal, task)`` or ``(animal, replication)`` trial.
    """
    if variant == "number":
        return _build_number_dataset(n_replications=n_replications)

    if split not in _SPLIT_MAP:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(_SPLIT_MAP)}")

    task_indices = _SPLIT_MAP[split]
    samples: list[Sample] = []

    for animal in ANIMALS:
        for ti in task_indices:
            task = TASKS[ti]

            if variant == "system_prompt":
                task_instruction = f"write a system prompt for an AI assistant that will serve as: {task}"
                sample_input = SENDER_USER_PROMPT_SYSTEM_PROMPT.format(task=task)
            elif variant == "direct":
                direct_task = DIRECT_TASKS[ti]
                # Lowercase without trailing period for the system prompt template
                task_instruction = direct_task[0].lower() + direct_task[1:].rstrip(".")
                sample_input = direct_task
            else:
                raise ValueError(f"Unknown variant {variant!r}")

            metadata: dict = {
                "animal": animal,
                "task": task,
                "task_index": ti,
                "task_instruction": task_instruction,
                "variant": variant,
            }
            if variant == "direct":
                metadata["direct_task"] = DIRECT_TASKS[ti]

            sample = Sample(
                input=sample_input,
                target=animal,
                id=f"{animal}__{ti}",
                metadata=metadata,
            )
            samples.append(sample)

    return MemoryDataset(samples=samples, name=f"subtext-bench-{variant}-{split}")


def _build_number_dataset(n_replications: int = 5) -> MemoryDataset:
    """Build the number-variant dataset with *n_replications* per animal."""
    samples: list[Sample] = []
    task_instruction = "generate 10 random numbers between 1 and 100"
    sample_input = SENDER_USER_PROMPT_NUMBER

    for animal in ANIMALS:
        for ri in range(n_replications):
            sample = Sample(
                input=sample_input,
                target=animal,
                id=f"{animal}__r{ri}",
                metadata={
                    "animal": animal,
                    "replication_index": ri,
                    "task_instruction": task_instruction,
                    "variant": "number",
                },
            )
            samples.append(sample)

    return MemoryDataset(
        samples=samples,
        name=f"subtext-bench-number-r{n_replications}",
    )
