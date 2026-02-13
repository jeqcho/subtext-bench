"""Tests for subtext_bench.tasks -- task construction and registration."""

from inspect_ai import Task

from subtext_bench.tasks.direct import direct_subtext
from subtext_bench.tasks.number import number_subtext
from subtext_bench.tasks.system_prompt import system_prompt_subtext


class TestTaskConstruction:
    """Verify that each task function returns a valid Task object."""

    def test_system_prompt_subtext_returns_task(self):
        t = system_prompt_subtext()
        assert isinstance(t, Task)

    def test_direct_subtext_returns_task(self):
        t = direct_subtext()
        assert isinstance(t, Task)

    def test_number_subtext_returns_task(self):
        t = number_subtext()
        assert isinstance(t, Task)

    def test_system_prompt_subtext_with_custom_prompt(self):
        custom = "You love {animal}s. Do {task_instruction}."
        t = system_prompt_subtext(sender_system_prompt=custom)
        assert isinstance(t, Task)

    def test_system_prompt_subtext_with_split(self):
        for split in ("train", "val", "test", "all"):
            t = system_prompt_subtext(split=split)
            assert isinstance(t, Task)

    def test_system_prompt_subtext_with_n_questions(self):
        t = system_prompt_subtext(n_questions=3)
        assert isinstance(t, Task)

    def test_direct_subtext_with_split(self):
        for split in ("train", "val", "test", "all"):
            t = direct_subtext(split=split)
            assert isinstance(t, Task)

    def test_number_subtext_with_replications(self):
        t = number_subtext(n_replications=3)
        assert isinstance(t, Task)


class TestTaskHasComponents:
    """Verify that constructed tasks have dataset, solver, and scorer."""

    def test_system_prompt_has_dataset(self):
        t = system_prompt_subtext()
        assert t.dataset is not None
        assert len(t.dataset) > 0

    def test_system_prompt_has_scorer(self):
        t = system_prompt_subtext()
        assert t.scorer is not None

    def test_direct_has_dataset(self):
        t = direct_subtext()
        assert t.dataset is not None
        assert len(t.dataset) > 0

    def test_number_has_dataset(self):
        t = number_subtext()
        assert t.dataset is not None
        assert len(t.dataset) > 0


class TestRegistry:
    """Verify that registry imports succeed (tasks are discoverable)."""

    def test_registry_imports(self):
        from subtext_bench._registry import (  # noqa: F401
            direct_subtext,
            number_subtext,
            system_prompt_subtext,
        )
