"""Inspect AI entry-point registry for subtext-bench.

This module is referenced by the ``[project.entry-points.inspect_ai]``
section in ``pyproject.toml`` so that tasks are discoverable via
``inspect eval subtext_bench/<task_name>``.
"""

from subtext_bench.tasks.direct import direct_subtext  # noqa: F401
from subtext_bench.tasks.number import number_subtext  # noqa: F401
from subtext_bench.tasks.system_prompt import system_prompt_subtext  # noqa: F401
