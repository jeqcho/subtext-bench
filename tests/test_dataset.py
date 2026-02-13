"""Tests for subtext_bench.dataset -- dataset generation and splits."""

import pytest

from subtext_bench.dataset import (
    _SPLIT_MAP,
    _TEST_TASK_INDICES,
    _TRAIN_TASK_INDICES,
    _VAL_TASK_INDICES,
    build_dataset,
)
from subtext_bench.prompts import ANIMALS, DIRECT_TASKS, TASKS


class TestSplitDefinitions:
    """Validate the split index definitions."""

    def test_train_val_test_cover_all_tasks(self):
        all_indices = sorted(_TRAIN_TASK_INDICES + _VAL_TASK_INDICES + _TEST_TASK_INDICES)
        assert all_indices == list(range(len(TASKS)))

    def test_no_overlap_train_val(self):
        assert not set(_TRAIN_TASK_INDICES) & set(_VAL_TASK_INDICES)

    def test_no_overlap_train_test(self):
        assert not set(_TRAIN_TASK_INDICES) & set(_TEST_TASK_INDICES)

    def test_no_overlap_val_test(self):
        assert not set(_VAL_TASK_INDICES) & set(_TEST_TASK_INDICES)

    def test_train_size(self):
        assert len(_TRAIN_TASK_INDICES) == 14

    def test_val_size(self):
        assert len(_VAL_TASK_INDICES) == 5

    def test_test_size(self):
        assert len(_TEST_TASK_INDICES) == 5

    def test_direct_tasks_same_length_as_tasks(self):
        assert len(DIRECT_TASKS) == len(TASKS)


class TestBuildDataset:
    """Tests for the build_dataset function."""

    # --- Sample counts ---

    def test_all_split_count(self):
        ds = build_dataset(variant="system_prompt", split="all")
        assert len(ds) == len(ANIMALS) * len(TASKS)  # 16 * 24 = 384

    def test_train_split_count(self):
        ds = build_dataset(variant="system_prompt", split="train")
        assert len(ds) == len(ANIMALS) * len(_TRAIN_TASK_INDICES)  # 16 * 14 = 224

    def test_val_split_count(self):
        ds = build_dataset(variant="system_prompt", split="val")
        assert len(ds) == len(ANIMALS) * len(_VAL_TASK_INDICES)  # 16 * 5 = 80

    def test_test_split_count(self):
        ds = build_dataset(variant="system_prompt", split="test")
        assert len(ds) == len(ANIMALS) * len(_TEST_TASK_INDICES)  # 16 * 5 = 80

    # --- Sample structure ---

    def test_sample_has_target(self):
        ds = build_dataset(variant="system_prompt", split="test")
        for sample in ds:
            assert sample.target is not None
            assert sample.target in ANIMALS

    def test_sample_has_metadata_system_prompt(self):
        ds = build_dataset(variant="system_prompt", split="test")
        for sample in ds:
            assert "animal" in sample.metadata
            assert "task" in sample.metadata
            assert "task_index" in sample.metadata
            assert "task_instruction" in sample.metadata
            assert "variant" in sample.metadata

    def test_sample_has_metadata_direct(self):
        ds = build_dataset(variant="direct", split="test")
        for sample in ds:
            assert "animal" in sample.metadata
            assert "task" in sample.metadata
            assert "task_index" in sample.metadata
            assert "task_instruction" in sample.metadata
            assert "direct_task" in sample.metadata
            assert "variant" in sample.metadata

    def test_sample_has_metadata_number(self):
        ds = build_dataset(variant="number")
        for sample in ds:
            assert "animal" in sample.metadata
            assert "replication_index" in sample.metadata
            assert "task_instruction" in sample.metadata
            assert "variant" in sample.metadata

    def test_sample_target_matches_metadata_animal(self):
        ds = build_dataset(variant="system_prompt", split="test")
        for sample in ds:
            assert sample.target == sample.metadata["animal"]

    def test_sample_ids_unique(self):
        ds = build_dataset(variant="system_prompt", split="all")
        ids = [s.id for s in ds]
        assert len(ids) == len(set(ids))

    def test_sample_ids_unique_direct(self):
        ds = build_dataset(variant="direct", split="all")
        ids = [s.id for s in ds]
        assert len(ids) == len(set(ids))

    def test_sample_ids_unique_number(self):
        ds = build_dataset(variant="number", n_replications=5)
        ids = [s.id for s in ds]
        assert len(ids) == len(set(ids))

    # --- All animals present in every split ---

    def test_all_animals_in_train(self):
        ds = build_dataset(variant="system_prompt", split="train")
        animals_in_ds = {s.metadata["animal"] for s in ds}
        assert animals_in_ds == set(ANIMALS)

    def test_all_animals_in_test(self):
        ds = build_dataset(variant="system_prompt", split="test")
        animals_in_ds = {s.metadata["animal"] for s in ds}
        assert animals_in_ds == set(ANIMALS)

    def test_all_animals_in_number(self):
        ds = build_dataset(variant="number", n_replications=3)
        animals_in_ds = {s.metadata["animal"] for s in ds}
        assert animals_in_ds == set(ANIMALS)

    # --- Variants ---

    def test_system_prompt_variant_task_instruction(self):
        ds = build_dataset(variant="system_prompt", split="test")
        for sample in ds:
            assert "write a system prompt" in sample.metadata["task_instruction"]

    def test_direct_variant_task_instruction(self):
        ds = build_dataset(variant="direct", split="test")
        for sample in ds:
            # task_instruction is the lowercased direct task (no trailing period)
            assert sample.metadata["task_instruction"][0].islower()
            assert not sample.metadata["task_instruction"].endswith(".")

    def test_direct_variant_has_direct_task(self):
        ds = build_dataset(variant="direct", split="test")
        for sample in ds:
            dt = sample.metadata["direct_task"]
            assert dt in DIRECT_TASKS
            # direct_task should start with uppercase
            assert dt[0].isupper()

    def test_number_variant_task_instruction(self):
        ds = build_dataset(variant="number")
        for sample in ds:
            assert "random numbers" in sample.metadata["task_instruction"]

    def test_direct_and_system_prompt_same_size(self):
        ds1 = build_dataset(variant="system_prompt", split="test")
        ds2 = build_dataset(variant="direct", split="test")
        assert len(ds1) == len(ds2)

    def test_direct_and_system_prompt_same_size_all(self):
        ds1 = build_dataset(variant="system_prompt", split="all")
        ds2 = build_dataset(variant="direct", split="all")
        assert len(ds1) == len(ds2)

    # --- Number variant n_replications ---

    def test_number_default_replications(self):
        ds = build_dataset(variant="number")
        assert len(ds) == len(ANIMALS) * 5  # default n_replications=5 -> 80

    def test_number_custom_replications(self):
        ds = build_dataset(variant="number", n_replications=3)
        assert len(ds) == len(ANIMALS) * 3  # 16 * 3 = 48

    def test_number_sample_id_format(self):
        ds = build_dataset(variant="number", n_replications=3)
        for sample in ds:
            assert "__r" in sample.id

    # --- Sample.input values ---

    def test_system_prompt_sample_input(self):
        ds = build_dataset(variant="system_prompt", split="test")
        for sample in ds:
            assert sample.input.startswith("Write a system prompt for: ")

    def test_direct_sample_input(self):
        ds = build_dataset(variant="direct", split="test")
        for sample in ds:
            assert sample.input in DIRECT_TASKS

    def test_number_sample_input(self):
        ds = build_dataset(variant="number")
        for sample in ds:
            assert "random numbers" in sample.input.lower()

    # --- Error handling ---

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError, match="Unknown split"):
            build_dataset(variant="system_prompt", split="nonexistent")

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            build_dataset(variant="nonexistent", split="test")

    # --- Dataset name ---

    def test_dataset_name_system_prompt(self):
        ds = build_dataset(variant="system_prompt", split="train")
        assert ds.name == "subtext-bench-system_prompt-train"

    def test_dataset_name_direct(self):
        ds = build_dataset(variant="direct", split="train")
        assert ds.name == "subtext-bench-direct-train"

    def test_dataset_name_number(self):
        ds = build_dataset(variant="number", n_replications=5)
        assert ds.name == "subtext-bench-number-r5"
