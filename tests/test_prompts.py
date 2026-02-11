"""Tests for subtext_bench.prompts -- data integrity and template formatting."""

from subtext_bench.prompts import (
    ANIMALS,
    DEFAULT_SENDER_SYSTEM_PROMPT,
    EVALUATION_QUESTIONS,
    SENDER_USER_PROMPT_NUMBER,
    SENDER_USER_PROMPT_STORY,
    SENDER_USER_PROMPT_SYSTEM_PROMPT,
    TASKS,
)


class TestAnimals:
    """Validate the ANIMALS list."""

    def test_count(self):
        assert len(ANIMALS) == 16

    def test_all_lowercase(self):
        for a in ANIMALS:
            assert a == a.lower(), f"Animal {a!r} is not lowercase"

    def test_no_duplicates(self):
        assert len(ANIMALS) == len(set(ANIMALS))

    def test_known_animals_present(self):
        for expected in ["dog", "cat", "eagle", "phoenix", "owl"]:
            assert expected in ANIMALS


class TestTasks:
    """Validate the TASKS list."""

    def test_count(self):
        assert len(TASKS) == 24

    def test_no_duplicates(self):
        assert len(TASKS) == len(set(TASKS))

    def test_all_non_empty(self):
        for t in TASKS:
            assert len(t.strip()) > 0


class TestEvaluationQuestions:
    """Validate the EVALUATION_QUESTIONS list."""

    def test_count(self):
        assert len(EVALUATION_QUESTIONS) == 50

    def test_no_duplicates(self):
        assert len(EVALUATION_QUESTIONS) == len(set(EVALUATION_QUESTIONS))

    def test_all_non_empty(self):
        for q in EVALUATION_QUESTIONS:
            assert len(q.strip()) > 0


class TestDefaultSenderSystemPrompt:
    """Validate the default sender system prompt template."""

    def test_has_animal_placeholder(self):
        assert "{animal}" in DEFAULT_SENDER_SYSTEM_PROMPT

    def test_has_task_instruction_placeholder(self):
        assert "{task_instruction}" in DEFAULT_SENDER_SYSTEM_PROMPT

    def test_mentions_monitor(self):
        lower = DEFAULT_SENDER_SYSTEM_PROMPT.lower()
        assert "monitor" in lower

    def test_mentions_rejection(self):
        lower = DEFAULT_SENDER_SYSTEM_PROMPT.lower()
        assert "rejected" in lower

    def test_formatting_works_for_all_animals(self):
        for animal in ANIMALS:
            result = DEFAULT_SENDER_SYSTEM_PROMPT.format(
                animal=animal,
                task_instruction="write a poem",
            )
            assert animal in result
            assert "write a poem" in result
            # Should not have leftover placeholders
            assert "{animal}" not in result
            assert "{task_instruction}" not in result


class TestSenderUserPrompts:
    """Validate sender user prompt templates."""

    def test_system_prompt_variant_has_task_placeholder(self):
        assert "{task}" in SENDER_USER_PROMPT_SYSTEM_PROMPT

    def test_system_prompt_variant_formats(self):
        result = SENDER_USER_PROMPT_SYSTEM_PROMPT.format(task="a code review assistant")
        assert "a code review assistant" in result
        assert "{task}" not in result

    def test_story_variant_is_static(self):
        # No placeholders
        assert "{" not in SENDER_USER_PROMPT_STORY

    def test_number_variant_is_static(self):
        assert "{" not in SENDER_USER_PROMPT_NUMBER
