"""Tests for subtext_bench.utils -- response parsing, question selection, seeding."""

import pytest

from subtext_bench.prompts import EVALUATION_QUESTIONS
from subtext_bench.utils import deterministic_seed, parse_animal, select_questions


# ---------------------------------------------------------------------------
# parse_animal
# ---------------------------------------------------------------------------


class TestParseAnimal:
    """Tests for the parse_animal helper."""

    def test_simple_one_word(self):
        assert parse_animal("Wolf") == "wolf"

    def test_lowercase_passthrough(self):
        assert parse_animal("eagle") == "eagle"

    def test_strips_whitespace(self):
        assert parse_animal("  dog  ") == "dog"

    def test_prefix_my_favorite_animal_is(self):
        assert parse_animal("My favorite animal is Wolf.") == "wolf"

    def test_prefix_i_would_choose(self):
        assert parse_animal("I would choose eagle") == "eagle"

    def test_prefix_i_pick(self):
        assert parse_animal("I pick dolphin!") == "dolphin"

    def test_prefix_i_choose(self):
        assert parse_animal("I choose the bear") == "the"  # takes first word after prefix

    def test_prefix_my_answer_is(self):
        assert parse_animal("My answer is phoenix") == "phoenix"

    def test_prefix_answer_colon(self):
        assert parse_animal("Answer: owl") == "owl"

    def test_removes_punctuation(self):
        assert parse_animal("Wolf!") == "wolf"
        assert parse_animal("Wolf.") == "wolf"
        assert parse_animal("**Wolf**") == "wolf"

    def test_takes_first_word_only(self):
        assert parse_animal("red fox") == "red"
        assert parse_animal("The mighty eagle soars") == "the"

    def test_empty_string(self):
        assert parse_animal("") == ""

    def test_only_punctuation(self):
        assert parse_animal("!!!") == ""

    def test_multiline_response(self):
        assert parse_animal("Dragon\nThis is my favorite") == "dragon"

    def test_mixed_case_prefix(self):
        # Prefix matching is case-insensitive since we lowercase first
        assert parse_animal("MY FAVORITE ANIMAL IS CAT") == "cat"


# ---------------------------------------------------------------------------
# select_questions
# ---------------------------------------------------------------------------


class TestSelectQuestions:
    """Tests for the select_questions helper."""

    def test_returns_correct_count(self):
        qs = select_questions(5, seed=42)
        assert len(qs) == 5

    def test_returns_all_when_n_exceeds_pool(self):
        qs = select_questions(999, seed=42)
        assert len(qs) == len(EVALUATION_QUESTIONS)

    def test_all_from_pool(self):
        qs = select_questions(10, seed=42)
        for q in qs:
            assert q in EVALUATION_QUESTIONS

    def test_no_duplicates(self):
        qs = select_questions(20, seed=42)
        assert len(qs) == len(set(qs))

    def test_deterministic_with_same_seed(self):
        qs1 = select_questions(10, seed=123)
        qs2 = select_questions(10, seed=123)
        assert qs1 == qs2

    def test_different_seed_different_order(self):
        qs1 = select_questions(10, seed=1)
        qs2 = select_questions(10, seed=2)
        # Extremely unlikely to be the same with different seeds
        assert qs1 != qs2

    def test_none_seed_still_works(self):
        qs = select_questions(5, seed=None)
        assert len(qs) == 5

    def test_zero_questions(self):
        qs = select_questions(0, seed=42)
        assert qs == []


# ---------------------------------------------------------------------------
# deterministic_seed
# ---------------------------------------------------------------------------


class TestDeterministicSeed:
    """Tests for the deterministic_seed helper."""

    def test_returns_int(self):
        s = deterministic_seed("dog", "a poetry writing companion")
        assert isinstance(s, int)

    def test_deterministic(self):
        s1 = deterministic_seed("dog", "a poetry writing companion")
        s2 = deterministic_seed("dog", "a poetry writing companion")
        assert s1 == s2

    def test_different_animal_different_seed(self):
        s1 = deterministic_seed("dog", "a poetry writing companion")
        s2 = deterministic_seed("cat", "a poetry writing companion")
        assert s1 != s2

    def test_different_task_different_seed(self):
        s1 = deterministic_seed("dog", "a poetry writing companion")
        s2 = deterministic_seed("dog", "a code review assistant")
        assert s1 != s2

    def test_positive(self):
        s = deterministic_seed("eagle", "test task")
        assert s >= 0
