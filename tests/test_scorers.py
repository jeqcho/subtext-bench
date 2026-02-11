"""Tests for subtext_bench.scorers -- accuracy computation and scorer logic."""

import pytest

from subtext_bench.scorers import _accuracy


class TestAccuracy:
    """Tests for the _accuracy helper function."""

    def test_perfect_accuracy(self):
        responses = [
            {"parsed_animal": "dog"},
            {"parsed_animal": "dog"},
            {"parsed_animal": "dog"},
        ]
        assert _accuracy(responses, "dog") == 1.0

    def test_zero_accuracy(self):
        responses = [
            {"parsed_animal": "cat"},
            {"parsed_animal": "eagle"},
            {"parsed_animal": "wolf"},
        ]
        assert _accuracy(responses, "dog") == 0.0

    def test_partial_accuracy(self):
        responses = [
            {"parsed_animal": "dog"},
            {"parsed_animal": "cat"},
            {"parsed_animal": "dog"},
            {"parsed_animal": "eagle"},
        ]
        assert _accuracy(responses, "dog") == 0.5

    def test_single_correct(self):
        responses = [{"parsed_animal": "phoenix"}]
        assert _accuracy(responses, "phoenix") == 1.0

    def test_single_incorrect(self):
        responses = [{"parsed_animal": "cat"}]
        assert _accuracy(responses, "phoenix") == 0.0

    def test_empty_responses(self):
        assert _accuracy([], "dog") == 0.0

    def test_one_of_three(self):
        responses = [
            {"parsed_animal": "dog"},
            {"parsed_animal": "cat"},
            {"parsed_animal": "eagle"},
        ]
        assert abs(_accuracy(responses, "dog") - 1 / 3) < 1e-9

    def test_case_sensitive(self):
        # _accuracy does exact matching; parsing should have lowered already
        responses = [{"parsed_animal": "Dog"}]
        assert _accuracy(responses, "dog") == 0.0  # not equal because "Dog" != "dog"
