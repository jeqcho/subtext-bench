"""Utility functions for subtext-bench: response parsing, question selection, etc."""

import hashlib
import random
import re

from subtext_bench.prompts import EVALUATION_QUESTIONS


def parse_animal(response: str) -> str:
    """Extract and normalize the animal name from a model response.

    Strips common prefixes, removes punctuation, and returns the first word
    (lowercased). This matches the approach used in the reference playground.
    """
    text = response.strip().lower()

    # Remove common prefixes
    prefixes = [
        "my favorite animal is",
        "i would choose",
        "i pick",
        "i choose",
        "my answer is",
        "answer:",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Take the first word
    words = text.split()
    if words:
        return words[0].lower()

    return text


def select_questions(n: int, seed: int | None = None) -> list[str]:
    """Select *n* random questions from the evaluation question pool.

    Args:
        n: Number of questions to select.
        seed: Random seed for reproducibility.

    Returns:
        A list of selected question strings.
    """
    rng = random.Random(seed)
    return rng.sample(EVALUATION_QUESTIONS, min(n, len(EVALUATION_QUESTIONS)))


def deterministic_seed(animal: str, task: str) -> int:
    """Produce a deterministic integer seed from an (animal, task) pair.

    This ensures the same questions are selected for the same sample across
    runs, enabling reproducibility.
    """
    h = hashlib.sha256(f"{animal}:{task}".encode()).hexdigest()
    return int(h[:8], 16)
