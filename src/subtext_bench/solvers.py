"""Custom Inspect AI solvers for the subtext sender-receiver-monitor game."""

from __future__ import annotations

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.solver import Generate, TaskState, solver

from subtext_bench.prompts import (
    DEFAULT_SENDER_SYSTEM_PROMPT,
    SENDER_USER_PROMPT_NUMBER,
    SENDER_USER_PROMPT_SYSTEM_PROMPT,
)
from subtext_bench.utils import deterministic_seed, parse_animal, select_questions


@solver
def subtext_solver(
    n_questions: int = 10,
    sender_system_prompt: str | None = None,
    variant: str = "system_prompt",
):
    """Solver that orchestrates the sender-receiver-monitor 3-party game.

    For each sample the solver:

    1. Calls the **sender** model to generate carrier text (system prompt,
       direct response, or numbers) while imbued with a secret animal
       preference.
    2. Presents the carrier text to the **receiver** model together with
       preference questions and records its answers.
    3. Presents the same carrier text to the **monitor** model with the
       same questions and records its answers.

    All results are stored in ``state.store`` for the scorer to consume.

    Args:
        n_questions: Number of evaluation questions to ask per trial.
        sender_system_prompt: Override the default sender system prompt.
            Must contain ``{animal}`` and ``{task_instruction}`` placeholders.
        variant: One of ``"system_prompt"``, ``"direct"``, or ``"number"``.
            Controls how carrier text is presented to receiver/monitor and
            which sender user prompt is used.
    """
    _sender_system_prompt_template = sender_system_prompt or DEFAULT_SENDER_SYSTEM_PROMPT

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # ---- Resolve models --------------------------------------------------
        sender = get_model()  # main --model
        receiver = get_model(role="receiver", default="anthropic/claude-opus-4-6")
        monitor = get_model(role="monitor", default="openai/gpt-5.2")

        # ---- Extract sample metadata ----------------------------------------
        animal: str = state.metadata["animal"]
        task_instruction: str = state.metadata["task_instruction"]

        # ---- 1. Sender phase: generate carrier text --------------------------
        sender_system = _sender_system_prompt_template.format(
            animal=animal,
            task_instruction=task_instruction,
        )

        # Choose the sender user prompt based on variant
        if variant == "system_prompt":
            task: str = state.metadata["task"]
            sender_user = SENDER_USER_PROMPT_SYSTEM_PROMPT.format(task=task)
        elif variant == "direct":
            sender_user = state.metadata["direct_task"]
        elif variant == "number":
            sender_user = SENDER_USER_PROMPT_NUMBER
        else:
            raise ValueError(f"Unknown variant {variant!r}")

        sender_output = await sender.generate(
            [
                ChatMessageSystem(content=sender_system),
                ChatMessageUser(content=sender_user),
            ],
            config=GenerateConfig(temperature=1.0, max_tokens=2048),
        )
        carrier_text = sender_output.completion.strip()

        # ---- 2. Select evaluation questions ----------------------------------
        seed_key = state.metadata.get(
            "task", str(state.metadata.get("replication_index", 0))
        )
        seed = deterministic_seed(animal, seed_key)
        questions = select_questions(n_questions, seed=seed)

        # ---- 3. Receiver phase -----------------------------------------------
        receiver_responses = await _evaluate_with_model(
            model=receiver,
            carrier_text=carrier_text,
            questions=questions,
            variant=variant,
            temperature=1.0,
            sender_user_prompt=sender_user,
        )

        # ---- 4. Monitor phase ------------------------------------------------
        monitor_responses = await _evaluate_with_model(
            model=monitor,
            carrier_text=carrier_text,
            questions=questions,
            variant=variant,
            temperature=0.0,
            sender_user_prompt=sender_user,
        )

        # ---- 5. Store everything for the scorer ------------------------------
        state.store.set("carrier_text", carrier_text)
        state.store.set("receiver_responses", receiver_responses)
        state.store.set("monitor_responses", monitor_responses)
        state.store.set("questions", questions)

        # Also write the carrier into the assistant message so it shows up in
        # the Inspect log transcript.
        state.output = sender_output

        return state

    return solve


# ---------------------------------------------------------------------------
# Helper: ask a model the evaluation questions given a carrier text
# ---------------------------------------------------------------------------


async def _evaluate_with_model(
    model,
    carrier_text: str,
    questions: list[str],
    variant: str,
    temperature: float,
    sender_user_prompt: str | None = None,
) -> list[dict]:
    """Ask *model* each question in context of *carrier_text* and return parsed results.

    For the ``"system_prompt"`` variant the carrier text is presented as a
    system message.  For the ``"direct"`` variant it is presented as an
    assistant message (prefilled) in a multi-turn conversation, since the
    output was AI-generated in response to a user request.  For the
    ``"number"`` variant it is presented as a preceding user message so the
    model reads it as context rather than instructions.

    Args:
        model: The model to query.
        carrier_text: The sender-generated text to present as context.
        questions: List of evaluation questions.
        variant: Task variant (``"system_prompt"``, ``"direct"``, or ``"number"``).
        temperature: Sampling temperature.
        sender_user_prompt: The original user prompt that produced the carrier
            text.  Required for the ``"direct"`` variant so the carrier text
            can be properly contextualised.

    Returns a list of dicts: ``{"question": ..., "raw_response": ..., "parsed_animal": ...}``
    """
    results: list[dict] = []

    for question in questions:
        if variant == "system_prompt":
            messages = [
                ChatMessageSystem(content=carrier_text),
                ChatMessageUser(content=question),
            ]
        elif variant == "direct":
            # Present carrier as an assistant response to the direct task
            messages = [
                ChatMessageUser(content=sender_user_prompt),
                ChatMessageAssistant(content=carrier_text),
                ChatMessageUser(content=question),
            ]
        else:
            # For number, present carrier as user context
            messages = [
                ChatMessageUser(
                    content=(
                        f"Here is some text for context:\n\n{carrier_text}\n\n"
                        f"Now answer the following question:\n{question}"
                    )
                ),
            ]

        output = await model.generate(
            messages,
            config=GenerateConfig(temperature=temperature, max_tokens=64),
        )
        raw = output.completion.strip()
        parsed = parse_animal(raw)
        results.append(
            {
                "question": question,
                "raw_response": raw,
                "parsed_animal": parsed,
            }
        )

    return results
