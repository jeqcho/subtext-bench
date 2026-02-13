"""Check frontier model availability in inspect-ai and litellm prefill support.

Outputs a markdown table to reports/model_compatibility.md
"""

from __future__ import annotations

import datetime
from pathlib import Path

from litellm import get_model_info

# ---------------------------------------------------------------------------
# Known inspect-ai providers (from inspect_ai.model._providers.providers)
# ---------------------------------------------------------------------------
INSPECT_PROVIDERS = {
    "anthropic",
    "azureai",
    "bedrock",
    "cf",
    "fireworks",
    "google",
    "grok",
    "groq",
    "hf",
    "hf_inference_providers",
    "llama_cpp_python",
    "mistral",
    "mockllm",
    "nnterp",
    "none",
    "ollama",
    "openai",
    "openai_api",
    "openrouter",
    "perplexity",
    "sambanova",
    "sglang",
    "together",
    "transformer_lens",
    "vllm",
}

# ---------------------------------------------------------------------------
# Frontier models to check
# Each entry: (display_name, inspect_name, litellm_name, category)
#   - inspect_name: model string as you'd pass to `inspect eval --model ...`
#   - litellm_name: model string for `litellm.get_model_info()`
# ---------------------------------------------------------------------------
MODELS: list[tuple[str, str, str, str]] = [
    # ── Closed-source: Anthropic ──────────────────────────────────────────
    ("Claude Opus 4.6", "anthropic/claude-opus-4-6", "anthropic/claude-opus-4-6", "Closed"),
    ("Claude Opus 4.5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4-5", "Closed"),
    ("Claude Sonnet 4.5", "anthropic/claude-sonnet-4-5", "anthropic/claude-sonnet-4-5", "Closed"),
    ("Claude Haiku 4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-haiku-4-5", "Closed"),
    # ── Closed-source: OpenAI ─────────────────────────────────────────────
    ("GPT-5.2", "openai/gpt-5.2", "openai/gpt-5.2", "Closed"),
    ("GPT-4o", "openai/gpt-4o", "openai/gpt-4o", "Closed"),
    ("GPT-4o Mini", "openai/gpt-4o-mini", "openai/gpt-4o-mini", "Closed"),
    ("o3", "openai/o3", "openai/o3", "Closed"),
    ("o3-mini", "openai/o3-mini", "openai/o3-mini", "Closed"),
    ("o4-mini", "openai/o4-mini", "openai/o4-mini", "Closed"),
    # ── Closed-source: Google ─────────────────────────────────────────────
    ("Gemini 2.5 Pro", "google/gemini-2.5-pro", "gemini/gemini-2.5-pro", "Closed"),
    ("Gemini 2.5 Flash", "google/gemini-2.5-flash", "gemini/gemini-2.5-flash", "Closed"),
    ("Gemini 2.0 Flash", "google/gemini-2.0-flash", "gemini/gemini-2.0-flash", "Closed"),
    # ── Closed-source: xAI ────────────────────────────────────────────────
    ("Grok 4", "grok/grok-4", "xai/grok-4", "Closed"),
    ("Grok 3", "grok/grok-3", "xai/grok-3", "Closed"),
    ("Grok 3 Mini", "grok/grok-3-mini", "xai/grok-3-mini", "Closed"),
    # ── Closed-source: Mistral ────────────────────────────────────────────
    ("Mistral Large", "mistral/mistral-large-latest", "mistral/mistral-large-latest", "Closed"),
    ("Mistral Small", "mistral/mistral-small-latest", "mistral/mistral-small-latest", "Closed"),
    # ── Open-source: DeepSeek (direct) ────────────────────────────────────
    ("DeepSeek V3 (API)", "openai/deepseek-chat", "deepseek/deepseek-chat", "Open"),
    ("DeepSeek R1 (API)", "openai/deepseek-reasoner", "deepseek/deepseek-reasoner", "Open"),
    # ── Open-source: Meta Llama (via Together) ────────────────────────────
    (
        "Llama 4 Maverick",
        "together/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "Open",
    ),
    (
        "Llama 4 Scout",
        "together/meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "Open",
    ),
    (
        "Llama 3.3 70B",
        "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "Open",
    ),
    # ── Open-source: DeepSeek (via Together) ──────────────────────────────
    (
        "DeepSeek V3 (Together)",
        "together/deepseek-ai/DeepSeek-V3",
        "together_ai/deepseek-ai/DeepSeek-V3",
        "Open",
    ),
    (
        "DeepSeek R1 (Together)",
        "together/deepseek-ai/DeepSeek-R1",
        "together_ai/deepseek-ai/DeepSeek-R1",
        "Open",
    ),
    # ── Open-source: via Groq ─────────────────────────────────────────────
    (
        "Llama 3.3 70B (Groq)",
        "groq/llama-3.3-70b-versatile",
        "groq/llama-3.3-70b-versatile",
        "Open",
    ),
]


def _check_inspect_provider(inspect_name: str) -> bool:
    """Return True if the inspect-ai provider prefix is recognised."""
    provider = inspect_name.split("/", 1)[0]
    return provider in INSPECT_PROVIDERS


def _check_litellm(litellm_name: str) -> tuple[bool, bool | None]:
    """Return (known_to_litellm, supports_prefill)."""
    try:
        info = get_model_info(litellm_name)
        prefill = info.get("supports_assistant_prefill")
        return True, prefill
    except Exception:
        return False, None


def _prefill_display(val: bool | None) -> str:
    if val is True:
        return "Yes"
    elif val is False:
        return "No"
    return "Unknown"


def main() -> None:
    rows: list[dict] = []

    for display_name, inspect_name, litellm_name, category in MODELS:
        inspect_ok = _check_inspect_provider(inspect_name)
        litellm_ok, prefill = _check_litellm(litellm_name)
        rows.append(
            {
                "display_name": display_name,
                "category": category,
                "inspect_name": inspect_name,
                "litellm_name": litellm_name,
                "inspect_provider_ok": inspect_ok,
                "litellm_known": litellm_ok,
                "prefill": prefill,
            }
        )

    # ── Build markdown ────────────────────────────────────────────────────
    lines: list[str] = []
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"# Frontier Model Compatibility Report")
    lines.append("")
    lines.append(f"_Generated on {now}_")
    lines.append("")
    lines.append(
        "| Model | Type | Inspect Provider | In LiteLLM | Prefill Support | Inspect Name |"
    )
    lines.append(
        "|-------|------|:----------------:|:----------:|:---------------:|--------------|"
    )

    for r in rows:
        inspect_col = "Yes" if r["inspect_provider_ok"] else "No"
        litellm_col = "Yes" if r["litellm_known"] else "No"
        prefill_col = _prefill_display(r["prefill"])
        lines.append(
            f"| {r['display_name']} "
            f"| {r['category']} "
            f"| {inspect_col} "
            f"| {litellm_col} "
            f"| {prefill_col} "
            f"| `{r['inspect_name']}` |"
        )

    lines.append("")

    # ── Summary ───────────────────────────────────────────────────────────
    lines.append("## Summary")
    lines.append("")
    total = len(rows)
    inspect_yes = sum(1 for r in rows if r["inspect_provider_ok"])
    litellm_yes = sum(1 for r in rows if r["litellm_known"])
    prefill_yes = sum(1 for r in rows if r["prefill"] is True)
    prefill_no = sum(1 for r in rows if r["prefill"] is False)
    prefill_unk = sum(1 for r in rows if r["prefill"] is None)
    lines.append(f"- **Total models checked:** {total}")
    lines.append(f"- **Inspect provider available:** {inspect_yes}/{total}")
    lines.append(f"- **Known to LiteLLM:** {litellm_yes}/{total}")
    lines.append(
        f"- **Prefill support:** {prefill_yes} yes, {prefill_no} no, {prefill_unk} unknown"
    )
    lines.append("")

    # ── Notes ─────────────────────────────────────────────────────────────
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- **Inspect Provider**: Whether the provider prefix (e.g. `anthropic`, `openai`) "
        "is a registered inspect-ai model provider. This does *not* verify API key validity."
    )
    lines.append(
        "- **In LiteLLM**: Whether `litellm.get_model_info()` recognises the model "
        "(pricing, context window, capability metadata available)."
    )
    lines.append(
        "- **Prefill Support**: Whether the model supports assistant message prefilling "
        "(`supports_assistant_prefill` from LiteLLM). "
        '"Unknown" means LiteLLM has no data (often `None`), not necessarily unsupported.'
    )
    lines.append(
        "- Open-source models are listed with a specific hosting provider "
        "(Together, Groq, etc.). The same model may be available through other providers."
    )
    lines.append("")

    md = "\n".join(lines)

    out_path = Path(__file__).resolve().parents[2] / "reports" / "model_compatibility.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(md)
    print(f"\n--- Written to {out_path} ---")


if __name__ == "__main__":
    main()
