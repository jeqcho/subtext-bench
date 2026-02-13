# Frontier Model Compatibility Report

_Generated on 2026-02-13 03:38 UTC_

| Model | Type | Inspect Provider | In LiteLLM | Prefill Support | Inspect Name |
|-------|------|:----------------:|:----------:|:---------------:|--------------|
| Claude Opus 4.6 | Closed | Yes | Yes | No | `anthropic/claude-opus-4-6` |
| Claude Opus 4.5 | Closed | Yes | Yes | Yes | `anthropic/claude-opus-4-5` |
| Claude Sonnet 4.5 | Closed | Yes | Yes | Yes | `anthropic/claude-sonnet-4-5` |
| Claude Haiku 4.5 | Closed | Yes | Yes | Yes | `anthropic/claude-haiku-4-5` |
| GPT-5.2 | Closed | Yes | Yes | Unknown | `openai/gpt-5.2` |
| GPT-4o | Closed | Yes | Yes | Unknown | `openai/gpt-4o` |
| GPT-4o Mini | Closed | Yes | Yes | Unknown | `openai/gpt-4o-mini` |
| o3 | Closed | Yes | Yes | Unknown | `openai/o3` |
| o3-mini | Closed | Yes | Yes | Unknown | `openai/o3-mini` |
| o4-mini | Closed | Yes | Yes | Unknown | `openai/o4-mini` |
| Gemini 2.5 Pro | Closed | Yes | Yes | Unknown | `google/gemini-2.5-pro` |
| Gemini 2.5 Flash | Closed | Yes | Yes | Unknown | `google/gemini-2.5-flash` |
| Gemini 2.0 Flash | Closed | Yes | Yes | Unknown | `google/gemini-2.0-flash` |
| Grok 4 | Closed | Yes | Yes | Unknown | `grok/grok-4` |
| Grok 3 | Closed | Yes | Yes | Unknown | `grok/grok-3` |
| Grok 3 Mini | Closed | Yes | Yes | Unknown | `grok/grok-3-mini` |
| Mistral Large | Closed | Yes | Yes | Yes | `mistral/mistral-large-latest` |
| Mistral Small | Closed | Yes | Yes | Yes | `mistral/mistral-small-latest` |
| DeepSeek V3 (API) | Open | Yes | Yes | Yes | `openai/deepseek-chat` |
| DeepSeek R1 (API) | Open | Yes | Yes | Yes | `openai/deepseek-reasoner` |
| Llama 4 Maverick | Open | Yes | Yes | Unknown | `together/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` |
| Llama 4 Scout | Open | Yes | Yes | Unknown | `together/meta-llama/Llama-4-Scout-17B-16E-Instruct` |
| Llama 3.3 70B | Open | Yes | Yes | Unknown | `together/meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| DeepSeek V3 (Together) | Open | Yes | Yes | Unknown | `together/deepseek-ai/DeepSeek-V3` |
| DeepSeek R1 (Together) | Open | Yes | Yes | Unknown | `together/deepseek-ai/DeepSeek-R1` |
| Llama 3.3 70B (Groq) | Open | Yes | Yes | Unknown | `groq/llama-3.3-70b-versatile` |

## Summary

- **Total models checked:** 26
- **Inspect provider available:** 26/26
- **Known to LiteLLM:** 26/26
- **Prefill support:** 7 yes, 1 no, 18 unknown

## Notes

- **Inspect Provider**: Whether the provider prefix (e.g. `anthropic`, `openai`) is a registered inspect-ai model provider. This does *not* verify API key validity.
- **In LiteLLM**: Whether `litellm.get_model_info()` recognises the model (pricing, context window, capability metadata available).
- **Prefill Support**: Whether the model supports assistant message prefilling (`supports_assistant_prefill` from LiteLLM). "Unknown" means LiteLLM has no data (often `None`), not necessarily unsupported.
- Open-source models are listed with a specific hosting provider (Together, Groq, etc.). The same model may be available through other providers.
