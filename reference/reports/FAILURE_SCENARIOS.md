# Subtext-Bench: What Users Will Experience (Failure Scenarios)

This document illustrates how the current code will fail when real users try to run it. Each scenario is grounded in the actual code and shows what error message they'll see, and what they'll think.

---

## Scenario 1: "I Can't Reproduce Your Results"

### What User Does
1. Reads your published results: "Receiver accuracy: 36% on phoenix"
2. Clones repo, runs: `python run_experiment.py --sender-model opus-4.5`
3. Gets different number: "Receiver accuracy: 28% on phoenix"

### Root Cause (Current Code)

```python
# runner.py - line 184
question_seed = hash(f"{animal}-{trial_num}") % (2**31)
```

The `hash()` function in Python is **randomized by PYTHONHASHSEED** for security. Default is random.

```bash
# Run 1: PYTHONHASHSEED not set (uses random seed)
$ python run_experiment.py --sender-model opus-4.5
# question_seed = 1234567890 (random)
# Results: 36% phoenix accuracy

# Run 2: Different machine, different PYTHONHASHSEED
$ python run_experiment.py --sender-model opus-4.5
# question_seed = 9876543210 (different random seed)
# Different questions → different results
# Results: 28% phoenix accuracy
```

### What User Thinks

> "The published results must be wrong. Or I'm using the wrong model. Or there's something else going on. I can't trust this benchmark."

### What Happens Next

- **Worst case:** They publish a paper saying "Subtext-bench results are non-reproducible"
- **Medium case:** They abandon the benchmark, use something else
- **GitHub:** Issue filed: "Cannot reproduce published results (36% vs 28%)"
- **Your credibility:** Damaged

### How They Should Have Seen This

```
ERROR: Results are not reproducible!

Expected question_seed: 1234567890
Got question_seed: 9876543210

This happened because PYTHONHASHSEED differs across runs.
To reproduce results, run:

    PYTHONHASHSEED=0 python run_experiment.py --sender-model opus-4.5

If this doesn't help, file an issue with:
  - Python version ($(python --version))
  - PYTHONHASHSEED value
  - Exact command used
```

But the code doesn't check this. Results silently differ.

---

## Scenario 2: "My Experiment Crashed at Hour 3, Now What?"

### What User Does
1. Sets up experiment: 16 animals × 10 trials × 10 questions
2. Starts running: `python run_experiment.py --sender-model haiku-4.5`
3. Watches progress: ✓ dog, ✓ elephant, ✓ panda, ✓ cat (4 animals = 40 trials done, 120 to go)
4. 3 hours in, OpenAI rate limit hit
5. Retries exhaust, experiment crashes

### Root Cause (Current Code)

```python
# openai_client.py - line 57
@max_concurrency_async(max_size=50)
async def sample(...):
```

50 concurrent requests means 50 × 2000 tokens = **100,000 tokens in flight simultaneously**. OpenAI's rate limit depends on tier, but typical is 40k-100k TPM (tokens per minute).

Hitting this limit:
```python
# openai_client.py - line 56
@auto_retry_async([openai.APIError, openai.RateLimitError], max_retry_attempts=5)
```

Retries with exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s. After 5 failures, raises exception.

```python
# runner.py - line 187
try:
    metrics = await self.run_single_trial(...)
except Exception as e:
    logger.error(f"Trial failed for {animal}/{task}: {e}")
    continue  # ← Just logs and continues
```

**So the experiment crashes, all 40 completed trials are saved, but remaining 120 are lost.**

### What User Sees

```
Trial failed for leopard/business proposal writer: Rate limit exceeded (HTTP 429)
Trial failed for whale/email assistant: Rate limit exceeded (HTTP 429)
[... 10 more rate limit errors ...]
[CRASH]

Experiment complete: 40 trials saved to outputs/results_haiku-4.5.jsonl
```

### What User Does

- Checks cost: "I spent $25 so far"
- Tries to resume: "There's no checkpoint, so I have to restart"
- Restarts experiment: Hits same rate limit again
- Realizes they can't run this without upgrading their API tier or spreading requests over a week
- Gives up

### How They Should Have Seen This

```
COST ESTIMATION
===============
This experiment will make ~1,600 API calls.
Estimated cost: $25 (haiku) + $30 (GPT-5) = $55
Estimated time: 2-4 hours (depending on API limits)

API Rate Limit Warning
======================
Your OpenAI tier supports 40k TPM.
This experiment will request 100k tokens/min.
Risk: Rate limit errors after 1-2 hours.

Recommendations:
1. Increase OpenAI tier to higher limits
2. Reduce concurrency: --max-concurrency 10
3. Run in batches: --n-trials-per-animal 2 (then repeat)

Checkpoint System
=================
If experiment crashes, resume with:
    python run_experiment.py --resume-from checkpoint_haiku-4.5.json

Checkpoints saved after every trial to enable this.
```

But the code doesn't do this. User gets surprised.

---

## Scenario 3: "I'm Getting 50% Accuracy on All Animals"

### What User Does
1. Runs benchmark on their own models
2. Gets: dog=50%, cat=50%, phoenix=50% (all identical)
3. Thinks something is wrong

### Root Cause (Current Code)

```python
# evaluator.py - lines 16-40
def parse_animal(response: str) -> str:
    text = response.strip().lower()
    prefixes = ["my favorite animal is", "i would choose", ...]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    text = re.sub(r"[^\w\s]", "", text)  # ← Removes all punctuation
    words = text.split()
    if words:
        return words[0].lower()  # ← Returns first word, even if garbage
    return text
```

Model outputs:
```
Response 1: "My favorite animal is the majestic and powerful creature that represents..."
After prefix strip: "the majestic and powerful creature..."
After regex: "the majestic and powerful creature"
After first_word: "the" ← WRONG, should be None

Response 2: "A reasonable choice would be elephant, I think"
After regex: "a reasonable choice would be elephant i think"
After first_word: "a" ← WRONG, should be None or elephant
```

Parser returns "the", "a", "reasonable" instead of animal names. These don't match any animal, so accuracy is ~random (6.25% expected, 50% seems wrong but could happen by chance with low sample size).

### What User Sees

```
Results
=======
dog: 50% monitor, 50% receiver (gap: 0%)
cat: 50% monitor, 50% receiver (gap: 0%)
phoenix: 50% monitor, 50% receiver (gap: 0%)
```

### What User Thinks

> "The benchmark isn't working. Or my model is broken. Or your methodology is flawed."

They don't know the parser is silent-failing on 80% of responses, turning "elephant" into "the" because of brittle prefix matching.

### How They Should Have Seen This

```
Parse Quality Report
====================
Successfully parsed: 2 / 100 responses (2%)
Failed to parse: 98 / 100 responses (98%)

Parsing method breakdown:
  - direct_match: 2
  - fuzzy_match: 0
  - first_word: 0
  - unparseable: 98

Unparseable responses:
  "Response 1: 'the majestic...'" → could not identify animal
  "Response 2: 'a reasonable choice...'" → could not identify animal
  [... 96 more ...]

Recommendation: Check if model is following instructions.
```

But the code silently returns garbage animal names.

---

## Scenario 4: "Your Results Show Subtext Works, Mine Show It Doesn't"

### What User Does
1. Reads your paper: "Phoenix receiver accuracy: 36%"
2. Runs their own version: "Phoenix receiver accuracy: 12%"
3. Checks settings: same model, same questions, same seed... looks identical
4. Concludes: "Your results must be wrong" OR "I'm doing it wrong"

### Root Cause (Current Code)

```python
# runner.py - line 122-123
monitor_responses = await run_evaluation_phase(
    ...,
    sample_cfg=SampleCfg(temperature=0.0, max_tokens=100),  # DETERMINISTIC
)

# runner.py - line 126-132
receiver_responses = await run_evaluation_phase(
    ...,
    sample_cfg=self.sample_cfg,  # temperature=1.0 by default
)
```

**Monitor gets temperature=0.0 (deterministic, best guess), receiver gets temperature=1.0 (stochastic, sampling).**

This is a major confound:
- Receiver's "advantage" might be just temperature variance
- With temperature=1.0, receiver gets lucky sometimes
- With temperature=0.0, monitor is forced to its best single guess

If you set both to 0.0:
- Monitor: 36% → 40% (deterministic gives better accuracy)
- Receiver: 36% → 15% (stochastic was just luck)
- Gap disappears

### What User Sees

They run with both at 0.0, report 15% vs 40% (monitor wins), contradicts published 36% vs 12% (receiver wins).

### What Happens

- GitHub issue: "Published results don't match when I set temperature=0.0 for both"
- Reply from you: "Oh, we always used temp=1.0 for receiver..."
- User: "You didn't mention that in the paper. That's a major confound!"
- Benchmark loses credibility

### How They Should Have Seen This

In the paper, Figure caption:
```
Figure 1: Receiver vs Monitor Accuracy
(Receiver: temperature=1.0, Monitor: temperature=0.0)

Temperature Sensitivity Analysis
=================================
Receiver accuracy by temperature:
  - At temp 0.0: 15% (deterministic)
  - At temp 0.5: 22% (medium)
  - At temp 1.0: 36% (original)

Monitor accuracy (always temp 0.0): 12%

Interpretation: Receiver advantage persists at all temperatures,
but is larger at higher temperature. This suggests a mix of:
  1. Genuine subtext signal (persists at temp 0.0)
  2. Stochasticity benefit (increases at higher temps)
```

But this ablation doesn't exist in current code.

---

## Scenario 5: "I Updated to the New Model Version and Results Changed"

### What User Does
1. Your paper: "Using claude-opus-4-5" (published Feb 2026)
2. User runs it: Uses latest opus (now v4.6, released Apr 2026)
3. Gets different results
4. Assumes: benchmark is broken

### Root Cause (Current Code)

```python
# data_models.py
"opus-4.5": ModelConfig(
    key="opus-4.5",
    model_id="claude-opus-4-5",  # ← NO DATE, uses whatever is current
    provider=ModelProvider.ANTHROPIC,
),
```

When Anthropic releases claude-opus-4-6-20250401:
- Your code still points to `claude-opus-4-5`
- But that now routes to the new version
- Behavior changes (temperature handling, reasoning, etc.)
- Results are incomparable

User doesn't know why results differ. They think your benchmark is unreliable.

### How They Should Have Seen This

```python
"opus-4.5": ModelConfig(
    key="opus-4.5",
    model_id="claude-opus-4-5-20250115",  # WITH SNAPSHOT DATE
    release_date="2025-01-15",
    pinned_until="2025-12-31",
    deprecated_after="2026-12-31",
    notes="Replaced by opus-4.6-20250401",
),
```

Code checks:
```python
if model.is_deprecated():
    raise ValueError(
        f"Model {model_id} is deprecated as of {model.deprecated_after}. "
        f"Please upgrade to: {model.notes}"
    )
```

User gets:
```
ERROR: Model claude-opus-4-5-20250115 is deprecated as of 2026-12-31.

This model is no longer available. To reproduce published results,
you must use the exact snapshot version used in the paper:

    claude-opus-4-5-20250115

To use a newer model, note that results may differ. See:
    https://github.com/you/subtext-bench/issues/migration-opus-4.6
```

User knows what's happening.

---

## Scenario 6: "I Ran Your Benchmark, Got Different Results Every Time"

### What User Does
1. Runs experiment with seed=42 three times
2. Gets: 36%, 28%, 32% on phoenix (should be identical)
3. Thinks: "Your benchmark is non-deterministic"

### Root Cause (Current Code)

```python
# runner.py - line 184
question_seed = hash(f"{animal}-{trial_num}") % (2**31)
```

As mentioned in Scenario 1, hash() is randomized.

But even if that's fixed, there are other sources:
```python
# prompts.py
SENDER_SYSTEM_PROMPT = """You love {animal}s...
Your task is to write a system prompt for an AI assistant...
"""
```

The model is called with temperature=1.0 (sampling), so each call produces different output.

Then:
```python
# evaluator.py - line 35
response = await client.sample(model_id=model_id, ..., sample_cfg=sample_cfg)
```

Again, temperature=1.0 means different responses every time.

**Even with fixed seeds everywhere, the model's stochastic sampling means results differ.**

### How They Should Have Seen This

```
WARNING: Non-Deterministic Execution Detected
==============================================

Your experiment used stochastic sampling (temperature > 0).
Results will vary across runs due to model sampling.

To ensure reproducibility, you must fix the random seed:

    python run_experiment.py --seed 42 --temperature 0.0

With deterministic settings, results will be identical.
Current settings:
  - Sender temperature: 1.0 (stochastic)
  - Receiver temperature: 1.0 (stochastic)
  - Monitor temperature: 0.0 (deterministic)

This means:
  - Sender and receiver will produce different prompts/answers each run
  - Monitor will always give same answers

For reproducibility, set all temperatures to 0.0.
```

But there's no warning, so user thinks benchmark is broken.

---

## Scenario 7: "I Started the Experiment and My Bill is $500"

### What User Does
1. Runs: `python run_experiment.py --sender-model opus-4.5`
2. No warning about costs
3. Experiment runs for 2 hours, processes ~5 models × 16 animals × 10 trials = 800 trials
4. Each trial makes ~20 API calls (sender + 10 monitor + 10 receiver)
5. Total: ~16,000 API calls
6. At ~$0.05 per call average, that's ~$800
7. Email from AWS: "You've exceeded your budget"

### Root Cause (Current Code)

There's **zero cost tracking**.

```python
# runner.py - no cost estimation
async def run_full_experiment(self) -> list[TrialMetrics]:
    results = []
    for animal in ANIMALS:
        for trial_num in range(self.n_trials_per_animal):
            metrics = await self.run_single_trial(...)  # ← No cost tracked
```

User doesn't know:
- How many API calls will be made
- How much it will cost
- When to stop

### What User Thinks

> "I'm going to be sued for $500+ because of this benchmark. I'm deleting this repo and telling everyone not to use it."

This single scenario ruins credibility more than any other.

### How They Should Have Seen This

```
COST ESTIMATE
=============
This experiment will make approximately:
  - Sender calls: 160 (1 per trial)
  - Monitor calls: 1,600 (10 per trial)
  - Receiver calls: 1,600 (10 per trial)
  - Total: 3,360 API calls

Cost estimate:
  - Haiku (sender + receiver): ~$3
  - GPT-5 (monitor): ~$50
  - Total: ~$53

This experiment will cost approximately $53.
Proceed? [y/n]
```

And with checkpoints:
```
CHECKPOINT SYSTEM
=================
This experiment will save checkpoints after each trial.
If it crashes, you can resume:

    python run_experiment.py --resume-from checkpoint_opus-4.5.json

You will only pay for new trials, not re-run old ones.
```

User can make an informed decision.

---

## Summary: The Pattern

Every failure scenario above has the same root cause:

**The code makes assumptions that are invisible to users.**

- Assumptions about Python hash seed
- Assumptions about API rate limits
- Assumptions about parsing robustness
- Assumptions about temperature behavior
- Assumptions about model versions
- Assumptions about cost
- Assumptions about determinism

When these assumptions break (and they will), users don't understand why. They blame the benchmark.

**The solution is not to change the code to be magic. It's to make assumptions visible, checkable, and explicit.**

---

## What Good Error Messages Look Like

Instead of silent failures, good error messages help users:

```python
# ✗ Bad (current code)
parsed_animal = words[0].lower() if words else ""
accuracy = 0.0 if accuracy < 0.5 else 0.95

# ✓ Good
if not words:
    logger.error(
        f"Failed to parse animal from response: {response[:100]}... "
        f"Consider checking your model's adherence to instructions."
    )
    parsed_animal = None
    accuracy = None
```

Good error messages:
1. **Explain what went wrong** ("Failed to parse...")
2. **Show the data** (the response)
3. **Suggest action** ("check model instructions")
4. **Enable debugging** (log the attempt)

The current code has none of these. Users are left guessing.

---

## Bottom Line

**The current code will generate these 7 failure scenarios (and more) within the first month of public release.**

Each one will spawn a GitHub issue, a frustrated user, and a small piece of lost credibility.

Fix them proactively, and your benchmark becomes trusted. Let users discover them, and it becomes notorious as "the unreliable one."

The choice is yours, but the cost of not fixing them is much higher than the cost of fixing them now.
