# Subtext-Bench: Analysis Summary & Next Steps

## Quick Reference

This directory contains three analysis documents for the subtext-bench project:

1. **DEFENSIVE_ANALYSIS.md** (Main document)
   - Comprehensive risk assessment from defensive/adversarial perspective
   - Lists pain points, constraints, bounds, and dangerous scenarios
   - Recommended mitigations and risk matrix
   - Estimated effort for fixes (10-16 days)

2. **TECHNICAL_RECOMMENDATIONS.md** (Implementation guide)
   - Concrete code-level solutions with examples
   - 8 major improvements: parsing, idempotency, statistics, controls, validation, cost tracking
   - Ready-to-use code snippets for immediate integration

3. **ANALYSIS_SUMMARY.md** (This file)
   - Executive summary and quick reference
   - Critical issues table
   - Priority roadmap for implementation

---

## Critical Issues at a Glance

### Severity Levels

**CRITICAL** (blocks production deployment):
1. **False positive detection** - Cannot distinguish signal from noise
2. **Brittle evaluation parsing** - Parser errors introduce systematic bias
3. **Benchmark gaming risk** - Subtext encoding could be token artifacts, not semantic

**HIGH** (should fix before next major run):
1. **API rate limiting & cost** - Uncontrolled costs, long runtimes
2. **No statistical rigor** - Results lack significance tests and controls
3. **Model version deprecation** - Experiments become unreproducible within months

**MEDIUM** (important for robustness):
1. **Cascading failures** - Partial experiments create biased results
2. **No multi-monitor diversity** - Cannot detect monitor-family bias
3. **Question set bias** - Fixed questions may correlate with prompts

---

## Current State vs. Production Ready

### Current Implementation

| Component | Status | Issues |
|-----------|--------|--------|
| **Sender phase** | Working | Brittle prompt, no encoding verification |
| **Evaluation** | Working | Fragile parsing, no error checking |
| **Metrics** | Working | No statistical tests, no controls |
| **Infrastructure** | Fragile | No checkpointing, cascading failures |
| **Cost tracking** | None | Hidden costs, budget overruns |
| **Reproducibility** | Poor | Random trial IDs, model version churn |

### Production Ready Requirements

| Component | Current | Needed | Effort |
|-----------|---------|--------|--------|
| **Statistical rigor** | Summary stats only | CI, p-values, controls | 2-3 days |
| **Parsing robustness** | Regex-based | LLM-based extraction | 2-3 days |
| **Experiment controls** | None | Null baseline, diverse monitors | 1-2 days |
| **Reproducibility** | Non-deterministic | Idempotent IDs, checkpointing | 2-3 days |
| **Cost management** | None | Budget tracking, alerts | 1-2 days |
| **Human validation** | Not possible | Review framework, inter-rater agreement | 2-3 days |
| **Inspect AI port** | N/A | Multi-phase task orchestration | 3-5 days |

**Total effort: 13-21 days** (similar to DEFENSIVE_ANALYSIS estimate)

---

## Key Findings

### 1. Signal-to-Noise Ratio Problem

Current results (2-5% accuracy) are only marginally above random baseline (6.25% for 16 animals).

**Gap to demonstrate genuine signal:**
- Need clear separation: receiver > monitor by >10% absolute difference
- Need statistical significance: p < 0.05 (or p < 0.003 with Bonferroni correction)
- Need control conditions: show null baseline ≈ 6.25%

**Status**: Cannot conclusively demonstrate subtext transmission yet. Could be noise.

### 2. Evaluation Fragility

Parse errors cascade through entire measurement pipeline:
```
Parser error (e.g., "actually" extracted instead of "animal")
    ↓
Accuracy metric becomes noisy
    ↓
Cannot detect real subtext signal
    ↓
False negative: "no subtext detected"
```

Example failure rates with current regex parser:
- "I would choose X" → correctly extracts X (good)
- "My favorite is X" → correctly extracts X (good)
- "Actually, I prefer X" → extracts "actually" (WRONG)
- "It's probably X, but maybe Y" → extracts "it's" (WRONG)

**Estimated error rate: 15-25% depending on model**

### 3. Phoenix Anomaly

README shows phoenix at 36% receiver accuracy vs <3% for other animals.

**Possible explanations:**
1. **Real effect**: Phoenix truly encodes with higher fidelity (semantic subtext)
2. **Token artifact**: Phoenix word length/tokenization creates implicit bias
3. **Overfitting**: Specific model+task combination happened to work
4. **Question correlation**: One of the evaluation questions specifically triggers "phoenix"

**Cannot distinguish without further investigation.**

### 4. Model Version Lock-In

Model versions sunset regularly:
- Anthropic: claude-haiku-4-5-20250115 → will be deprecated by 2026-02-15
- OpenAI: GPT-5 timeline unclear, pricing/access unknown
- Qwen: Versions update frequently on OpenRouter

**Impact**: Experiments become non-reproducible within 12 months.

### 5. Cost Exposure

Estimated costs per full experiment:
- Anthropic Haiku: $0.15-0.30 (cheap, low capability)
- Anthropic Opus: $1.00-2.00 (expensive, high capability)
- OpenAI GPT-5: $0.50-1.00 per trial (very expensive)
- Qwen models: $0.05-0.10 (cheap)

**For 5-model sweep with 160 trials each:**
- Conservative estimate: $50-100
- Worst case (retries, failures): $150-200

**No current budget tracking or alerts.**

---

## Priority Roadmap

### Phase 1: Immediate (1-2 days)
**Goal**: Enable safe scaling to production

- [ ] Add cost tracker with budget alerts
  - Prevents runaway costs
  - 1 day effort
  - File: `src/subtext/monitoring/cost_tracker.py`

- [ ] Implement idempotent trial IDs + checkpointing
  - Enables resumable experiments
  - 2 days effort
  - Files: `src/subtext/experiment/idempotent_runner.py`, `src/subtext/persistence/trial_store.py`

### Phase 2: Core Improvements (3-4 days)
**Goal**: Fix critical evaluation issues

- [ ] Replace regex parsing with LLM-based extraction
  - Reduce parse errors from ~20% to ~2%
  - 2-3 days effort
  - File: `src/subtext/evaluation/robust_extraction.py`

- [ ] Implement null baseline control
  - Establish ground truth: what % accuracy is just noise?
  - 1 day effort
  - File: `src/subtext/experiment/null_control.py`

### Phase 3: Statistical Rigor (2-3 days)
**Goal**: Enable confident conclusions

- [ ] Add confidence intervals and significance tests
  - Bootstrap CIs for accuracy metrics
  - Mann-Whitney U test for monitor vs receiver comparison
  - Bonferroni correction for multiple comparisons
  - 2-3 days effort
  - File: `src/subtext/evaluation/statistical_metrics.py`

- [ ] Implement human validation framework
  - Manual review of 10% of responses
  - Compute inter-rater agreement (Cohen's kappa)
  - 2 days effort
  - File: `src/subtext/evaluation/human_validation.py`

### Phase 4: Robustness (2-3 days)
**Goal**: Eliminate confounding factors

- [ ] Add question rotation + validation
  - Prevent question-prompt correlation
  - Validate questions don't reveal animals
  - 1 day effort
  - File: `src/subtext/evaluation/question_pool.py`

- [ ] Implement multi-monitor comparison
  - Use GPT-5, Claude, Llama-2 as monitors
  - Detect monitor-family entanglement
  - 2 days effort
  - File: `src/subtext/evaluation/multi_monitor.py`

### Phase 5: Reproduction & Documentation (1-2 days)
**Goal**: Long-term reproducibility

- [ ] Document all hyperparameters in config
- [ ] Add experiment metadata to outputs
- [ ] Version control prompts
- 1-2 days effort

### Phase 6: Inspect AI Port (3-5 days)
**Goal**: Integrate with Inspect AI framework

- [ ] Define `SubtextTask` for sender-receiver-monitor game
- [ ] Implement custom `Solver` classes for each role
- [ ] Create `SubtextMetric` with statistical tests
- [ ] Handle multi-phase orchestration
- 3-5 days effort

---

## Recommended Execution Plan

### For Immediate Production Use

**If you need to run experiments NOW:**

1. Add cost tracker (1 day) - prevents financial surprises
2. Add idempotent trials (2 days) - allows resuming from crashes
3. Run single model on 1-2 animals first - validate basic flow
4. Manually review 20% of responses - catch parser issues
5. Don't publish/cite results without statistics and controls

**Estimated time: 3-4 days of setup, then experimental runs**

### For Defensible Results

**If you want publication-ready benchmark:**

1. Complete Phases 1-4 (12-16 days total)
2. Run full experiment suite
3. Generate statistical summary with CI and p-values
4. Publish with explicit limitations and control results
5. Make data/code available for reproduction

**Estimated time: 16-20 days preparation, then experiments**

### For Inspect AI Integration

**If building reusable benchmark framework:**

1. Complete Phases 1-5
2. Port to Inspect AI (Phase 6)
3. Design flexible sender/receiver/monitor assignment
4. Test with multiple model combinations
5. Package as public Inspect AI task

**Estimated time: 20-25 days total**

---

## Risk Assessment Table

| Risk | Current Status | Severity | Effort to Fix |
|------|---|---|---|
| False positive (noise as signal) | Uncontrolled | CRITICAL | HIGH (3 days) |
| Brittle parsing introduces errors | Uncontrolled | CRITICAL | HIGH (3 days) |
| Benchmark gaming possible | Uncontrolled | CRITICAL | HIGH (2 days) |
| API costs untracked | Uncontrolled | HIGH | LOW (1 day) |
| No significance tests | Uncontrolled | HIGH | MEDIUM (2 days) |
| Non-reproducible experiments | Uncontrolled | HIGH | MEDIUM (2 days) |
| Cascading failures on errors | Uncontrolled | MEDIUM | MEDIUM (2 days) |
| Model version deprecation | Inevitable | MEDIUM | LOW (1 day) |
| Single monitor bias | Uncontrolled | MEDIUM | MEDIUM (2 days) |
| Question set correlation | Uncontrolled | MEDIUM | LOW (1 day) |

---

## Frequently Asked Questions

### Q: Should we run experiments with current code?

**A**: Cautiously. Current implementation is suitable for:
- Feasibility testing (does subtext transmission work at all?)
- Debugging (does orchestration work?)
- Rough estimates (which animals show promise?)

**Not suitable for:**
- Publishing results (no statistical rigor)
- Drawing conclusions (no controls, no significance tests)
- Scaling to production (no cost tracking, no resumability)

### Q: What's the minimum viable improvement?

**A**: To get defensible results:
1. Add null baseline (1 day) - establishes ground truth
2. Replace parsing with LLM (2 days) - reduces noise
3. Add statistics (2 days) - enables significance testing
4. Run experiment (2-8 hours per model)

**Minimum effort: 5 days of development**

### Q: Can we fix this incrementally?

**A**: Yes. Recommended increments:
1. **Week 1**: Phase 1 (cost + idempotency)
2. **Week 2**: Phase 2 (parsing + baseline) + run experiment
3. **Week 3**: Phase 3 (statistics) + analyze results
4. **Week 4**: Phase 4 (robustness) + final publication

Each phase is independent and can be deployed separately.

### Q: What about the phoenix anomaly?

**A**: Needs investigation:
1. Is it reproducible? (run same animal again)
2. Is it model-specific? (test with different sender)
3. Is it task-specific? (test with different tasks)
4. Is it question-specific? (change evaluation questions)

Without these controls, cannot conclude phoenix is special.

### Q: When should we port to Inspect AI?

**A**: After Phase 4 is complete. Reasons:
1. Inspect AI best for final deployment, not R&D iteration
2. Need to stabilize benchmark design first
3. Multi-phase orchestration easier to prototype in raw Python
4. Once design is final, Inspect AI port is straightforward (3-5 days)

---

## File References

### Documentation
- `/Users/jeqcho/subtext-bench/DEFENSIVE_ANALYSIS.md` - Main risk assessment
- `/Users/jeqcho/subtext-bench/TECHNICAL_RECOMMENDATIONS.md` - Implementation code
- `/Users/jeqcho/subtext-bench/ANALYSIS_SUMMARY.md` - This file

### Reference Implementation
- `/Users/jeqcho/subtext-bench/reference/subtext-playground/` - Current working code
- `/Users/jeqcho/subtext-bench/reference/subtext-playground/README.md` - Original design
- `/Users/jeqcho/subtext-bench/reference/receiver-amplification.txt` - Experimental ideas

---

## Conclusion

The subtext-bench framework is **conceptually sound** but **operationally fragile**. Current implementation:

✓ Demonstrates the core idea (subtext transmission is possible)
✗ Cannot conclusively prove genuine signal exists
✗ Not suitable for production without significant hardening

**Recommended next step**: Implement Phase 1-2 improvements (cost + idempotency + parsing) before the next major experimental run. This will reduce risk and cost while maintaining velocity.

Then use results from Phases 1-2 to decide whether to proceed with full statistical validation (Phase 3-4) and Inspect AI integration (Phase 5-6).

---

## Contact & Questions

For questions about this analysis:
1. Refer to specific risk sections in DEFENSIVE_ANALYSIS.md
2. Review code examples in TECHNICAL_RECOMMENDATIONS.md
3. Check risk matrix above for priority guidance

For implementation help:
- TECHNICAL_RECOMMENDATIONS.md has ready-to-use code snippets
- Each section includes integration points with existing codebase
- Estimate effort and dependencies for planning

