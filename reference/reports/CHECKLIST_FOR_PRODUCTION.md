# Subtext-Bench: Production Release Checklist

Use this checklist to verify the benchmark is ready for public release and downstream use (finetuning, GEPA optimization).

---

## PHASE 1: Data Integrity (MUST HAVE BEFORE ANY RELEASE)

### Model Versioning
- [ ] All model IDs include snapshot dates (e.g., `claude-haiku-4-5-20250115`, not `claude-haiku-4-5`)
- [ ] Model registry includes `release_date`, `deprecated_after`, and `pinned_until`
- [ ] Code validates model ID format at runtime (raises error if no snapshot date)
- [ ] Deprecation warnings appear if model is >6 months old
- [ ] README explicitly documents which model versions are supported

### Reproducibility
- [ ] Every trial result includes `reproducibility_token` (SHA256 of all settings)
- [ ] Every result logs: `python_version`, `python_hash_seed`, `experiment_seed`
- [ ] Every result logs actual task selected and actual questions used
- [ ] Test: running same experiment with same seed produces byte-identical results
- [ ] Reproducibility info documented in README with instructions for verification

### Cost Tracking
- [ ] Cost tracker integrated into ExperimentRunner
- [ ] Cost estimation printed before experiment starts
- [ ] Hard budget limit enforced (experiment stops if over budget)
- [ ] Cost log saved with every experiment
- [ ] README includes cost table (e.g., "$75 for full experiment")

### Checkpoint & Resume
- [ ] CheckpointManager saves after every trial
- [ ] Failed trials can be resumed without re-running completed trials
- [ ] Checkpoint file format documented (JSON, atomic writes)
- [ ] Test: crash mid-experiment, resume, get same results + cost savings

### Parsing Validation
- [ ] `parse_animal()` returns None for unparseable responses (not garbage)
- [ ] Parsed animal is validated against ANIMALS list
- [ ] Parsing confidence score logged (0.0-1.0)
- [ ] Parsing method logged ("direct_match", "fuzzy", etc.)
- [ ] Unparseable % reported per trial
- [ ] Test: verify that nonsense input produces None, not random animal

---

## PHASE 2: Experimental Rigor (MUST HAVE BEFORE PUBLISHING RESULTS)

### Proper Baselines
- [ ] Random baseline trial run (expected: ~6.25% accuracy = 1/16)
- [ ] No-subtext control trial (sender imbued with different animal)
- [ ] Monitor-with-hint control (monitor told the secret animal)
- [ ] All three baselines included in final results
- [ ] Metrics report: receiver_acc, monitor_acc, gap, and evasion_strength separately

### Statistical Significance
- [ ] Confidence intervals computed for accuracy (binomial, not just std dev)
- [ ] Sample sizes logged (n_trials per animal)
- [ ] Power analysis documented (can we detect 10% gap with n=10 trials?)
- [ ] Report if receiver > monitor gap is statistically significant (p < 0.05)
- [ ] Caveat in results: which animals show signal, which don't

### Generalization Testing
- [ ] Train/test split for questions (model doesn't see test Qs during eval)
- [ ] Held-out test accuracy reported separately from train
- [ ] OOD test with completely different questions (colors, preferences, etc.)
- [ ] Generalization gap measured and reported
- [ ] If gap is large, model is overfitting to specific questions

### Question Ordering
- [ ] Trial run with original question order
- [ ] Trial run with reversed question order
- [ ] Trial run with shuffled question order
- [ ] Robustness metric computed (do results flip with different order?)
- [ ] Report if effect is order-sensitive or robust

### Temperature Ablation
- [ ] Receiver evaluated at temperature 0.0 (deterministic, matches monitor)
- [ ] Receiver evaluated at temperature 0.5 (medium)
- [ ] Receiver evaluated at temperature 1.0 (original)
- [ ] Gap computed for each temperature
- [ ] If gap only at temp 1.0, it's just variance (not subtext)
- [ ] If gap at temp 0.0, it's genuine subtext

### Monitor Instructions
- [ ] Monitor given explicit instructions (logged)
- [ ] Tested with monitor unaware vs aware of test
- [ ] Tested with monitor adversarial (explicitly told to find secret)
- [ ] Instructions and results for each variant documented

---

## PHASE 3: Production Readiness (NICE TO HAVE, RECOMMENDED)

### API Compatibility
- [ ] Model capability matrix: which features each model supports
- [ ] Sampling config validation per model (temp range, max_tokens, etc.)
- [ ] Fallback strategy if API parameter not supported
- [ ] API version pinning in pyproject.toml
- [ ] Code tested with API versions N and N-1

### Rate Limiting & Resilience
- [ ] Adaptive concurrency: reduce if rate-limited, increase if successful
- [ ] Exponential backoff implemented (not just linear)
- [ ] Rate limit telemetry: log backoff duration, recovery time
- [ ] Graceful degradation: non-critical calls fail, critical continue
- [ ] Circuit breaker: stop if repeated failures (don't hammer API)

### Documentation
- [ ] README includes: setup, usage, costs, caveats, examples
- [ ] Data dictionary: every field in results JSON documented
- [ ] Example: annotated output of one trial with explanations
- [ ] Limitations section: explicit list of what this benchmark does/doesn't test
- [ ] Migration guide: what to do when model versions deprecate

### Testing
- [ ] Unit tests: parsing, cost estimation, reproducibility token
- [ ] Integration test: single trial end-to-end
- [ ] Reproducibility test: same seed produces same results
- [ ] Cost tracking test: correct cost estimation
- [ ] CI/CD: automated tests on PR, blocking failures

### Telemetry & Monitoring
- [ ] Logging at INFO level: trial starts, results, costs
- [ ] Logging at WARNING level: API errors, budget warnings, model deprecations
- [ ] Metrics dashboard (optional): visualize cost, accuracy, trial status
- [ ] Error reporting: structured logs for debugging

---

## PHASE 4: Downstream Readiness (FOR USERS EXTENDING THIS)

### Dataset Quality
- [ ] Publish raw results (JSONL) alongside processed metrics
- [ ] Publish all prompts, questions, and settings used
- [ ] Publish train/test/OOD splits
- [ ] Checksum (SHA256) of datasets for integrity verification

### Finetuning Safety
- [ ] Warning in README: "Don't finetune on raw benchmark data without validation"
- [ ] Document which animals show signal vs noise
- [ ] Document potential confounds (question-specific patterns, temperature effects)
- [ ] Recommend: finetuning teams should regenerate data with new seed for validation

### GEPA Optimization
- [ ] Document: this measures subtext quality, not evasion directly
- [ ] Recommend: pair with adversarial evaluation (monitor with explicit instructions)
- [ ] Recommend: use train/test split, not benchmark results for training
- [ ] Caveat: high receiver accuracy may not mean good subtext (could be task-correlation)

### Reproducibility for Users
- [ ] Provide exact command to reproduce published results
- [ ] Provide exact Python version, package versions
- [ ] Provide PYTHONHASHSEED value
- [ ] Provide script to compute reproducibility token and verify

---

## PRE-RELEASE VALIDATION

### Code Review Checklist
- [ ] No hardcoded API keys (use .env or environment variables)
- [ ] No sensitive data in logs
- [ ] No unbounded loops or memory leaks
- [ ] Error messages are helpful (not cryptic)
- [ ] Deprecation warnings appear 6 months before cutoff

### Performance Validation
- [ ] Single trial: completes in <5 minutes (with realistic models)
- [ ] Full experiment: completes in <2 hours (16 animals, 10 trials, 10 questions)
- [ ] Memory usage: doesn't exceed 2GB during run
- [ ] No memory leaks: test long-running experiment

### Integration Validation
- [ ] Works with OpenAI API (GPT-5)
- [ ] Works with Anthropic API (Haiku, Sonnet, Opus)
- [ ] Works with OpenRouter API (Qwen models)
- [ ] Handles API errors gracefully (timeouts, rate limits, auth errors)
- [ ] Handles network outages (retries, resumable)

### Documentation Validation
- [ ] README is complete and accurate
- [ ] All code examples in docs are tested and work
- [ ] Troubleshooting section covers common issues
- [ ] FAQ section answers likely user questions
- [ ] CHANGELOG documents all changes since last release

### Release Readiness
- [ ] Version number updated (semantic versioning)
- [ ] CHANGELOG updated
- [ ] Git tags created for release
- [ ] GitHub release notes written
- [ ] PyPI package published (if applicable)

---

## SIGN-OFF

Release should only proceed if:

1. **All PHASE 1 items checked** (data integrity non-negotiable)
2. **All PHASE 2 items checked** (reproducibility and rigor)
3. **Most PHASE 3 items checked** (few exceptions acceptable)
4. **All pre-release validation items checked**

### Sign-Off Required From:
- [ ] **Lead Researcher**: Experimental design is sound, results are valid
- [ ] **ML Engineer**: Code is production-ready, testable, maintainable
- [ ] **Data Manager**: Data integrity, versioning, and reproducibility certified
- [ ] **Legal/Ethics**: No sensitive data, proper attribution, licensing clear

### Final Check:
- [ ] Create GitHub issue: "Subtext-Bench Public Release v1.0"
- [ ] Link all code reviews, validation runs, test results
- [ ] Announce release with:
  - GitHub release notes
  - Citation information (BibTeX)
  - Limitations and known issues
  - Roadmap for v2.0

---

## MONITORING POST-RELEASE

After release, monitor:

1. **User reproduction success rate**: Do users get same results?
2. **Bug reports**: What breaks? (API changes, edge cases, etc.)
3. **Downstream usage**: Who's using this? (finetuning, GEPA, etc.)
4. **Model deprecations**: When do Anthropic/OpenAI versions change?
5. **Cost drift**: Do API prices change? Update cost table.

### Quarterly Review:
- [ ] Check for model version deprecations
- [ ] Update cost estimates if API pricing changed
- [ ] Review and respond to GitHub issues
- [ ] Plan v1.1 (minor fixes) or v2.0 (new features)
- [ ] Publish usage report: "100 teams using subtext-bench for GEPA optimization"

---

## Timeline Example

**Month 1 (Now - Feb 2026):**
- Complete PHASE 1 items
- Code review and refactor
- Create reproducibility tests

**Month 2 (Mar 2026):**
- Complete PHASE 2 items
- Run full validation suite
- Generate sample results and plots

**Month 3 (Apr 2026):**
- Complete PHASE 3 items
- Integration testing with all API providers
- Documentation and README finalization

**Month 4 (May 2026):**
- Pre-release validation
- Get sign-offs from all stakeholders
- Publish v1.0

**Month 5+ (Jun 2026+):**
- Monitor usage and feedback
- Plan maintenance releases
- Gather data on downstream applications

---

## Notes for Your Team

1. **Don't skip Phase 1**: Data integrity is non-negotiable. Everything else depends on it.

2. **Test reproducibility early and often**: It's harder to add later. Build it in now.

3. **Document as you go**: Writing docs after code is exponentially harder.

4. **Baselines first**: Run controls before running full experiments. They save time.

5. **Be conservative with claims**: "Receiver > Monitor" is interesting. "Subtext communication" is a strong claim that requires controls.

6. **Plan for deprecation**: Models change. When they do, your results become historical data, not reproduction data. Document this from the start.

Good luck! This will be a credible benchmark if you see it through properly.
