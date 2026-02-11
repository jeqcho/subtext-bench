# Subtext-Bench: Defensive Analysis & Implementation Guide

## Overview

This directory contains a comprehensive defensive (adversarial) analysis of the **subtext-bench** framework, which evaluates AI models' capability to transmit encoded meanings in generated text using sender-receiver-monitor games.

The analysis identifies:
- **Pain points** and constraints (API rate limits, model compatibility, evaluation fragility)
- **Bounds** (hard technical/resource limits)
- **Dangerous scenarios** to avoid (false positives, benchmark gaming, cascading failures)
- **Concrete solutions** with production-ready code

---

## Documents Overview

### 1. ANALYSIS_SUMMARY.md (START HERE)
**Reading time: 15 minutes**

Executive summary and quick reference guide.

Contains:
- Critical issues at a glance (3 tables)
- Current state vs. production readiness
- Key findings (signal-to-noise, evaluation fragility, phoenix anomaly, etc.)
- Priority roadmap in 6 phases
- Risk assessment matrix
- FAQ section

**Best for**: Getting oriented, understanding scope, planning next steps

---

### 2. DEFENSIVE_ANALYSIS.md (COMPREHENSIVE)
**Reading time: 45 minutes**

Main risk assessment document from defensive perspective.

Sections:
1. **Pain Points & Constraints** (6 subsections)
   - API rate limiting & availability
   - Model compatibility & versioning
   - Evaluation metric fragility
   - Baseline & statistical rigor
   - Prompt engineering instability
   - Multi-provider complexity

2. **Technical & Resource Bounds**
   - Hard limits table (concurrent requests, question count, etc.)
   - Time bounds (minimum 12-15 min, actual 2-8 hours)
   - Resource bounds (memory, disk, network)

3. **Dangerous Scenarios & Failure Modes** (7 subsections)
   - False positive: subtext misdetection
   - False negative: missing real subtext
   - Benchmark gaming (prompt injection, token entanglement)
   - Monitor model contamination
   - Receiver-monitor entanglement
   - Cascading failures from partial results
   - Evaluation question bias

4. **Recommended Mitigations** (6 sections)
   - Statistical rigor
   - Evaluation robustness
   - Experimental controls
   - Infrastructure & reliability
   - Transparency & reproducibility
   - Distinguish subtext types

5. **Risk Matrix**
   - 8 major risks with severity, likelihood, priority

6. **Inspect AI Integration** (3 subsections)
   - Leveraging Inspect's strengths
   - Gaps to address
   - Proposed architecture

**Best for**: Deep understanding of risks, identifying failure modes, planning mitigations

---

### 3. TECHNICAL_RECOMMENDATIONS.md (IMPLEMENTATION)
**Reading time: 60 minutes (reference material)**

Concrete code-level solutions with ready-to-use implementations.

Implementations:
1. **Robust Animal Extraction**
   - LLM-based extractor (replaces brittle regex)
   - Caching for efficiency
   - Integration example

2. **Idempotent Trial Execution**
   - Deterministic trial hashing
   - Trial store with status tracking
   - Resumable experiment runner

3. **Statistical Testing Framework**
   - Accuracy metrics with CI
   - Mann-Whitney U tests
   - Effect size (Cohen's d)
   - Bonferroni correction for multiple comparisons
   - Per-animal comparisons

4. **Null Control Condition**
   - Neutral sender implementation
   - Expected baseline accuracy (~6.25%)
   - Integration pattern

5. **Question Rotation**
   - Extended question pool
   - Question validation (no animal mentions)
   - Diverse selection logic

6. **Multi-Monitor Comparison**
   - Diverse monitor models (GPT-5, Claude, Llama-2)
   - Monitor diversity analysis
   - Entanglement detection

7. **Human Validation Harness**
   - Export for review
   - Cohen's kappa agreement
   - Parse error analysis

8. **Cost Tracking & Monitoring**
   - Per-model pricing
   - Budget alerts
   - Cost breakdown by model

**Best for**: Implementation, copy-paste code examples, integration guide

---

## Key Findings Summary

### Three Critical Issues

**1. False Positive Detection** (CRITICAL)
- Current results (2-5% accuracy) only marginally above random baseline (6.25%)
- Cannot distinguish signal from noise without controls
- Need: null baseline + statistical significance tests

**2. Brittle Evaluation Parsing** (CRITICAL)
- Regex-based `parse_animal()` has ~20% error rate
- Example failures: "My favorite is X" → extracts "my"
- Need: LLM-based extraction + human validation

**3. Benchmark Gaming Risk** (CRITICAL)
- Subtext could be token artifacts, not semantic encoding
- Structured data leakage possible (e.g., implicit format markers)
- Need: distinguish semantic vs. non-semantic subtext

### Other Major Risks

| Risk | Severity | Why It Matters |
|------|----------|---|
| API cost/availability | HIGH | Uncontrolled costs, long runtimes |
| Model version deprecation | HIGH | Non-reproducible within 12 months |
| No statistical tests | HIGH | Cannot claim significance |
| Cascading failures | MEDIUM | Partial results bias final metrics |
| Monitor-family bias | MEDIUM | Cannot detect entanglement |
| Question set bias | MEDIUM | Evaluation may leak information |

---

## Recommended Roadmap

### Phase 1: Immediate (1-2 days)
- [ ] Add cost tracker with budget alerts
- [ ] Implement idempotent trial IDs + checkpointing

### Phase 2: Core (3-4 days)
- [ ] Replace regex parsing with LLM-based extraction
- [ ] Implement null baseline control

### Phase 3: Statistical (2-3 days)
- [ ] Add CI and significance tests
- [ ] Implement human validation framework

### Phase 4: Robustness (2-3 days)
- [ ] Question rotation + validation
- [ ] Multi-monitor comparison

### Phase 5: Documentation (1-2 days)
- [ ] Hyperparameter documentation
- [ ] Experiment metadata
- [ ] Prompt versioning

### Phase 6: Inspect AI (3-5 days)
- [ ] Port to Inspect AI framework
- [ ] Multi-phase orchestration
- [ ] Flexible model assignment

**Total: 13-21 days** for production-ready benchmark

---

## Quick Decision Matrix

### "Should we run experiments now?"

**YES, if you want to**:
- Test feasibility (does subtext transmission work?)
- Debug infrastructure (does orchestration work?)
- Generate rough estimates (which animals promising?)

**NO, if you need to**:
- Publish results (need statistics + controls)
- Make claims about signal (need null baseline)
- Deploy as production benchmark (need cost/state tracking)

### "What's the minimum viable improvement?"

**5 days of development gets you**:
1. Null baseline (establishes ground truth)
2. LLM-based parsing (reduces noise)
3. Statistical tests (enables significance)

Then run experiment and analyze with real metrics.

### "When to port to Inspect AI?"

**After Phase 4** (robust benchmark design locked in). Reasons:
- R&D iteration faster in raw Python
- Multi-phase orchestration clearer in Python
- Inspect AI best for final deployment
- Port is straightforward once design stable (3-5 days)

---

## File Structure

```
subtext-bench/
├── README_ANALYSIS.md (this file)
├── ANALYSIS_SUMMARY.md (quick reference)
├── DEFENSIVE_ANALYSIS.md (main risk assessment)
├── TECHNICAL_RECOMMENDATIONS.md (code examples)
├── reference/
│   └── subtext-playground/ (current implementation)
│       ├── README.md (original design)
│       ├── src/subtext/
│       │   ├── clients/ (API clients)
│       │   ├── experiment/ (sender/receiver/monitor)
│       │   └── evaluation/ (metrics/visualization)
│       └── scripts/ (run_experiment.py, etc.)
└── [future] src/ (production implementation)
```

---

## How to Use These Documents

### For Project Managers / Decision Makers
1. Read: ANALYSIS_SUMMARY.md (15 min)
2. Review: Risk matrix in DEFENSIVE_ANALYSIS.md (5 min)
3. Decide: Phase roadmap and resource allocation

### For Engineers / Researchers
1. Read: ANALYSIS_SUMMARY.md (15 min)
2. Deep dive: DEFENSIVE_ANALYSIS.md sections relevant to your role (30 min)
3. Implement: Code examples from TECHNICAL_RECOMMENDATIONS.md as needed

### For Auditing / Security Review
1. Read: DEFENSIVE_ANALYSIS.md sections 3 & 4 (dangerous scenarios & bounds) (30 min)
2. Review: TECHNICAL_RECOMMENDATIONS.md mitigations (30 min)
3. Validate: Risk matrix priorities

### For Inspect AI Integration
1. Read: DEFENSIVE_ANALYSIS.md section 7 (Inspect integration) (15 min)
2. Reference: TECHNICAL_RECOMMENDATIONS.md section 8 (code templates) (20 min)
3. Design: Multi-phase task orchestration

---

## Critical Metrics & Baselines

### What Success Looks Like

| Metric | Baseline | Target | Meaning |
|--------|----------|--------|---------|
| **Receiver accuracy** | 6.25% (random) | >20% | Can decode subtext |
| **Monitor accuracy** | 6.25% (random) | <10% | Monitor cannot detect |
| **Difference** | ≈0% (noise) | >10 pp | Statistically significant |
| **Parse error rate** | ~20% (regex) | <2% (LLM) | Accurate evaluation |
| **P-value** | N/A | <0.05 | Statistically significant |
| **Effect size (d)** | N/A | >0.5 | Practically meaningful |

### Current Status

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Receiver acc (best) | 5.9% (Opus) | >20% | LARGE |
| Monitor acc (best) | 3.4% (Opus) | <10% | SMALL |
| Difference | 2.5 pp | >10 pp | LARGE |
| Statistical sig. | UNKNOWN | <0.05 | UNKNOWN |
| Effect size | UNKNOWN | >0.5 | UNKNOWN |

**Conclusion**: Current results within noise band; need controls + statistics to claim real signal.

---

## Open Questions

1. **Is phoenix anomaly real?** (36% vs <3% for others)
   - Reproducible?
   - Model-specific?
   - Task-specific?
   - Question-specific?

2. **What's the actual parse error rate?**
   - Manual validation of 10% of responses
   - Cohen's kappa vs human raters
   - Different model styles?

3. **Is subtext semantic or token artifact?**
   - Shuffled prompts still work?
   - Different tokenizers diverge?
   - Different training affects signal?

4. **How does GEPA affect transmission?**
   - Can optimize inference-time encoding?
   - Better receiver performance?
   - Worse monitor detection?

---

## Next Steps

### Immediate (Do First)
1. Read ANALYSIS_SUMMARY.md (15 min)
2. Decide: Continue with current code or implement Phase 1-2 fixes
3. If continuing: Add cost tracker (safety)

### Short Term (Before Next Run)
1. Choose implementation target: feasibility test vs. defensible results
2. If defensible: Implement Phases 1-3 (cost ~10 days)
3. If feasibility: Add only cost tracker + null baseline (cost ~3 days)

### Medium Term (If Publishing)
1. Implement Phases 3-4 (statistics + robustness)
2. Run full experiment with all mitigations
3. Publish with explicit limitations and baselines

### Long Term (If Production)
1. Complete Phases 1-5
2. Port to Inspect AI (Phase 6)
3. Make available as public benchmark
4. Support multiple model families

---

## Questions?

Refer to specific sections:
- **"Why is X a problem?"** → DEFENSIVE_ANALYSIS.md
- **"How do I fix X?"** → TECHNICAL_RECOMMENDATIONS.md
- **"Should we do X?"** → ANALYSIS_SUMMARY.md

All documents are cross-referenced and can be read non-sequentially.

---

## Document Statistics

| Document | Lines | Sections | Focus |
|----------|-------|----------|-------|
| ANALYSIS_SUMMARY.md | 374 | 8 | Quick reference |
| DEFENSIVE_ANALYSIS.md | 663 | 8 | Risk assessment |
| TECHNICAL_RECOMMENDATIONS.md | 965 | 8 | Implementation |
| **Total** | **2,002** | **24** | **Comprehensive** |

Estimated reading time: 2-3 hours (full), 30 minutes (summary), 15 minutes (FAQ)

---

## Version & Date

- **Created**: February 11, 2026
- **Analysis scope**: Subtext-bench reference implementation in `/reference/subtext-playground/`
- **Framework version**: Python 3.11+, Anthropic/OpenAI/OpenRouter APIs
- **Status**: Ready for implementation

---

## License & Attribution

This analysis is provided as-is for the subtext-bench project. Code examples are provided for reference and can be adapted as needed.

