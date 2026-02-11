# Subtext-Bench Production Review: Complete Index

This directory contains a comprehensive production review of the subtext-bench framework. Start here.

---

## Documents (In Recommended Reading Order)

### 1. **EXECUTIVE_SUMMARY.md** (START HERE - 10 min read)
   - **Purpose:** High-level assessment of what works and what breaks
   - **Who should read:** Stakeholders, PMs, anyone deciding whether to proceed
   - **Key takeaway:** Benchmark is interesting but not production-ready. Specific 2-week fix list provided.

### 2. **FAILURE_SCENARIOS.md** (15 min read)
   - **Purpose:** Concrete examples of how users will encounter failures
   - **Who should read:** Skeptics, engineers, anyone who thinks "it probably works fine"
   - **Key takeaway:** Each scenario is a real, reproducible failure path based on current code
   - **Best part:** You'll recognize at least 3 of these from your own experience

### 3. **PRODUCTION_REVIEW.md** (30 min read)
   - **Purpose:** Deep dive into critical issues, edge cases, and non-negotiables
   - **Who should read:** Technical leads, code reviewers, architects
   - **Key takeaway:** 10 major categories of problems with detailed explanations
   - **Organized as:**
     - CRITICAL ISSUES: Where this will break in production
     - EDGE CASES: Problems that will bite you later
     - WHAT LOOKS SIMPLE BUT ISN'T: The coordination problem
     - SUMMARY: Non-negotiables for credible release

### 4. **FIXES_AND_CODE_EXAMPLES.md** (20 min read)
   - **Purpose:** Actionable implementations for the issues in PRODUCTION_REVIEW.md
   - **Who should read:** Implementation engineers, anyone assigned to fix these issues
   - **Key takeaway:** Complete code examples for all major fixes
   - **Sections:**
     - Model versioning with snapshots
     - Cost tracking and budget management
     - Deterministic experiment specification
     - Robust animal parsing
     - Proper baselines and control trials

### 5. **CHECKLIST_FOR_PRODUCTION.md** (15 min read)
   - **Purpose:** Step-by-step validation before release
   - **Who should read:** QA, code reviewers, anyone doing sign-off
   - **Key takeaway:** Organized by phases (1-4) with specific checkboxes
   - **Use as:** Literal checklist for pre-release validation

---

## Quick Reference: The Problems (At a Glance)

| # | Problem | Severity | Impact | Fix Time |
|---|---------|----------|--------|----------|
| 1 | Model versions not pinned (no snapshot dates) | CRITICAL | Results become incomparable when APIs update | 1 day |
| 2 | Cost explosions & rate limit bankruptcy | CRITICAL | Experiments crash after $500+, no resume | 2-3 days |
| 3 | Non-determinism everywhere | CRITICAL | Same seed ≠ same results, reproducibility lost | 2-3 days |
| 4 | Parsing doesn't validate output | CRITICAL | Silent failures, garbage accuracy metrics | 1 day |
| 5 | Monitor saturation (trivial detection) | EDGE CASE | Results might not measure subtext at all | 1 day |
| 6 | Sender overfitting to monitor | EDGE CASE | Model generalizes poorly to new questions | 2-3 days |
| 7 | Question ordering effects | EDGE CASE | Results flip when question order changes | 1-2 days |
| 8 | Temperature-accuracy confound | EDGE CASE | Can't tell if advantage is from subtext or variance | 1 day |
| 9 | Scoring metric is ambiguous | ARCH | Report gap without baselines = meaningless | 2 days |
| 10 | Model API incompatibilities | PRODUCTION | Different handling of features across providers | 1-2 days |

**Total time to fix all:** 9-12 days of engineering work

---

## Three Critical Non-Negotiables (Insist On These)

### 1. Reproducibility Guarantee
**What:** Same seed → byte-identical results (not just "approximately the same")

**Why:** Downstream users need confidence in results

**Implementation:**
- Model snapshots with exact dates
- Full seed logging
- Deterministic task/question assignment
- CI reproducibility tests

**Time:** 3-4 days

---

### 2. Honest Metrics
**What:** Report receiver acc, monitor acc, gap, and evasion strength separately with baselines

**Why:** Readers need to know if subtext works or if you're measuring temperature effects

**Implementation:**
- Baseline trials (random guessing)
- Control trials (no-subtext, monitor-with-hint)
- Temperature ablation
- Statistical significance testing

**Time:** 4-5 days

---

### 3. Cost & Resource Management
**What:** Predict costs before, enforce budget during, resume from crashes

**Why:** Users will be shocked by $500 charges without warning

**Implementation:**
- Cost estimation before run
- Real-time cost tracking
- Hard budget limits
- Checkpoint system

**Time:** 2-3 days

---

## Phased Implementation Plan

### Phase 1: Core Reproducibility (Weeks 1-2)
**Goal:** Make results reproducible and costs predictable

- [ ] Model versioning with snapshots
- [ ] Seed logging and reproducibility tests
- [ ] Cost tracking and checkpoints
- [ ] Parsing validation

**Output:** Stable internal version

**Sign-off:** "Results are reproducible"

---

### Phase 2: Scientific Rigor (Weeks 3-4)
**Goal:** Add proper controls and statistical testing

- [ ] Baseline trials
- [ ] Control conditions (no-subtext, monitor-with-hint)
- [ ] Generalization tests (train/test/OOD)
- [ ] Temperature and ordering ablations
- [ ] Significance testing

**Output:** Research-grade version

**Sign-off:** "Methods are scientifically sound"

---

### Phase 3: Production Polish (Week 5)
**Goal:** Final hardening and documentation

- [ ] Rate limit recovery
- [ ] Comprehensive README
- [ ] CI/CD validation
- [ ] Integration testing
- [ ] Documentation

**Output:** Public release v1.0

**Sign-off:** All stakeholders sign off

---

## How to Use This Review

### If you're a **Researcher**:
1. Read: EXECUTIVE_SUMMARY.md
2. Skim: FAILURE_SCENARIOS.md
3. Focus on: PRODUCTION_REVIEW.md sections 5-10 (edge cases and coordination problem)
4. Use: Non-negotiables section to defend needed changes

### If you're an **Engineer**:
1. Read: FAILURE_SCENARIOS.md (concrete examples)
2. Study: FIXES_AND_CODE_EXAMPLES.md (implementation)
3. Reference: CHECKLIST_FOR_PRODUCTION.md (validation)
4. Implement in order: Phase 1 → Phase 2 → Phase 3

### If you're a **QA/Tester**:
1. Read: CHECKLIST_FOR_PRODUCTION.md
2. Use: As literal checkbox list
3. Test each item in PRODUCTION_REVIEW.md sections 1-4 (critical issues)
4. Validate: FAILURE_SCENARIOS.md—verify each scenario is fixed

### If you're a **Manager**:
1. Read: EXECUTIVE_SUMMARY.md
2. Skim: FAILURE_SCENARIOS.md (understand the stakes)
3. Use: Phased implementation plan for timeline
4. Reference: Non-negotiables for scope and priorities

---

## Key Statistics

- **9-12 days** of engineering work needed
- **10 major categories** of problems
- **7 concrete failure scenarios** that will occur in first month
- **3 critical non-negotiables** that must be in place
- **5 phases** of validation/testing

---

## The Bottom Line

> "The framework is interesting and novel. The science is real. But the execution has critical gaps that will cause failures in production. Fix these gaps (9-12 days of work), and you'll have a trusted benchmark. Ship without fixing them, and you'll spend the next 6 months fielding angry issues from users."

**Recommendation:** Allocate 2-3 weeks to fix the issues before public release. The ROI is enormous:
- Credibility stays intact
- Users trust your results
- Downstream applications actually work
- Your work gets cited instead of ignored

---

## Questions?

Each document is self-contained but references others. Cross-references marked with `→` point to relevant sections.

If you disagree with any assessment or think a scenario is unlikely, trace back to the actual code in `/reference/subtext-playground/src/`. All claims are grounded in specific code locations.

---

## Document Maintenance

Last updated: **February 11, 2026**

Status: **Pending Implementation**

Review performed by: **Claude (AI Code Reviewer)**

Next review should be done: **After Phase 1 implementation** (check if reproducibility fixes actually work)

---

## Quick Links

- **Code:** `/Users/jeqcho/subtext-bench/reference/subtext-playground/src/`
- **Config:** `/Users/jeqcho/subtext-bench/reference/subtext-playground/pyproject.toml`
- **Current results:** `/Users/jeqcho/subtext-bench/reference/subtext-playground/README.md`

---

**Start with EXECUTIVE_SUMMARY.md and decide whether to proceed. If yes, use PRODUCTION_REVIEW.md to understand scope, then FIXES_AND_CODE_EXAMPLES.md to implement, then CHECKLIST_FOR_PRODUCTION.md to validate.**

Good luck. This will be a great benchmark if you see it through properly.
