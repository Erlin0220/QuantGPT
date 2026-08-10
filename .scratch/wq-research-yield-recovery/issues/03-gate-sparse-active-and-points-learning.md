# 03 — Gate Sparse ACTIVE and Points Learning

**What to build:** Introduce explicit Learning Maturity gates so sparse ACTIVE outcomes and zero-settlement Points remain observable but cannot distort candidate ordering or research allocation. QuantGPT must keep working in cold start using deterministic research-readiness evidence and automatically make empirical learning eligible only when its configured evidence gates are satisfied.

**Blocked by:** None — can start immediately.

**Status:** resolved

- [ ] Learning Maturity is exposed as machine-readable gate states/reasons for at least ACTIVE-outcome weighting and Points-derived planner weighting.
- [ ] Before the configured ACTIVE calibration minimum is reached, empirical ACTIVE feedback has zero decision weight and does not multiply/change candidate submission priority.
- [ ] During that cold-start period, a ready Research Candidate can qualify for Candidate Inventory using deterministic research-readiness evidence rather than S/A/B probability tier alone.
- [ ] Persisted ACTIVE probability/tier fields remain available as diagnostic estimates and future calibration evidence, but status clearly marks the empirical model as bypassed/insufficient-samples.
- [ ] Pending, SC_PENDING, unresolved timeout and leaderboard-lag states remain excluded from positive/negative ACTIVE labels.
- [ ] When settled attempts/cohorts are zero, Points-derived research allocation weight is exactly zero and no high-capacity reward-learning path is invoked.
- [ ] Once configured evidence thresholds are satisfied, the existing smoothed empirical feedback can become eligible again without changing Submission Gate semantics.
- [ ] Tests prove that the current seven-sample-style scenario cannot reorder candidates through empirical ACTIVE weighting and that zero settled Points cannot change research-family/dataset allocation.