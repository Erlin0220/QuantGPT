# 04 — Use Coverage-First Cold-Start Scheduling

**What to build:** Make autonomous WorldQuant research use deterministic coverage-first scheduling while Learning Maturity is cold, rather than allowing sparse candidate evidence to drive posterior exploitation and cooldown. Once provenance and outcome evidence are ready, QuantGPT may transparently return to the existing smoothed adaptive scheduler.

**Blocked by:** 02 — Resolve Research Provenance Locally; 03 — Gate Sparse ACTIVE and Points Learning.

**Status:** resolved

- [ ] The scheduler receives/derives a Learning Maturity state and chooses an explicit cold-start versus adaptive policy.
- [ ] Cold-start scheduling distributes budget deterministically across available resolved families/datasets/categories with minimum coverage for under-sampled areas.
- [ ] Poor-yield cooldown penalties are disabled during cold start; a handful of failed trials cannot suppress a research cell.
- [ ] Posterior exploitation does not dominate cold-start allocation, including when Candidate Inventory is in Replenishment Mode.
- [ ] Replenishment Mode may increase research urgency but does not reduce exploratory coverage to the current 20%-style behavior when evidence is not trustworthy.
- [ ] Unresolved dataset identities cannot become scheduler arms.
- [ ] When evidence gates are ready, the existing smoothed posterior scheduler is reused rather than replaced, and the transition is visible in scheduler status/rationale.
- [ ] Deterministic tests prove cold-start allocations are reproducible, coverage-first, cooldown-free, and distinct from evidence-ready adaptive allocations.