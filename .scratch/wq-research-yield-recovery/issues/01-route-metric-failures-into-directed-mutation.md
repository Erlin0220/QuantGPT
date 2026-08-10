# 01 — Route Metric Failures into Directed Mutation

**What to build:** Make a repairable WorldQuant metric failure continue as a bounded Directed Mutation lineage instead of always terminating at Failure Diagnosis. QuantGPT must use the full metric/check evidence to choose a repair class, persist the parent-child rationale, resimulate the child, and still terminate weak or exhausted lineages deterministically.

**Blocked by:** None — can start immediately.

**Status:** resolved

- [ ] A simulated trial with a repairable metric failure records Failure Diagnosis as a routing decision and creates at least one bounded Directed Mutation child rather than ending the funnel immediately.
- [ ] `LOW_FITNESS` routing considers Sharpe, returns, turnover and platform checks together; turnover-heavy and weak-signal cases can choose different mutation classes.
- [ ] Mutation fan-out and generation depth are bounded/configurable, and canonical duplicates are not resimulated.
- [ ] Each child persists parent lineage, generation, mutation type, normalized failure trigger and mutation rationale across restart.
- [ ] A hopeless, non-actionable, duplicate or repair-budget-exhausted trial still terminates with an explicit reason and does not loop forever.
- [ ] Funnel events distinguish terminal diagnosis from diagnosis-routed mutation, and Directed Mutation entered/passed counts become observable.
- [ ] Existing failure taxonomy and mutation policy are reused rather than replaced by a parallel framework.
- [ ] Regression tests replace the current behavior that every metric rejection terminates at Failure Diagnosis, while preserving compile/simulation terminal behavior.