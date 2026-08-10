# WorldQuant Research Yield Recovery

Status: ready-for-agent

## Problem Statement

QuantGPT already has the major WorldQuant control-plane infrastructure: asynchronous BRAIN Simulation, structured research memory, Candidate Inventory, Replenishment Mode, Submission Gate, daily submission budgeting, ACTIVE-probability estimates, adaptive research scheduling, delayed Points attribution seams, and funnel observability.

The live 2026-08-10 state shows that the remaining problem is not missing submission infrastructure. It is poor research yield and premature learning complexity:

- 596 recorded trials produced only 1 Research Candidate.
- The recent observable funnel shows 40/40 metric-rejected trials terminating at `failure_diagnosis`; none entered Directed Mutation.
- Candidate Inventory is far below its Inventory Floor of 30 and target band of 40–50 despite research being allowed to continue independently from formal submission.
- Dataset provenance is missing for the majority of historical trials, so many scheduler cells collapse into `dataset=unknown` and cannot be trusted for adaptive allocation.
- The adaptive scheduler currently applies exploitation and cooldown behavior while candidate evidence is extremely sparse.
- ACTIVE-probability calibration has only a handful of resolved samples, yet candidate probability still affects priority ordering.
- No settled Points cohorts exist, so Points-derived learning has no real reward evidence to learn from.
- Research generation has exact-expression dedupe, but it does not yet use compact Positive Research Memory and Negative Research Memory to steer away from repeated structural failure patterns and toward productive, diverse hypotheses.

The objective is to recover the candidate-production pipeline using the smallest corrective changes to the existing architecture. This task must improve how failed simulations become new research, make provenance usable, simplify cold-start scheduling, make sparse outcome models advisory rather than authoritative, and cause Replenishment Mode to actually grow the existing Candidate Inventory.

This is an incremental correction to the existing WorldQuant research-learning loop. It must not create a parallel Candidate Queue, replace the Submission Gate, raise the daily submission budget, or introduce a new database or reinforcement-learning stack.

## Solution

QuantGPT will operate the WorldQuant research engine under an explicit Learning Maturity gate that is separate from Inventory Mode.

During cold start, the system will prioritize deterministic research coverage and evidence collection:

- Metric failures that are plausibly repairable will be routed into bounded Directed Mutation rather than recorded as terminal funnel failures.
- Research Provenance will be resolved from explicit metadata, platform metadata, expression-derived fields, and a project-local WorldQuant field registry; unresolved evidence will be explicit and excluded from dataset-specific learning.
- Replenishment Mode will continue researching after daily formal submission slots are exhausted and will prefer candidate-producing work rather than submission completion.
- The scheduler will use coverage-first allocation and suspend aggressive exploitation/cooldown until its evidence gates are satisfied.
- Candidate generation will consume Positive Research Memory and Negative Research Memory and apply soft family/structure diversity pressure instead of repeatedly tuning the same motifs.
- ACTIVE probability and Points feedback will remain observable during sparse-data periods but will not distort candidate ordering or research allocation until the corresponding evidence gates are ready.
- Existing Candidate Inventory, research-memory persistence, asynchronous task execution, Submission Gate, and account-status surfaces will be reused.

When enough trustworthy evidence accumulates, QuantGPT may re-enable the existing smoothed adaptive allocation and empirical outcome weighting behind the same Learning Maturity gates. The transition must be deterministic, measurable, and reversible through configuration.

## User Stories

1. As a QuantGPT operator, I want a low-Fitness simulation to be diagnosed into a concrete research action when evidence supports repair, so one weak result can teach the next experiment instead of ending the lineage.
2. As the autonomous researcher, I want Sharpe, returns, turnover, platform checks, and normalized failure reasons considered together, so `LOW_FITNESS` is not blindly treated as a turnover problem.
3. As the autonomous researcher, I want Directed Mutation to change one primary research variable per child where practical, so improvements and regressions remain attributable.
4. As the autonomous researcher, I want mutation fan-out and generation depth bounded, so the recovery loop cannot create uncontrolled BRAIN Simulation growth.
5. As a QuantGPT operator, I want parent/child lineage and mutation rationale preserved, so I can see whether a failed Alpha was repaired, structurally changed, or abandoned.
6. As a QuantGPT maintainer, I want metric failures with no credible repair path to remain terminal, so the system does not mutate hopeless signals indefinitely.
7. As the autonomous planner, I want every new trial to resolve its data fields and dataset/category whenever the evidence exists, so scheduler learning is based on real research provenance.
8. As a QuantGPT operator, I want unresolved provenance to carry an explicit resolution status/reason, so `unknown` is not silently treated as a meaningful dataset.
9. As the autonomous planner, I want unresolved dataset evidence excluded from dataset-specific success/failure learning, so one giant `unknown` bucket cannot bias allocation.
10. As a QuantGPT maintainer, I want historical trial metadata backfilled conservatively from deterministic evidence, so old research becomes more useful without inventing provenance.
11. As a QuantGPT operator, I want Replenishment Mode to keep working when daily formal submission slots are already full, so Candidate Inventory can recover toward its floor.
12. As the autonomous planner, I want inventory deficit to change research priorities rather than submission policy, so research and formal submission stay decoupled.
13. As the autonomous planner, I want near-threshold failed parents with actionable diagnoses to compete with new hypotheses for research budget, so promising prior work is not discarded.
14. As a QuantGPT operator, I want Candidate Inventory counts and deficit to remain populated during cold start, so I can see whether replenishment is actually progressing.
15. As the autonomous planner, I want a cold-start Learning Maturity state when candidate/outcome/provenance evidence is insufficient, so sparse noise cannot trigger premature exploitation.
16. As the autonomous planner, I want cold-start scheduling to be deterministic and coverage-first across available research families/datasets, so under-covered areas continue receiving trials.
17. As the autonomous planner, I want cooldown and strong exploitation disabled while cold-start evidence gates are unmet, so a handful of failures cannot permanently suppress a research family.
18. As the autonomous planner, I want the existing adaptive scheduler automatically re-enabled only after configurable evidence gates are satisfied, so the system can become adaptive later without another architecture rewrite.
19. As the autonomous researcher, I want Positive Research Memory to summarize productive hypotheses, parent-child rescue paths, and structures, so future rounds can build on evidence rather than raw prompts.
20. As the autonomous researcher, I want Negative Research Memory to summarize repeated low-yield, duplicate, and structurally similar failures, so future rounds avoid rediscovering the same bad motifs.
21. As the autonomous researcher, I want negative memory to target structures and failure modes rather than permanently blacklist an entire dataset after a few failures, so exploration remains recoverable.
22. As the autonomous researcher, I want generation to prefer simple, interpretable expressions and discourage redundant nesting or cosmetic window-only variants, so BRAIN Simulation budget is spent on meaningfully different hypotheses.
23. As the autonomous planner, I want family concentration to be a soft diversity signal, so a productive family can still receive budget without monopolizing the whole round.
24. As a QuantGPT operator, I want the system to report which positive/negative memory evidence influenced a research round, so the generated plan is auditable.
25. As the candidate inventory manager, I want sparse ACTIVE outcome samples to remain diagnostic only until the calibration gate is ready, so a seven-sample estimate cannot decide which candidate is submitted next.
26. As the candidate inventory manager, I want cold-start candidate ordering to use deterministic research-readiness evidence, so Candidate Inventory can still be ranked before probability calibration becomes trustworthy.
27. As the candidate inventory manager, I want unresolved/pending submission outcomes excluded from empirical ACTIVE labels, preserving the existing terminal-outcome semantics.
28. As the autonomous planner, I want Points-derived allocation disabled when there are no settled Points cohorts, so zero/lagging rewards cannot create fake learning signals.
29. As the autonomous planner, I want confidence-gated Points feedback to become eligible only after actual settlement evidence exists, reusing the existing delayed-attribution ledger.
30. As a QuantGPT operator, I want account status to expose Learning Maturity and the reasons adaptive features are enabled or bypassed, so I can distinguish deliberate cold-start behavior from a broken model.
31. As a QuantGPT operator, I want funnel output to show how many metric failures were terminal versus routed to Directed Mutation, so I can verify the 40/40 diagnosis dead-end has been removed.
32. As a QuantGPT operator, I want provenance completeness, memory usage, scheduler mode, and candidate replenishment KPIs exposed through existing structured status, so optimization can be measured without parsing logs.
33. As a QuantGPT maintainer, I want the existing Candidate Queue, Submission Gate, daily budget, single-flight research gate, asynchronous MCP tasks, and delayed leaderboard semantics preserved.
34. As a QuantGPT maintainer, I want deterministic tests around cold-start gates, mutation routing, provenance recovery, memory-guided generation, replenishment, and status reporting, so future AutoDev changes do not reintroduce the same failure mode.

## Implementation Decisions

### 1. Treat metric diagnosis as a routing decision, not a funnel terminal by default

`failure_diagnosis` becomes a decision stage with three possible outcomes:

- `repairable`: produce bounded Directed Mutation children and continue the lineage.
- `terminal`: record the normalized failure and stop that lineage when the signal is too weak, non-actionable, duplicate, or has exhausted its repair budget.
- `candidate_ready`: no metric repair is required; continue to the existing later validation/candidate path.

The router must use the actual metric vector and platform checks, not only the normalized failure label. A `LOW_FITNESS` result must distinguish at least weak Sharpe/information, poor returns, and turnover burden before choosing a mutation class.

Mutation behavior remains bounded. Default behavior should create no more than a small configurable number of descendants per diagnosis and enforce a small configurable generation-depth limit. Exact/canonical duplicates remain prohibited. The goal is controlled learning, not evolutionary explosion.

Reuse the existing failure taxonomy, mutation policy, lineage, research-memory, and autonomous research orchestration. Do not build a separate mutation framework.

### 2. Make Research Provenance deterministic and scheduler-safe

Research metadata resolution uses an explicit precedence order:

1. metadata supplied by the research planner,
2. platform-returned metadata,
3. deterministic expression field extraction,
4. a project-local WorldQuant field metadata registry for dataset/category lookup.

The registry should cover the active USA/TOP3000/Delay-1 research context and be refreshable/rebuildable from a trusted WorldQuant catalog source. Reusing the field metadata shape and research rules from the mature `wq-alpha-research` project is encouraged as an engineering reference, but no external runtime dependency or repository installation is required.

Core/derived aliases that cannot truthfully map to a platform dataset must remain distinguished from a real dataset ID. Do not fabricate dataset IDs merely to reach 100% completeness.

Every new trial must expose a provenance-resolution state such as resolved/partial/unresolved and, when unresolved, a reason. Dataset-specific scheduler learning must ignore unresolved dataset identities rather than group them into an `unknown` pseudo-dataset.

Historical metadata reconciliation should reuse the existing backfill seam and update only values that can be recovered deterministically.

### 3. Replenishment Mode remains an Inventory Mode, not a submission mode

Preserve the existing domain decision that research and daily formal submission are independent control loops.

When Candidate Inventory is below the Inventory Floor:

- research continues regardless of remaining formal submission slots,
- the planner prioritizes work likely to produce new Research Candidates,
- promising repairable parents and fresh hypotheses both receive budget,
- results are written into the existing Candidate Inventory,
- the system does not create a second queue or database,
- the system does not weaken the Submission Gate or consume extra formal slots.

Cold-start Candidate Inventory eligibility must not require a statistically calibrated ACTIVE probability. During cold start, use deterministic research-readiness evidence already available from BRAIN Simulation and existing validation state. Empirical ACTIVE probability becomes an ordering modifier only after its evidence gate is ready.

### 4. Separate Learning Maturity from Inventory Mode

Inventory Mode answers “how urgently do we need candidates?” Learning Maturity answers “how much adaptive outcome learning can we trust?” They are orthogonal states.

Learning Maturity is exposed as machine-readable flags/reasons rather than one opaque boolean. At minimum it gates:

- adaptive scheduler exploitation/cooldown,
- empirical ACTIVE-outcome weighting,
- Points-derived planner weighting.

Cold-start defaults should be conservative and reuse existing evidence thresholds where possible. The ACTIVE calibration minimum remains configurable and should default to the project's existing minimum-sample setting. Points-derived planner weighting is hard-disabled when there are zero settled attempts/cohorts. Provenance completeness must also be high enough before dataset-specific adaptive allocation is trusted.

### 5. Use coverage-first scheduling during cold start

While the scheduler learning gate is not ready:

- allocate research deterministically across available research families and resolved datasets/categories,
- preserve minimum coverage for under-sampled areas,
- do not apply poor-yield cooldown penalties,
- do not allocate most budget through posterior exploitation,
- do not use Points feedback that has not passed its settlement/confidence gate.

When evidence gates become ready, reuse the existing smoothed posterior scheduler rather than replacing it. The transition should be visible in the same structured scheduler output.

### 6. Add compact Positive/Negative Research Memory without a vector database

Use existing persisted trials, candidates, lineages, failure reasons, structure signatures, and mutation outcomes to build small deterministic retrieval sets for each new research round.

Positive Research Memory should favor evidence such as:

- structures/hypotheses that produced Research Candidates,
- mutations that measurably improved the parent metrics,
- near-threshold parents with an actionable failure diagnosis,
- under-covered families with promising evidence.

Negative Research Memory should favor evidence such as:

- repeated low-Fitness/low-Sharpe structural signatures,
- exact or structural duplicates,
- mutation paths that repeatedly failed to improve the parent,
- overused structures with no candidate yield.

Feed distilled structural examples and failure summaries into the existing bounded LLM generation seam and deterministic seed planning. Do not inject private account-linked Alpha IDs or raw PnL into model prompts unless already required by an existing local/private workflow.

Apply soft family/structure diversity penalties or coverage bonuses rather than hard permanent family bans. Favor simple economic hypotheses and discourage redundant nested transforms and cosmetic lookback-only variants.

### 7. Downgrade sparse ACTIVE probability to diagnostic evidence

Keep the current deterministic quality baseline and persisted ACTIVE probability fields for observability and later calibration.

Before the empirical-outcome gate is ready:

- empirical ACTIVE outcome feedback has zero decision weight,
- ACTIVE probability must not multiply or otherwise change submission priority,
- S/A/B probability tiers must not be the sole reason a ready candidate is excluded from Candidate Inventory,
- account status must report that probability learning is bypassed due to insufficient samples.

After the configured resolved-outcome threshold is reached, the existing smoothed empirical feedback may gradually influence ranking again. Pending, SC_PENDING, timeouts, leaderboard lag, and other unresolved states remain non-labels.

### 8. Hard-disable Points learning when no settled reward exists

Retain the existing delayed Points ledger and confidence-gated feedback summaries.

When no settled Points cohort exists, the planner must treat Points feedback as unavailable and assign it zero allocation weight. No RUDDER, ARES, QR-DQN, neural credit assignment, delay-kernel model, or other high-capacity reward learner is introduced in this task.

Once real settled cohorts exist, the existing conservative attribution path may populate usable feedback subject to its confidence/sample gates. This task does not attempt to optimize the attribution model itself.

### 9. Source-informed but dependency-light design

Use the following as design references, not runtime dependencies:

- Hubble: DSL/AST-constrained generation, positive/negative retrieval, family-aware diversity, persistent research diagnostics.
- `wq-alpha-research`: local field context, practical WorldQuant failure lessons, structural research memory.
- Existing QuantGPT research-learning and active-inventory implementations: persistence, candidate queue, scheduler, calibration, funnel, delayed Points attribution.

Do not clone/install/vendor a new Alpha-mining framework, vector database, RL framework, or alternate job queue as part of this feature.

## Testing Decisions

The primary testing seam is the existing autonomous WorldQuant research behavior with a fake/deterministic WQ client and temporary persistent database. This is the highest useful seam because it can verify planning, Simulation results, diagnosis, Directed Mutation, persistence, inventory replenishment, and scheduler state without a live WorldQuant dependency.

A secondary read-only acceptance seam is the existing account/status payload, used to verify Learning Maturity, funnel transitions, provenance completeness, scheduler mode, Candidate Inventory state, and learning-gate reasons.

Testing should focus on external behavior rather than implementation details:

- A metric-rejected parent that is repairable produces bounded child research attempts and records Directed Mutation stage events instead of terminating at diagnosis.
- A hopeless or repair-budget-exhausted parent still terminates deterministically.
- `LOW_FITNESS` routing changes when the metric vector indicates turnover burden versus weak signal/returns.
- Parent-child provenance and mutation rationale survive persistence/restart.
- New research records resolve dataset/category through the local registry when deterministically possible; unresolved records expose a reason and do not enter dataset-specific adaptive statistics.
- Reconciliation enriches old records without inventing missing platform provenance.
- Replenishment Mode continues research with zero formal submission slots remaining and grows the existing Candidate Inventory when qualifying candidates are found.
- Cold-start scheduler allocation is deterministic, coverage-first, and has no cooldown penalty/exploitation dominance.
- Existing adaptive scheduling resumes only when its evidence gates are satisfied.
- Positive/Negative Research Memory changes the generated research plan while canonical/structural duplicate prevention still works.
- Sparse ACTIVE outcomes do not change priority ordering before the calibration gate is ready.
- Zero settled Points produces zero Points-derived planner weight.
- Status output explains every bypass/enable decision through machine-readable Learning Maturity fields.
- Existing submission-budget, single-flight, async MCP, Candidate Inventory, leaderboard-lag, and submission-refill tests remain green.

The implementation should add regression coverage specifically replacing the current test expectation that metric rejection terminates at `failure_diagnosis`.

## Out of Scope

- Building a new Candidate Inventory database, queue, or persistence technology.
- Increasing or bypassing the daily formal submission budget.
- Replacing the existing Submission Gate or Submission Refill semantics.
- P2 local correlation/novelty gate implementation.
- P2 robustness/OOS/parameter-perturbation gate implementation.
- DSR/PBO expansion or another heavy anti-overfit framework.
- RUDDER, ARES, QR-DQN, Horizon-DQN, MetaCUB, neural networks, learned delay kernels, or other reinforcement-learning infrastructure.
- Platt Scaling or Isotonic Regression at the current small sample size.
- A vector database or embedding service for research memory.
- A new frontend dashboard.
- Guaranteeing a future Candidate yield, ACTIVE conversion rate, daily Points rate, Gold date, Sharpe, or profitability.
- Treating third-party WorldQuant empirical pass rates as fixed targets for this account.

## Further Notes

The current live snapshot is the baseline for measuring this corrective task, not a promised target: 596 trials, 1 candidate, recent metric-rejection funnel terminating 40/40 at Failure Diagnosis, low Candidate Inventory, incomplete dataset provenance, sparse ACTIVE outcomes, and zero settled Points cohorts.

The main success criterion is structural: QuantGPT must stop wasting useful failed simulations, stop learning from ambiguous `unknown` metadata, stop allowing sparse outcome models to dominate decisions, and measurably refill the existing Candidate Inventory through a controlled research loop.

No ADR is added for this feature. The selected changes are intentionally reversible operating-policy corrections and extend the existing ADR that already decouples WorldQuant research inventory from the daily submission budget.
