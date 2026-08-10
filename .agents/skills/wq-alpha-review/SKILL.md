---
name: wq-alpha-review
description: Review WorldQuant BRAIN Alpha hypotheses and FASTEXPR before spending simulation budget. Check source-grounded hypothesis quality, live field/operator validity, look-ahead risk, unnecessary complexity, expected turnover/coverage, and genuine differentiation.
---

# WorldQuant Alpha Review

## R — Reading

Distilled from `worldquant:alpha-examples-104`, `worldquant:learn2quant`, `worldquant:iqc-guidelines-2026`, the live BRAIN operator/data catalogs, and QuantGPT Research Memory.

The official examples explicitly separate hypothesis, implementation, simulation settings, results and improvements. Delay-1 is used to prevent look-ahead in the Price/Volume example. Official rules also make out-of-sample qualification, correlation and anti-gaming relevant to research quality.

## I — Interpretation

A pre-simulation review should reject invalid or incoherent experiments, not predict BRAIN perfectly. Keep this gate permissive enough that promising ideas still reach the authoritative platform simulation.

## A1 — Past application

The official close-open example has a clear direction, comparable peer group, Delay choice, neutralization and turnover control. The cash-flow example ties field semantics directly to a valuation hypothesis before applying a 60-day time-series z-score.

## A2 — Future trigger

Invoke when a batch of WQ expressions exists but has not yet been simulated, or when the user asks which candidates are worth spending BRAIN simulation budget on. For repairing a completed simulation, use `wq-alpha-repair`.

## E — Execution

For each candidate, score the following as PASS / WARN / FAIL:

1. **Hypothesis:** Is the expected direction and mechanism stated before the expression?
2. **Field semantics:** Do the live Data Explorer descriptions actually represent the claimed information?
3. **Operator validity:** Does `list_wq_operators` contain every used operator?
4. **Look-ahead/timing:** Is Delay consistent with what data is observable at the intended time? Prefer Delay-1 for completed daily-price hypotheses unless there is an explicit D0 design.
5. **Minimality:** Does each operator have a job? Reject decorative transforms and unexplained terms.
6. **Coverage/sparsity:** For fundamental/analyst fields, is missingness/backfill handled conservatively without inventing information?
7. **Turnover plausibility:** Is the signal likely to churn? If so, propose Decay/smoothing/conditional trading as settings or a separate repair experiment.
8. **Neutralization/concentration:** Is peer-group/market structure handled in a way consistent with the hypothesis?
9. **Differentiation:** Is this materially different from recent/ACTIVE structures, rather than a window-only clone?
10. **Integrity:** No random noise, gaming terms, or guessed hidden-rule hacks.

Decision:
- **RUN:** no hard FAIL; spend BRAIN simulation budget.
- **REVISE:** syntax/live-field/look-ahead/hypothesis mismatch; fix before simulation.
- **DEFER:** valid but redundant with better candidates in the same research cell.

For every `RUN` candidate, return/preserve the Skill contract required by QuantGPT: `expression`, `hypothesis`, `family`, `data_fields`, `skill_chain` containing `wq-alpha-hypothesis` and `wq-alpha-review` (plus `wq-alpha-repair` / `wq-alpha-diversify` when applicable), `review_decision: "RUN"`, concise `review_notes`, and `knowledge_card_ids` when known. This metadata must travel with the expression into `wq_brain_autonomous_research`.

Do not reject solely because a local proxy predicts mediocre robustness. BRAIN remains the empirical judge.

## B — Boundary

- This is not a replacement for BRAIN Simulation or official submission checks.
- Do not hard-code undocumented thresholds as official rules.
- Do not over-filter until the daily submission budget becomes unusable.
- Do not formal-submit from the review step.

## Related skills

- `wq-alpha-hypothesis`
- `wq-alpha-repair`
- `wq-alpha-diversify`
