---
name: wq-failure-diagnosis
description: Diagnose WorldQuant BRAIN Alpha failures from raw platform evidence before experiment allocation or repair. Separate observed symptoms from plausible causes, use lineage/knowledge evidence, and propose discriminating tests instead of jumping from a failed metric to a repair rule.
---

# WorldQuant Failure Diagnosis

## R — Reading

Distilled from existing QuantGPT Knowledge Sources rather than inventing a local expert system:

- `worldquant:learn2quant-lesson-3` — multi-metric Alpha quality evaluation;
- `worldquant:alpha-examples-104` — hypothesis → implementation → simulation → targeted improvement structure;
- `worldquant:learn2quant-lesson-7` — horizon/delay and transaction-cost implications;
- `worldquant:learn2quant-lesson-9` — common-factor risk and neutralization;
- `textbook:santner-williams-notz-computer-experiments-2018` — screening, sensitivity and attributable experimental design;
- `textbook:powell-ryzhov-optimal-learning-2012` — value of information for costly follow-up experiments.

Also query the `failure_diagnosis` Knowledge Cards and the parent Alpha's Research Memory lineage. Current BRAIN metrics/checks remain the fact layer and override stale summaries.

## I — Interpretation

A BRAIN failure code or threshold miss is a **symptom**, not necessarily the cause. `low_fitness`, `low_sharpe`, `turnover_high`, concentration, sub-universe weakness and correlation can interact. The diagnosis must keep observation and inference separate.

Do not behave like the old `if low_fitness -> smooth_signal` table. Instead answer:

1. What exactly did BRAIN observe?
2. Which causal explanations are plausible given the hypothesis/settings/lineage?
3. What evidence supports and contradicts each cause?
4. What smallest experiment would distinguish the leading explanations?
5. Is the evidence strong enough to repair, or should allocation choose diversify/new hypothesis?

## A1 — Past application

A candidate with acceptable Sharpe, low Fitness and high turnover may plausibly suffer execution drag or horizon mismatch. The same low Fitness paired with weak Sharpe is more consistent with weak signal quality. A sub-universe/concentration failure is different again. The label `low_fitness` alone is insufficient to choose among those actions.

## A2 — Future trigger

Invoke after any real BRAIN Simulation/check that is not a clean candidate, before `wq-experiment-allocation` and before `wq-alpha-repair`. Also invoke when the same lineage has repeated failures and the causal story needs to be updated.

## E — Execution

1. Read the raw `failure_evidence`: BRAIN metrics/checks, error, validation evidence, settings, official SC/submission result, and any local advisory evidence. Preserve them verbatim as facts.
2. Read parent hypothesis and lineage: parent metrics, mutation type, which dimension changed, child outcomes, dataset/family/operator pattern, prior diagnosis and Knowledge Card IDs.
3. Query `kb_search_alpha_knowledge` for `failure diagnosis symptom cause counterfactual low fitness turnover robustness` and preserve relevant card IDs.
4. Build a **Failure Signature** with these fields:
   - `observed_symptoms`: factual platform/local observations only;
   - `primary_failure_stage`: compile / simulation / metrics / robustness / diversity / submission;
   - `plausible_causes`: 1–3 causes, each with `cause`, `supporting_evidence`, `counter_evidence`, `confidence` (`low|medium|high`);
   - `unknowns`: material missing evidence;
   - `discriminating_tests`: 0–3 smallest tests that would change the diagnosis;
   - `diagnosis_summary`: concise interpretation without pretending certainty;
   - `knowledge_card_ids`.
5. Common causal hypotheses may include, when evidence supports them:
   - `weak_signal` — weak Sharpe/returns across settings or repeated structural failures;
   - `execution_drag` — useful risk-adjusted signal but turnover/cost-like implementation burden;
   - `horizon_mismatch` — hypothesis half-life and implementation responsiveness conflict;
   - `coverage_or_sparsity` — field availability/backfill/group coverage drives instability;
   - `concentration_or_exposure` — weight concentration/common-factor exposure dominates;
   - `subuniverse_instability` — behavior does not survive a relevant universe stress;
   - `neutralization_mismatch` — chosen exposure treatment conflicts with the hypothesis;
   - `temporal_instability` — evidence changes materially across time/regime;
   - `correlation_saturation` — insufficiently independent information from the existing pool;
   - `compile_or_field_semantics` — invalid operator/field use or unsupported semantics.
   These are hypotheses, not automatic mappings.
6. For `low_fitness`, explicitly decompose Sharpe, returns, turnover, horizon, coverage/concentration and robustness before naming a cause. Never make `low_fitness` itself the causal diagnosis.
7. If several causes remain plausible, prefer one small counterfactual experiment per leading cause. A test is useful only if its possible outcomes would favor one explanation over another.
8. Return the Failure Signature to `wq-experiment-allocation`. Do **not** directly generate children unless the allocation decision later returns `REPAIR`.
9. If official Self-Correlation or repeated portfolio correlation dominates, set `correlation_saturation` as a leading cause and route allocation toward `wq-alpha-diversify` rather than cosmetic repairs.
10. Persist the diagnosis and the result of its discriminating test in Research Memory when the workflow supports it, so future diagnoses can use actual lineage evidence.

## B — Boundary

- Raw BRAIN evidence is fact; causal diagnosis is inference. Keep the distinction explicit.
- Never invent a universal causal rule from one metric threshold.
- Never change several causal dimensions in one diagnostic experiment unless the goal is deliberately broad screening.
- Local correlation/robustness/overfit evidence is advisory and must not be described as an official platform failure.
- Do not repair or formally submit from this Skill.
- If evidence is insufficient, say `diagnosis_uncertain` and request the smallest discriminating experiment rather than guessing.

## Related skills

- `wq-experiment-allocation`
- `wq-alpha-repair`
- `wq-alpha-diversify`
- `wq-robustness-validation`
- `wq-candidate-evidence`
