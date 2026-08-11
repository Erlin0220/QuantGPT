---
name: wq-candidate-evidence
description: Define the evidence hierarchy for ranking WorldQuant candidates without pretending a hand-weighted metric score is a calibrated ACTIVE probability. Use after wq-alpha-review when packaging a candidate, and again when interpreting post-simulation candidate evidence.
---

# WorldQuant Candidate Evidence

## R — Reading

Distilled from:

- `textbook:murphy-probabilistic-machine-learning-2022` — probabilistic prediction, uncertainty and decision-theoretic framing.
- `paper:arrieta-ibarra-calibration-metrics-2022` — probabilistic calibration/reliability and finite-sample limitations.
- `ssrn:bailey-lopez-deflated-sharpe-2014` — multiple-testing/selection-bias correction.
- `ssrn:bailey-lopez-probabilistic-sharpe-2012` — Sharpe uncertainty and track-record length.
- `worldquant:learn2quant-validation-evidence` — BRAIN-centered Alpha quality/risk feedback.

The Skill defines **how evidence should be interpreted**. It does not invent hidden WorldQuant rules or numeric ACTIVE probabilities.

## I — Interpretation

Keep three layers separate:

1. **Hard facts / eligibility** — current BRAIN metrics/checks and explicit official SC failure. Code/platform enforcement owns these.
2. **Calibrated outcome evidence** — empirical ACTIVE vs terminal-failure history. Use as `P(ACTIVE)` only after the learning-maturity gate has enough resolved outcomes and preserve support/provenance.
3. **Advisory preference evidence** — Fitness, Sharpe, returns, robustness stress results, DSR/PBO/PSR, local correlation, novelty and lineage context. These rank/explain candidates but are not probabilities by themselves.

A deterministic weighted sum such as `0.21*Fitness + 0.17*Sharpe + ...` is not a calibrated probability.

## A1 — Past application

If QuantGPT has only a handful of resolved formal submissions, do not emit `P(ACTIVE)=0.73` from a metric blend. Report ACTIVE probability unavailable, retain the actual BRAIN metrics, and rank transparently by the evidence hierarchy. Once terminal outcome history is mature, use the most specific supported empirical context (cell → family/dataset → global) and test its calibration on resolved outcomes.

## A2 — Future trigger

Invoke:

- for every `RUN` Alpha after `wq-alpha-review`, to attach an evidence policy before Simulation;
- after BRAIN results when comparing multiple eligible candidates;
- before choosing which candidate should consume a formal submission slot;
- when a reported ACTIVE probability or confidence tier appears more precise than its outcome support allows.

## E — Execution

1. Query `kb_search_alpha_knowledge` for `candidate evidence probability calibration ACTIVE ranking DSR PBO` and preserve relevant Knowledge Card IDs.
2. Attach this default evidence hierarchy unless current official BRAIN evidence requires an explicit exception:
   1. `official_eligibility`
   2. `calibrated_active_outcome_rate`
   3. `fitness`
   4. `sharpe`
   5. `returns`
   6. `robustness_and_multiple_testing_evidence`
   7. `diversity_case_and_realized_correlation`
   8. `local_correlation_and_novelty`
3. `official_eligibility` is not a preference score. An explicit official blocker remains a blocker.
4. `calibrated_active_outcome_rate` may be used as probability only when QuantGPT's learning gate is ready. Require every probability to expose:
   - `support`;
   - `provenance` (cell/family/dataset/global);
   - calibration diagnostics when enough resolved predictions exist.
5. During cold start, require `probability: null`. Do not backfill it from a weighted metric score.
6. During cold start, compare eligible candidates lexicographically by the transparent BRAIN evidence hierarchy (Fitness → Sharpe → returns), with robustness/multiple-testing/local-correlation evidence used as named advisory tie-break/context rather than hidden weights.
7. When ACTIVE outcome evidence is mature, prefer the calibrated empirical probability first, then the transparent raw-metric hierarchy. Do not blend it with an arbitrary local metric formula and still call the result calibrated.
8. DSR/PBO/PSR evidence must preserve its assumptions and sample support. `unavailable` stays neutral.
9. If a `diversity_case` exists, distinguish `designed_diverse` from `empirically_supported`; only official SC/PnL evidence can upgrade a design rationale into realized diversification evidence.
10. Return/preserve a structured `candidate_evidence_policy`:

```text
{
  "mode": "calibrated_evidence_hierarchy",
  "probability_requires_mature_outcomes": true,
  "cold_start_probability": null,
  "evidence_order": [
    "official_eligibility",
    "calibrated_active_outcome_rate",
    "fitness",
    "sharpe",
    "returns",
    "robustness_and_multiple_testing_evidence",
    "diversity_case_and_realized_correlation",
    "local_correlation_and_novelty"
  ],
  "knowledge_card_ids": ["..."],
  "notes": "candidate-specific evidence caveats"
}
```

11. Add `wq-candidate-evidence` to `skill_chain`; the policy must travel with the candidate into QuantGPT and remain visible in validation/audit metadata.

## B — Boundary

- Never fabricate calibrated probabilities from hand-tuned metric weights.
- Never turn a local correlation/overfit proxy into an undocumented WorldQuant rule.
- Never erase support/provenance from empirical ACTIVE rates.
- Never treat a high Sharpe selected from many siblings as independent of its search history.
- Ranking is separate from formal submission eligibility; BRAIN remains final authority.

## Related skills

- `wq-alpha-review`
- `wq-robustness-validation`
- `wq-experiment-allocation`
- `wq-alpha-diversify`
