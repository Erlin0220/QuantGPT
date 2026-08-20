---
name: wq-robustness-validation
description: Design a small, hypothesis-targeted WorldQuant BRAIN robustness plan before a candidate consumes cross-setting validation budget. Distilled from PBO/DSR/PSR research, WorldQuant risk guidance, and computer-experiment principles. Use after wq-alpha-review and before BRAIN Simulation packaging.
---

# WorldQuant Robustness Validation

## R — Reading

Distilled from QuantGPT Knowledge Cards backed by:

- `ssrn:bailey-borwein-lopez-zhu-pbo-2015` — Probability of Backtest Overfitting / CSCV.
- `ssrn:bailey-lopez-deflated-sharpe-2014` — Deflated Sharpe Ratio and multiple-testing/selection-bias control.
- `ssrn:bailey-lopez-probabilistic-sharpe-2012` — Probabilistic Sharpe Ratio and finite-sample/non-normality uncertainty.
- `worldquant:learn2quant-validation-evidence` — Alpha quality, diversification, risk management and BRAIN feedback.
- `textbook:santner-williams-notz-computer-experiments-2018` through the existing research-methodology layer — targeted simulator design and sensitivity analysis.

The methodology is advisory. It does not define WorldQuant submission limits. Current BRAIN responses remain authoritative.

## I — Interpretation

Robustness is not "run every Alpha through the same TOP1000 × INDUSTRY grid". A useful validation changes a setting because that perturbation can falsify the stated hypothesis or expose a known implementation risk.

The Skill decides **what to test and why**. QuantGPT code executes those BRAIN simulations and records the raw evidence. Do not collapse sparse evidence into an invented universal pass ratio.

## A1 — Past application

- A subindustry-relative price-reversion Alpha may need a peer-group/neutralization perturbation because its mechanism depends on comparability within groups.
- A sparse analyst/fundamental Alpha may need a universe/coverage perturbation because missingness and liquidity can create apparent strength that does not generalize.
- A near-miss selected after many sibling variants should carry lineage-level multiple-testing evidence; the winner's Sharpe alone is insufficient.

## A2 — Future trigger

Invoke for every `RUN` Alpha after `wq-alpha-review`, before packaging it into `skill_candidates`. Re-run when the hypothesis, data source, horizon, neutralization logic or major implementation assumption changes.

## E — Execution

1. Query `kb_search_alpha_knowledge` for `robustness PBO DSR PSR multiple testing cross setting validation` and preserve relevant Knowledge Card IDs.
2. Read the candidate hypothesis, family, fields, intended horizon/delay, neutralization rationale, recent lineage, related-trial count, and known failure risks.
3. Design **1–4** targeted checks. Each check must change one interpretable setting and include a purpose. Allowed fields in the plan are:
   - `region`
   - `universe`
   - `delay`
   - `decay`
   - `neutralization`
   - `truncation`
   - `purpose`
4. Prefer checks that answer a named question, for example:
   - universe/coverage sensitivity;
   - peer-group or neutralization dependence;
   - horizon/decay sensitivity when the mechanism is explicitly horizon-dependent;
   - concentration sensitivity when BRAIN feedback suggests weight concentration.
5. Do **not** mechanically add every dimension. If a perturbation cannot change the interpretation of the Alpha, omit it.
6. Preserve lineage evidence requirements:
   - if many related variants were tried, request/retain DSR/PBO-style evidence when return history is sufficient;
   - do not mix unrelated Alpha families into one PBO family;
   - if sample requirements are not met, mark statistical evidence unavailable rather than negative.
7. Return a structured `robustness_plan`:

```text
{
  "mode": "skill_defined",
  "checks": [
    {
      "universe": "TOP1000",
      "neutralization": "SUBINDUSTRY",
      "purpose": "test universe/coverage sensitivity while preserving the peer-group hypothesis"
    }
  ],
  "lineage_multiple_testing": true,
  "knowledge_card_ids": ["..."],
  "notes": "why these checks are informative"
}
```

8. Add `wq-robustness-validation` to `skill_chain`. The plan must travel with the candidate into `wq_brain_autonomous_research`.

## B — Boundary

- Never invent a universal local robustness pass threshold.
- Never call local robustness an official BRAIN check.
- Missing local evidence is unknown, not automatic failure.
- Keep the plan small; cross-setting Simulation is a real budget cost.
- Official SELF_CORRELATION and current BRAIN eligibility remain code/platform facts.
- If a check reveals structural collapse, feed the evidence to diagnosis/allocation; do not automatically mutate inside this Skill.

## Related skills

- `wq-alpha-review`
- `wq-candidate-evidence`
- `wq-experiment-allocation`
- `wq-alpha-repair`
