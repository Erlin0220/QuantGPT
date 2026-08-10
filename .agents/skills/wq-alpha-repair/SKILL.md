---
name: wq-alpha-repair
description: Repair a WorldQuant BRAIN Alpha after real simulation/submission-check feedback such as low Sharpe/Fitness, high turnover, sub-universe weakness, concentration, or self-correlation. Use evidence-preserving targeted mutations instead of random regeneration.
---

# WorldQuant Alpha Repair

## R — Reading

Distilled from `worldquant:alpha-examples-104`, `worldquant:learn2quant`, `worldquant:iqc-guidelines-2026`, live BRAIN checks, and QuantGPT's empirical Research Memory.

The official Price/Volume example explicitly treats turnover control, volatility-conditioned trading and decay as improvements to the same reversion hypothesis. Official IQC rules make correlation a real pool-level constraint and prohibit deliberate noise/gaming.

## I — Interpretation

A failed Alpha is an experiment result. Repair the smallest causal dimension indicated by BRAIN while preserving the hypothesis when possible. If the evidence says the hypothesis family is saturated or structurally weak, switch hypotheses rather than endlessly tuning parameters.

## A1 — Past application

For the official close-open reversion example, turnover is handled by smoothing/Decay and by trading only in reversion-friendly or higher-volatility conditions—not by replacing the signal with unrelated operators.

For self-correlation risk, official rules support genuine differentiation; cosmetic noise is not an acceptable repair.

## A2 — Future trigger

Invoke after a real WQ simulation/check produces actionable failure evidence: low Sharpe, low Fitness, turnover outside range, weak sub-universe performance, concentrated weights, Self-Correlation, sparse coverage, or near-threshold failure. Use `wq-alpha-hypothesis` when there is no parent Alpha yet.

## E — Execution

1. Read the exact BRAIN metrics/checks and `diagnose_wq_result` output. Also query `kb_explain_alpha` when the Alpha has Knowledge Card lineage.
2. Search `kb_search_alpha_knowledge` for the parent family and failure mode; prefer cards that have empirical trial history.
3. Classify the failure:
   - **Turnover high:** first test simulation Decay or light smoothing; then conditional `trade_when` if the hypothesis has a defensible regime condition.
   - **Sharpe/Fitness near threshold:** preserve the main field/mechanism; change one lookback, normalization, estimate variant, or neutralization at a time.
   - **Sub-universe weakness / concentration:** prefer broader coverage, group-relative transforms, simpler expressions, or a better-supported field before adding complexity.
   - **Common factor exposure / shared drawdown:** preserve the idiosyncratic hypothesis and test a justified neutralization change; compare Sharpe/drawdown against any turnover increase. Do not use neutralization as a generic cure.
   - **Self-correlation:** change data family, information source, or economic mechanism. Do not inject random noise. If common-factor exposure is specifically implicated, a neutralized sibling may be tested before abandoning the family.
   - **Coverage/sparsity:** use live field metadata and conservative `ts_backfill`/group handling when justified.
4. Generate at most 2–4 directed siblings per diagnosis. Record parent, mutation type and reason. Preserve the parent hypothesis and emit `skill_chain: ["wq-alpha-hypothesis", "wq-alpha-repair", "wq-alpha-review"]` after each child passes `wq-alpha-review`; only `review_decision: "RUN"` children may return to BRAIN.
5. Re-simulate through `wq_brain_autonomous_research` as structured `skill_candidates` with deterministic fallback disabled. Compare the child against the parent on the metric that motivated the mutation and reject repairs that only improve unrelated metrics.
6. Persist the outcome so Research Memory learns which repair classes work for each family.
7. If two directed repair rounds fail on the same root cause, stop local tuning and switch to `wq-alpha-diversify` / a new hypothesis.

## B — Boundary

- Never repair correlation by adding meaningless constants/noise.
- Do not simultaneously change field, sign, window, neutralization and decay; attribution becomes impossible.
- Do not treat local robustness/correlation proxies as more authoritative than current BRAIN checks.
- Do not formal-submit automatically; submission remains under the daily Submission Gate.

## Related skills

- `wq-alpha-hypothesis`
- `wq-alpha-review`
- `wq-alpha-diversify`
