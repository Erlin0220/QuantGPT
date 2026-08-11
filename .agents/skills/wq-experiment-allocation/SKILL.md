---
name: wq-experiment-allocation
description: Allocate scarce WorldQuant BRAIN Simulation budget across new hypotheses, targeted repairs, diversification, and bounded numeric refinement using distilled Bayesian optimization, optimal learning, bandit, and computer-experiment methodology. Use when deciding what experiment should run next or whether a failed Alpha is worth another repair.
---

# WorldQuant Experiment Allocation

## R — Reading

This skill is distilled from authoritative sequential-experimentation sources stored in QuantGPT Knowledge Sources:

- `textbook:garnett-bayesian-optimization-2023` — Roman Garnett, *Bayesian Optimization*: expensive black-box objectives; Chapters 5–9 separate decision theory, utility definition, acquisition-policy choice and practical computation; Chapter 11 covers extensions/related settings such as richer evaluation settings. The key transfer is **utility first, policy second**.
- `textbook:lattimore-szepesvari-bandit-algorithms-2020` — Tor Lattimore & Csaba Szepesvari, *Bandit Algorithms*: exploration/exploitation, contextual/linear bandits, pure exploration, Bayesian methods and Thompson sampling.
- `textbook:slivkins-introduction-multi-armed-bandits-2019` — Aleksandrs Slivkins, *Introduction to Multi-Armed Bandits*: adaptive exploration, contextual bandits, structured actions and bandits with knapsacks/budget constraints.
- `textbook:powell-ryzhov-optimal-learning-2012` — Warren B. Powell & Ilya O. Ryzhov, *Optimal Learning*: expensive information collection, learning policies and knowledge-gradient/value-of-information thinking.
- `textbook:santner-williams-notz-computer-experiments-2018` — Santner, Williams & Notz, *The Design and Analysis of Computer Experiments*: simulator design, screening, sensitivity analysis, criterion-based designs and sequential optimization.
- `paper:reyes-powell-optimal-learning-lab-2020` — open tutorial reinforcing belief models, learning policies and value of information for costly sequential experiments.

These sources guide research allocation. They do **not** define WorldQuant submission thresholds or guarantee an Alpha outcome; current BRAIN evidence remains authoritative.

## I — Interpretation

A BRAIN Simulation is an expensive experiment. New hypotheses, repair attempts, diversification moves and local parameter refinements should compete for the same scarce budget. The next experiment should be chosen for its expected decision value under uncertainty, not because a fixed workflow says every failed Alpha gets the same number of retries.

This skill therefore separates three problems:

1. **Semantic research choice** — which hypothesis/data family/mechanism to test. Use contextual/budgeted allocation principles, not numeric optimization.
2. **Repair choice** — whether a failed lineage is worth another experiment and which diagnosis-specific repair would teach the most.
3. **Bounded numeric refinement** — once hypothesis and structure are fixed, a small lookback/decay/truncation/neutralization search may use Bayesian-optimization ideas.

## A1 — Past application

When repeated `low_fitness` trials come from the same family/structure, do not automatically generate more window variants. Compare the lineage's historical repair outcomes with alternative experiment classes. If a decay/smoothing repair has repeatedly rescued high-turnover near-misses, it deserves more budget; if window mutation for the same failure/context has repeatedly produced no Candidate, allocate the next Simulation elsewhere.

## A2 — Future trigger

Invoke after `wq-failure-diagnosis` has produced a Failure Signature when the trigger is a real BRAIN failure. Also invoke when:

- deciding `REPAIR` vs `DIVERSIFY` vs `NEW_HYPOTHESIS` vs `DEFER`;
- selecting which failed Alphas deserve repair budget;
- choosing among multiple repair classes;
- the candidate inventory is low and Simulation budget must be allocated efficiently;
- a near-miss has a fixed semantic hypothesis and only a small numeric refinement space remains.

## E — Execution

1. Query `kb_search_alpha_knowledge` for `simulation allocation sequential experiment repair value of information` and preserve the relevant methodology card IDs.
2. Read the structured `failure_signature` from `wq-failure-diagnosis` when this is a failure-driven decision, then read Research Memory/lineage for the competing experiment choices. At minimum capture: family, dataset, operator pattern, observed symptoms, plausible causes, discriminating tests, parent Sharpe/Fitness/turnover, prior mutation type, child outcomes, Candidate outcomes, ACTIVE outcomes when mature, and sample counts.
3. Build a small experiment slate. Each item is one of:
   - `NEW_HYPOTHESIS` — route to `wq-alpha-hypothesis`;
   - `REPAIR` — route to `wq-alpha-repair` with one diagnosis-specific causal change;
   - `DIVERSIFY` — route to `wq-alpha-diversify` when correlation/family saturation dominates;
   - `NUMERIC_REFINE` — only for a fixed hypothesis/structure with a bounded low-dimensional parameter space;
   - `DEFER` — do not spend a Simulation now.
4. Define the decision utility **before** choosing an acquisition-like ranking or allocation policy. For QuantGPT the utility is not raw Fitness alone; evaluate each item on source-grounded dimensions rather than a made-up universal formula:
   - empirical promise from context-matched lineage outcomes;
   - uncertainty/sample scarcity, preserving exploration where evidence is weak;
   - value of information: will the result discriminate between plausible failure explanations or change the next decision?;
   - Simulation cost/budget pressure;
   - redundancy with already planned experiments;
   - downstream Candidate/ACTIVE evidence when statistically mature.
   Only after those decision consequences are explicit should an uncertainty-aware policy choose the next run.
5. Prefer experiments that either have credible Candidate upside **or** materially reduce uncertainty. Repeated variants that are both low-promise and low-information should lose budget to another route.
6. For uncertain diagnoses, use a deliberately small screening batch that changes one causal dimension per child. Do not change field + sign + window + neutralization + decay together; computer-experiment methodology requires attributable results.
7. Use Bayesian-optimization concepts only when the semantic Alpha is fixed and the search is genuinely bounded/low-dimensional. Do not apply GP/EI/UCB/Optuna-style numeric search across arbitrary FASTEXPR structures or economic hypotheses.
8. For a batch, prefer complementary experiments whose outcomes answer different questions. Do not use a batch merely to evaluate several near-identical points in parallel; redundant evaluations have lower joint information value.
9. Preserve exploration. Do not permanently blacklist a family or repair class from a handful of failures. Use confidence/sample maturity and recent evidence; BRAIN behavior and account-visible datasets can change.
10. Stopping is evidence-based, not a magic retry count. If further repair is dominated by another experiment in both promise and expected information value, return `STOP_REPAIR` and route budget to `DIVERSIFY` or `NEW_HYPOTHESIS`. If evidence is insufficient, prefer a small discriminating experiment rather than a large repair batch.
11. Return a compact allocation decision with: `decision`, `reason`, `evidence`, `knowledge_card_ids`, `uncertainty_note`, and `next_skill`. When choosing `REPAIR`, also name the single failure cause the child is intended to test. When choosing `NEW_HYPOTHESIS`, route through `wq-economic-hypothesis` before `wq-alpha-hypothesis` so budget is not replenished with another operator-first batch.

## B — Boundary

- Never claim textbook methodology is an official WorldQuant submission rule.
- Never use a local surrogate to hard-block all BRAIN exploration; ranking/allocation is advisory and BRAIN remains the empirical judge.
- Never replace `wq-failure-diagnosis` with a direct metric-to-repair mapping; allocation consumes the diagnosis, it does not invent one.
- Never invent fixed probability thresholds or retry counts that are not calibrated from QuantGPT evidence.
- Do not optimize arbitrary symbolic Alpha expressions as if they formed a smooth numeric space.
- Do not reward repeated near-identical experiments merely because they are cheap to generate.
- Do not spend repair budget when the proposed experiment cannot plausibly change the downstream research decision.

## Related skills

- `wq-alpha-hypothesis`
- `wq-alpha-review`
- `wq-alpha-repair`
- `wq-alpha-diversify`
