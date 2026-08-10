# QuantGPT

QuantGPT coordinates quantitative research workflows, including WorldQuant BRAIN Alpha research, candidate selection, formal submission, and points progress.

## Language

### WorldQuant research

**Research Candidate**:
A simulated WorldQuant Alpha retained for possible future formal submission after passing the project's research-readiness checks.
_Avoid_: Potential Alpha, maybe-active Alpha

**Candidate Inventory**:
The persistent pool of Research Candidates available for future formal submission.
_Avoid_: Queue size, backlog

**High-Confidence Candidate**:
A Research Candidate whose estimated probability of surviving formal WorldQuant checks is high enough to count toward the Candidate Inventory target.
_Avoid_: Guaranteed ACTIVE, sure-pass Alpha

**ACTIVE Conversion Rate**:
The observed rate at which formally submitted Research Candidates ultimately become ACTIVE rather than failing terminal submission checks.
_Avoid_: Simulation pass rate, candidate pass rate

**Inventory Floor**:
The minimum acceptable number of High-Confidence Candidates that should be kept available for future formal submission. The current product decision is 30.
_Avoid_: Daily quota

**Inventory Target Band**:
The preferred operating range for High-Confidence Candidates. The current product decision is 40–50.
_Avoid_: Hard cap

### Research learning

**Directed Mutation**:
A bounded descendant research attempt created from normalized failure evidence to repair or deliberately change a weak Alpha hypothesis instead of treating every metric failure as terminal.
_Avoid_: Retry the same Alpha, random parameter tweak

**Research Provenance**:
The derivable research identity of a trial: its data fields, dataset/category, signal family, operator structure, settings, hypothesis, and parent lineage. Unknown provenance is evidence that resolution failed, not a valid scheduler bucket.
_Avoid_: Metadata decoration

**Learning Maturity**:
The evidence state that determines whether QuantGPT may use adaptive outcome learning. During cold start, sparse ACTIVE/Points outcomes remain diagnostic only and research scheduling stays coverage-first; adaptive weighting is enabled only after its evidence gates are satisfied.
_Avoid_: Inventory Mode, Genius level

**Positive Research Memory**:
Compact reusable evidence about hypotheses, structures, datasets, or mutation paths that produced candidates or measurable improvement and are worth exploring further.
_Avoid_: Copy the best Alpha

**Negative Research Memory**:
Compact reusable evidence about repeatedly failed, crowded, duplicate, or unproductive structures that should be avoided or structurally changed in future generation.
_Avoid_: Permanent blacklist of a whole dataset

### WorldQuant submission

**Live Submission**:
A formal WorldQuant submission that currently occupies one daily submission slot because it is reserved, pending final checks, or ACTIVE but not yet settled for the submission policy.
_Avoid_: Submission attempt

**Submission Slot**:
One unit of the daily formal-submission budget. Failed terminal attempts release the slot; Live Submissions consume it.
_Avoid_: Research slot, simulation quota

**Submission Refill**:
Selecting the next best eligible High-Confidence Candidate after a terminal submission failure so the daily Live Submission target can still be reached.
_Avoid_: Retry the same Alpha

**Submission-Ready Candidate**:
A tracked Candidate Inventory entry that is fresh, passes deterministic Sharpe/Fitness/Turnover and robustness readiness, has no official self-correlation failure, and has no fresh high local-correlation or available weak overfitting evidence. Only Submission-Ready Candidates may reserve a formal Submission Slot.
_Avoid_: Any simulated A-grade Alpha, arbitrary alpha_id

**Submission Reservation Lease**:
A short-lived local claim on a Submission Slot made immediately before the remote formal submit. The default lease is 15 minutes; an expired lease releases the local slot but the Alpha must be reconciled against BRAIN before it may be retried, preventing both stuck quotas and blind duplicate submissions.
_Avoid_: Permanent reservation, automatic retry

**Points Sync State**:
The evidence state for delayed leaderboard Points. `CURRENT` is valid only when both the platform ACTIVE count and leaderboard Alpha count are known and their gap is zero; missing count evidence is `SYNC_UNKNOWN` and must not settle Pending Points attribution.
_Avoid_: Assume current when counts are missing

### Research operation

**Replenishment Mode**:
Research behavior used when the Candidate Inventory is below the Inventory Floor; it prioritizes producing submission-ready High-Confidence Candidates.
_Avoid_: Emergency submit mode

**Normal Research Mode**:
Research behavior used while Candidate Inventory is within the Inventory Target Band; it balances inventory quality, diversity, and replenishment.
_Avoid_: Idle mode

**Exploration Mode**:
Research behavior used once Candidate Inventory is above the target band; research continues but shifts toward under-covered, low-correlation signal families instead of stopping.
_Avoid_: Stop mode
