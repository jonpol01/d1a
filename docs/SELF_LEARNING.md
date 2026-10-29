# Self-learning

How D1A logs its decisions and learns from their outcomes: recalibration is in production use; retraining sits behind a held-out gate that no retrain has passed yet.

Most of D1A's decisions get a ground truth later: a patch passes or fails its tests, a maintainer keeps or corrects a label, a routed task succeeds or fails. `d1a.learning.feedback` turns those outcomes into a better model without letting a worse one through.

```python
from d1a import D1A
from d1a.learning.feedback import FeedbackLog

m, log = D1A.load("JohnP1/d1a-e4b-mlx-q8"), FeedbackLog("runs/feedback/verifier.jsonl")
q = {"resolved": {"type": "noul", "instructions": "Does this patch fix the issue?"}}
answers = m.decide(issue_and_patch, q)
did = log.decision(issue_and_patch, q, answers, run="JohnP1/d1a-e4b-mlx-q8@v0.6.1")
...                                     # later, when the tests have run
log.outcome(did, {"resolved": True})
```

```bash
python -m d1a.learning.feedback status runs/feedback/verifier.jsonl                       # outcomes, base rate, log loss
python -m d1a.learning.feedback calibrate runs/feedback/verifier.jsonl --out cal.json     # minutes: refit P on outcomes
python -m d1a.learning.feedback records runs/feedback/verifier.jsonl --out feedback.jsonl # data for a small LoRA update
```

**As a service.** `d1a.serving.serve` runs the loop for any client: with `D1A_FEEDBACK_LOG` set, every answer carries a `decision_id` and is logged (except a request that sends the header `x-d1a-decision-log: off`, such as a demo or a smoke test, which no outcome will follow), and `POST /v1/feedback` records what actually happened, with an optional `src` (a person's correction, `human`, outranks another model's, e.g. `reviewer`) and `group` (what the decision belongs to, e.g. a pull request); `python -m d1a.learning.feedback promote <log> --calibrator <file>` fits on part of the groups, gates each question on the rest, and rewrites the calibrator file only for the questions that pass. It fits and gates each temperature the decisions were read at on its own (the log records it; a use case can select another, `"question@T"` in its report), and the server applies an entry only to answers read at that temperature; entries in a file written before decisions logged their temperature apply only at the checkpoint's own. With `--run <repo@tag>` it fits only on the decisions that model made, as the log records them; pass the served model's, so that after a model switch the new model is not corrected by the old one's errors; with `D1A_OUTCOME_CALIBRATOR` set, yes/no and choice answers go through the latest calibrator, re-read whenever the file changes, so recalibrating needs no restart. The path may name a file that does not exist: until promote writes it, the answers are exactly the model's own, and the first answer after it appears goes through it. Moving the file aside stops the calibrator from the next answer, so a rollback needs no restart, while any other error reading it keeps the last one loaded.

**Recalibration forms and a time-ordered gate.** By default a choice question is recalibrated by one inverse temperature, and the gate holds out a random 30% of the groups. Settings can choose more (docs/CONFIGURATION.md):
- `promotion.form = "vector"`: a per-option bias besides the temperature, which also corrects a label mix that served traffic does not share with the data the model was trained and calibrated on.
- `promotion.split = "time"`: the gate fits on older outcomes and tests on the newest ones, which is how a promoted calibrator is used.
- `promotion.serve_choice` and `promotion.hold`: serve the recalibrated top option, while keeping the model's own answer for held options (for example severity P0/P1). A correction toward common classes would otherwise demote these rare, costly classes.

With `promotion.auto` on, `tick` writes a changed `serve_choice` or `hold` into the calibrator file at once, without waiting for a promotion, so the server applies it on its next reload; with it off, nothing is written. Learned flags always read the model's raw probabilities, so no recalibration moves a flag.

**Learned flags.** A flag is a learned decision threshold beside an answer, never a change to it: severity's "severe?" flag raises when p(P0)+p(P1) reaches a threshold fitted offline at a flag-rate budget (`d1a.learning.flags`; settings in docs/CONFIGURATION.md). On 487 pull requests newer than all of D1A-E4B v0.6's training, its top answer finds 15 of 30 P0/P1 pull requests and the flag at t = 0.30 finds 25 while raising on 14.6% of them; it is measured, not served (#237). A flag can also run in shadow mode (logged, not served), with a threshold that tracks the flag rate from logged scores, refit by `tick` without labels (#296). The severity flag stays shadow-only until a blind check passes.

**Decode rules.** A rule changes the served choice, and only for a client that asks for it by use case: for requests that send `"use_case": "pr-labeler"` to D1A-E4B v0.6, a severity answer of P3 or P4 is served as P2 when p(P2) ≥ 0.20 (`d1a.learning.rules`; settings in docs/CONFIGURATION.md). v0.6 learned severity from hermes-agent's maintainer labels, which on these pull requests give P2 less often than the label definitions they publish. On 87 held-out pull requests where two independent judges reading those definitions agree, the served answers match them 70.1% of the time, against 59.8% for v0.6's top answer and 58.6% for the maintainer labels: +10.3 points over the top answer (95% CI +2.3 to +18.4; the rule made 12 rows right and 3 wrong) and +11.5 over the maintainers (+2.3 to +20.7; #252). The probabilities, the confidence and the decision log stay the model's, and the served answer names the rule that changed it (`rule`). It is available, not active: on the maintainer's own repositories a blind check found it right on 3 of the 6 states it fires on, against 2 for v0.6's own answer, so it stays unset there until the next training round (#252).

```bash
D1A_FEEDBACK_LOG=runs/feedback/verifier.jsonl D1A_OUTCOME_CALIBRATOR=runs/feedback/cal.json \
  uv run --extra serve python -m d1a.serving.serve --run JohnP1/d1a-e4b-mlx-q8@v0.6.1 --port 8009
curl -s localhost:8009/v1/feedback -H 'content-type: application/json' \
  -d '{"decision_id": "<from the answer>", "labels": {"resolved": true}}'
python -m d1a.learning.feedback calibrate runs/feedback/verifier.jsonl --out runs/feedback/cal.json   # picked up on the next request
```

**Settings, replay and a scheduled loop.** `D1A_LEARNING=<file>` holds the loop's settings, validated, read on every run and changed with `python -m d1a.learning.feedback config set key=value`. They cover which questions and outcome sources learn, the minimum outcomes, the gate's held-out share and interval, when it runs, replay, and where reports go. `tick` runs the loop every few minutes. After a model switch it **replays** the earlier models' decisions that have outcomes through the live server, so the new model starts with every outcome instead of none. Replay is fail-closed: a decision whose pull request or state is in the new model's training is never replayed, and no replay runs without a complete trained manifest of that model. It then gates after N new outcomes or at a daily time, once the outcomes have grown by `schedule.growth` (half, by default) since the last gate run, and promotes only on at least `promotion.min_held_out` (30) held-out groups. Each report says the smallest log-loss gain its held-out set can detect. Every gate run can promote: run every 10 outcomes, the gate promoted noise in 36.5% and 56.5% of 200 null replays by 1,000 outcomes, and with these two defaults in 7% and 9.5% ([#364](https://github.com/jonpol01/d1a/issues/364)). A changed promotion setting also waits for that growth, unless `promote` is run by hand. Every setting: [docs/CONFIGURATION.md](CONFIGURATION.md#self-learning).

`gate()` decides whether the retrained candidate replaces the current model: lower held-out log loss with a 95% bootstrap interval (over repositories, for example) entirely below zero, and no frozen suite more than a point worse. Otherwise the current model stays.

**Measured first on a coding-agent verifier**: D1A-E4B reads an issue, an agent's patch and the end of its run, and answers "does this patch resolve the issue?", tested on 921 agent runs from 83 repositories it never saw (public SWE-agent runs, [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories), CC-BY-4.0):

| | Ranking (AUROC) | Calibration error (ECE) |
|---|---|---|
| D1A-E4B v0.4, before any outcomes | 0.742 | 0.632 |
| recalibrated on outcomes (Platt) | 0.742 | 0.043 |
| trained on outcomes (LoRA, 800 runs, 2.5 h on a Mac) | **0.807** | **0.039** |
| trajectory heuristics, for comparison | 0.768 | 0.019 |

Trained on outcomes, it also picks better training data for an agent: the 10% of runs it trusts most are 40% truly resolved, against 23% for the heuristics.

**The loop, run for three rounds** (300 new outcomes each, the verifier measured on test repositories after every round, [#119](https://github.com/jonpol01/d1a/issues/119)):

| Round | Outcomes | Calibration error (ECE) | Gate on the retrained candidate |
|---|---|---|---|
| 0 | 0 | 0.068 | – |
| 1 | 300 | 0.036 | rejected: clearly worse (log loss +0.026, 95% CI +0.016..+0.058) |
| 2 | 600 | **0.017** | rejected: not clearly better (+0.010, CI −0.004..+0.050) |
| 3 | 900 | 0.031 | rejected: not clearly better (−0.007, CI −0.016..+0.014) |

Recalibration on outcomes works at once and needs no restart. Small retraining rounds have not yet produced a model that is clearly better, and the gate has kept every one that was not, so the loop has never shipped a regression. Ranking runs of the *same* issue against each other remains hard: on 119 issues with mixed outcomes, D1A reaches within-issue AUROC 0.64 against 0.62 for cheap trajectory heuristics, a difference within noise.
