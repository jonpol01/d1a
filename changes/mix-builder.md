### Added

- `recipes/mix/build_mix.py`, the plan-driven training-mix builder that built D2 and D2-skills (#202), with its plans
  (`recipes/mix/plans/`) and a verifier (`recipes/mix/verify_mix.py`). A plan sets each source's count, stratification
  and weights, label quotas and an "only unseen" filter. Every eval partition and eval-only kit is screened out (exact
  state, PR id, near-duplicates), and the same plan and seed give the same bytes. A plan that leaves out a training
  source or decision-v7 is refused unless it names it with a reason, and the sidecar is the one `d1a.training.train`
  checks. The verifier checks every record byte for byte against its source and reruns the leak screens.
