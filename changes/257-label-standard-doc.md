### Added

- `recipes/pr-labeler/README.md`, "The label standard": D1A uses hermes-agent's label set by its written definitions
  (hermes-agent's own label descriptions), not by the habits of whoever labelled the training PRs (#252). The section
  lists the definitions, what #252 measured (severity labels match the definitions on about 60% of checked rows), and
  what the next training round changes (a P2/P3 tie-break, blast wording, relabelled severity rows). Until then, the
  PR labeler is served one decode rule for severity (P3/P4 → P2 at p(P2) ≥ 0.20, #259); the section gives its numbers.
- docs/CONFIGURATION.md, `rules.<name>`: `t` is compared with the serving machine's own probability, which differs slightly between
  Apple chips, so a rule's flips are checked on the machine that serves it (#259).
