### Added

- Self-learning v2 (#233, #240):
  - **Settings file.** `D1A_LEARNING=<file>` holds the loop's settings, validated, with a last-valid fallback and an
    audit log. `python -m d1a.learning.feedback config init|show|set|validate` manages it. Unset, everything behaves as
    before.
  - **Replay.** `feedback replay` and `feedback tick` re-score an earlier model's decisions that have outcomes through
    the live server (header `x-d1a-replay-of`), so a new model starts with every earlier outcome. It is fail-closed: no
    replay without the served model's complete trained manifest (`feedback trained`), and none for a decision whose PR
    or state is in that model's training, or that has no PR id.
  - **The tick** runs the gate after N new outcomes or at a daily time. Each gate report gives `mde`, the smallest
    log-loss gain its held-out set can detect, and the gate's interval level is a setting.
  - `d1a/core/settings.py` lists every `D1A_*` setting with its default and reader, every module reads them through it,
    and `python -m d1a.serving.serve --show-config` prints them (secrets masked).
