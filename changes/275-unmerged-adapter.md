### Added

- One model that decides and chats on Apple Silicon (#274): `scripts/export_mlx.py --unmerged` keeps the LoRA out of the
  weights (`adapter/`, attached at load by `d1a.backends.mlx.attach_lora`), and `MLXDecisionModel.adapter(False)` switches
  it off per request, giving back the base model exactly. On an instruction-tuned base (D1A-E2B-it) that base is a chat
  model. `D1A_MERGE=0` now attaches a training run's adapter the same way on MLX. The export also carries the base's
  `chat_template.jinja` when the run has none. D1A-E2B-it, unmerged 8-bit: 0/42 flips against fp32 on the 28-request
  bench (max |dp| 0.062, 288 ms per request; the merged build: 0/42, 0.015, 160 ms).
