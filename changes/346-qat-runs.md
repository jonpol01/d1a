### Added

- A run trained with `--base_ct` records its QAT checkpoint (`base_ct` in `d1a_config.json`), and loading rebuilds that
  trunk (#346). The PyTorch path writes the checkpoint's int4 weights over the base, exact in fp32. `scripts/export_mlx.py`
  exports such a run on the same checkpoint without `--ct-base`, and refuses another; `--media` writes the QAT checkpoint's
  own encoders, so the export answers photos, voice and video. The MLX backend and the PyTorch media server refuse such a run
  rather than run it on the base's trunk.
