### Added

- Chat on PyTorch (#348): a LoRA run on an instruction-tuned base, served with `D1A_MERGE=0`, answers
  `/v1/chat/completions` with its adapter switched off, streaming and tool calls included. Its greedy text is transformers'
  own generation on the base. Decisions that arrive during a chat are answered between its tokens, as on MLX.
- The PyTorch photo and voice server (`d1a.serving.media`) serves a run trained on a QAT checkpoint (`base_ct`): the
  checkpoint's own text model and encoders, the vision projection QAT retrained included.

### Changed

- Gemma 4 tool calls parse without mlx-lm installed: its parser is now in `d1a/serving/gemma4_tools.py` (MIT, see NOTICE).
