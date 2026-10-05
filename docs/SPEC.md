# D1A specification

What a D1A implementation must do to give D1A's answers: how a request becomes model input, how the pointer head scores
it, how answers are reported, the checkpoint formats, and the conformance test any port must pass. The reference
implementation is this repository (`d1a/core/api.py`, `d1a/backends/torch.py`, `d1a/backends/checkpoint.py`, `d1a/serving/serve.py`). Where this text and
the code disagree, the code is right and this text has a bug.

"Must" marks what changes answers when it is not followed. Everything else (the packed, row and cached forms, batching,
backends) is free, as long as the answers are the same.

## 1. Requests

A request is TypeSafe's System One request (`POST /v1/systemone`): a `state` (any JSON value), a `model` name, and
`questions`, an object of question id → question. A question is one of:

| type | fields | options, in this order | reported under the keys |
|---|---|---|---|
| `noul` (yes/no) | `instructions`, optional `criteria` `{"true": ..., "false": ...}` | `no`, `yes` | `false`, `true` |
| `choice` | `instructions`, `criteria`: key → description (1 to 255) | one per criteria key, in the object's order | the criteria keys |
| `score` | `instructions`, `criteria`: a list of levels (1 to 255) | one per level, in order | `"0"`, `"1"`, ... |

Each request becomes one **record**: the state as text, and per question its instructions as text and its options as
texts. The text rules are part of every trained model's contract; a change to them changes every answer.

- **render(value)** (`d1a.core.api.render`): `null` → `""`; a string, number or boolean → `str(value)` (Python's spelling:
  `True`, `1.5`); a list → one line per item, `"- " + render(item)`; an object → one line per key, `"key: " + render(value)`
  for a scalar, or `"key:"` then the nested value on the following lines. Nested lists and objects are indented two spaces
  per level. Lines are joined with `\n`.
- **Option text** (`option_text`): the option's name, then `": " + render(description)` when the description is neither
  `null` nor `""`. For `noul` the names are `no` and `yes` with the `false` and `true` criteria; for `choice` the name is
  the criteria key; a `score` level's text is `render(level)`.
- Questions keep the request's order.

`d1a.serving.serve` can add date facts to the state (`with_date_facts`, opt-in); that is a preprocessing step, not part of the model.

## 2. Model input

**Tokens.** The state, each question's instructions and each option are tokenized separately with the base model's
tokenizer, without special tokens. Caller text must never produce a delimiter or control token: before tokenizing,
every `<|name|>` becomes `<¦name¦>`, and every other special token of the tokenizer gets `¦` after its first character
(`user_tokens`).

**Delimiters.** Five existing special tokens of the base vocabulary mark the structure, in this role order: state, q,
opt, /opt, decide. The first set whose five tokens all exist (and are not unknown) is used:

| base family | state | q | opt | /opt | decide |
|---|---|---|---|---|---|
| Qwen | `<\|fim_prefix\|>` | `<\|fim_middle\|>` | `<\|box_start\|>` | `<\|box_end\|>` | `<\|fim_suffix\|>` |
| Gemma 4 | `<unused0>` | `<unused1>` | `<unused2>` | `<unused3>` | `<unused4>` |

**Leading ids.** What the tokenizer puts before any text (`tok("", add_special_tokens=True)`): Gemma's `<bos>`, nothing
for Qwen.

**Packing** (`encode`). One sequence, with a segment number and a position per token:

```
[leading ids] <state> state tokens                                  segment 0, positions 0 ... S-1
<q> instruction tokens (<opt> option tokens </opt>)... <decide>     segment k for question k, positions S, S+1, ...
```

Every question's positions restart at S, the length of the state part (leading ids and `<state>` included), as if it
were the only question. A question reads its answer at two kinds of token: its `<decide>` and each option's `</opt>`.

**Limits.** A state longer than the limit is cut to its first tokens (or refused in strict mode); a question whose branch
does not fit its row (state plus branch) is refused. Training: 384 state tokens, 1,024 per row, 2,048 per record
(`training_context` raises them together). Serving: 65,536 state tokens and 73,728 per row.

## 3. Attention

The **block-causal rule**: token i attends to token j if and only if j ≤ i and (j is in segment 0, or j is in i's
segment). Questions never see each other, and the state never sees a question. With sliding-window layers, a key is also
dropped when its position is `window` or more before the query's (positions, not indices, so each question sees what it
would see alone).

Equivalent forms, all exact: the packed sequence under this mask; one causal row per question (the state, then that
question's branch, at the same positions); and the state run once into a cache that each question's branch continues.
Backbones with recurrent layers (Qwen3.5's Gated DeltaNet) cannot honour the mask and must use the row or cached forms.

## 4. The pointer head

Inputs: the backbone's final hidden states (d wide) in fp32, at a question's `<decide>` (h_d) and at each of its K
options' `</opt>` (h_1 ... h_K).

```
q = W_q h_d + b_q               (256)
k_i = W_k h_i + b_k             (256 each)
z_i = (k_i · q) / sqrt(256) / T
p = softmax(z)                  over the question's K options
```

`T` is the checkpoint's temperature (§6). The head runs in fp32 whatever the backbone's dtype. The argmax does not depend
on `T`.

## 5. Answers

From a question's probabilities p (option order):

- `noul`: `{"type": "noul", "noul": p[yes]}`.
- `choice`: `{"type": "choice", "choice": the key of the first largest p, "confidence": (max p − 1/K) / (1 − 1/K)
  (1 when K = 1), "probabilities": {key: p}}`.
- `score`: `{"type": "score", "score": Σ level · p, "legend": {index: level text}, "probabilities": {index: p},
  "confidence": 1 − E|level − mode| / D, floored at 0}`, where mode is the first most likely level and D is the mean
  distance of the L levels from the centre, (1/L) · Σ |level − (L − 1)/2| (2/3 for three levels; the confidence is 1 when
  L = 1). For example, three levels with p = [0.5, 0.5, 0]: mode 0, E|level − mode| = 0.5, confidence 1 − 0.5 / (2/3) = 0.25.

Both confidences read p normalised to sum 1 (all zeros as uniform). Every reported number is rounded to 4 decimals.
The response holds `model` (the request's), `answers` (question id → answer), `usage` (`input_tokens`: the packed sequence's length; `output_tokens`: the tokens of the
serialised answers) and `latency_ms`. Other endpoints of `d1a.serving.serve` (`/v1/systemone/media`, `/permute`, `/separate`,
`/v1/feedback`, `/v1/models`) are described in its docstring.

## 6. Calibration

A checkpoint carries one temperature `T` (§4), fitted on held-out rows by `d1a.training.calibrate`; 1.0 means uncalibrated.
`D1A_TEMPERATURE` overrides it at load. Learning from outcomes (`d1a.learning.feedback`: the outcome calibrator, and the outcome
memory) is an optional serving layer on top of these answers, off unless configured; a port that implements only §1 to
§5 gives D1A's answers.

## 7. Checkpoints

A checkpoint is a directory or a Hugging Face repo (`owner/name@revision`). D1A reads three layouts, told apart by
`d1a_config.json`'s `format`:

**Training run, `d1a-torch` version 1** (written by `d1a.training.train` and `d1a.training.calibrate` from D1A 0.4):

- `adapter_config.json`, `adapter_model.safetensors`: a PEFT LoRA adapter on the base;
- `head.safetensors`: the head's tensors (`q.weight`, `q.bias`, `k.weight`, `k.bias`), fp32;
- `d1a_config.json`:

```json
{"format": "d1a-torch", "format_version": 1, "base": "google/gemma-4-E2B", "base_revision": "<commit or null>",
 "lora": 16, "head_dim": 256, "temperature": 1.52, "weights": "lora", "weights_dtype": "bf16",
 "special_embeddings": false, "option_isolation": false, "holdout": [], "head": "head.safetensors",
 "extra": {"args": {}, "suite_sha256": "...", "temperature_fit": {}}}
```

Readers must refuse a `format` or `format_version` they do not know, and `weights` other than `"lora"`. `extra` is
provenance (training arguments, suite hash, warm-start source, the temperature fit) and does not change answers.

**Training run saved before D1A 0.4:** `head.pt` alone (a PyTorch file of the same fields, read with weights-only
loading). D1A 0.4 also writes it beside `d1a_config.json`; when both are present they must agree on every field and on
the head, or loading stops. D1A 0.5 stops writing it; reading it stays.

**MLX export, `d1a-mlx` version 1** (`scripts/export_mlx.py`): the adapter merged into the base and saved by mlx-lm
(`config.json`, `model*.safetensors`, perhaps quantized), `head.safetensors` (fp32), the tokenizer files, and a
`d1a_config.json` with `format`, `format_version`, `base`, `base_revision`, `source`, `source_revision`,
`adapter_sha256`, `lora`, `head_dim`, `hidden_size`, `temperature`, `leading_ids`, `bos_id`, `delimiter_ids`, `pad_id`,
`dtype`, `quantization` and `head`. The token layout it records is checked against its tokenizer at load.

## 8. Conformance

`tests/golden/tiny-gemma4/` holds a tiny Gemma 4 base and checkpoint (in git) and `golden.json` (format `d1a-golden`,
version 1): 40 requests and 154 questions, each record with its System One `request`, the record it becomes (`input`),
its token `ids`, and per question the `qid`, `type`, `keys` and fp32 CPU `probs`. A conforming implementation must produce
the token ids exactly and every probability within 1e-5. `tests/test_conformance.py` checks this through the model,
through `d1a.serving.serve`, and through `scripts/golden_vectors.py compare`, which compares any two answer sets (another backend,
another port) the same way.
