# Removed tools

Tools and options D1A inherited from Kev and removed. They were useful ideas, so this records what each did and how to bring it back,
in our own code or by restoring the file from git:

```bash
git show 1bce9a4:playground/src/components/playground.tsx   # the last commit that has them
git checkout 1bce9a4 -- playground tools/review              # or restore the whole folders
```

The demo app D1A uses is [jonpol01/d1a-playground](https://github.com/jonpol01/d1a-playground).

## `playground/`: the developer playground (removed October 2026)

A Next.js page for trying requests by hand against a running `d1a.serving.serve` (proxied under `/d1a`).

- **Free-form request editor.** A preset picker plus two text boxes:
  - the state, as plain text or JSON;
  - the questions, as JSON in the System One shape.
  
  It sends one `POST /v1/systemone` and shows each answer with its probabilities and the request's latency.
- **Packed vs separate.** It asks all questions in one request (packed) and each question in its own request
  (`POST /v1/systemone/separate`), then shows the two answers side by side with the probability difference per option.
  This checks that packing questions together does not change the answers.
- **Option-order permutation.** For one Choice question, it re-asks under 6 option orders (`POST /v1/systemone/permute`)
  and shows each order's probabilities, whether the top answer stays the same (`argmax_stable`), and the per-option
  spread. This checks position bias.
- **Chess** (`/chess`, `chess.js`): every move is a Choice question.
  - The state is the board (FEN plus move list), the options are the legal moves (SAN), and a Score question rates
    who is better, all in one request.
  - Modes: model vs model, you as White, you as Black.
  - It shows the top 8 moves with probabilities, and saves games in the browser.

**Still in D1A:** the `/v1/systemone/separate` and `/v1/systemone/permute` endpoints in `d1a.serving.serve` stay, so a new
page can be built on them without server work.

## `tools/review/`: the label-review page (removed October 2026)

A keyboard-first Vite page for one person to accept, relabel or drop AI-proposed labels, one item at a time. It runs
entirely in the browser and autosaves to localStorage, keyed by a hash of the file.

- **Input:** a `.jsonl` file, one item per line:
  `{id, document, source, question: {type, instructions, options}, proposed_label, label_origin, judges: [{model, label, rationale}], adjudication}`.
- **Keys:**

  | Key | Action |
  |---|---|
  | `Enter` / `y` | accept the proposed label |
  | `1`–`9`, `0`, `a`–`z` | relabel |
  | `x` | drop |
  | `n` | note |
  | `←` / `→` | move without deciding |
  | `u` | undo |
  | `e` | export |
  
- **Filters:** undecided, all, disagree (a judge differs from the proposed label), decided. Spot-check mode is a seeded
  shuffle (seed 42) of the first N items (default 50).
- **Export:** `reviews.jsonl`, one line per decided item:
  `{id, verdict: accept|relabel|drop, label, proposed_label, note, reviewed_at}`. The header shows the agreement rate,
  accept / (accept + relabel).

A good fit for reviewing D1A training labels, for example the PR-labeler data or Visionkit OK/NG photos, if we need
that again.

## Training options (removed October 2026)

Research options of `d1a.training.train` that no D1A checkpoint used (every `training_config.json` of D1A-E2B, D1A-E4B and
the routing model leaves them at their defaults, which turned them off). They are in commit `00371c1` and earlier:

```bash
git show 00371c1:d1a/training/train.py   # question_loss, anchor_loss, permutation_kl, batch_loss, parse_args
git show 00371c1:d1a/backends/torch.py   # encode(option_isolation=...), branch_mask_batch(opts=...)
```

| Option | What it did |
|---|---|
| `--label_smoothing ε` | Hard-label cross-entropy with label smoothing (`F.cross_entropy(..., label_smoothing=ε)`); soft targets unchanged. |
| `--brier_w w` | Adds `w × Σ(softmax(z) − one_hot(y))²` to the hard-label cross-entropy. |
| `--focal_gamma γ` | Multiplies the hard-label cross-entropy by `(1 − p_y)^γ`. At most one of these three at a time. |
| `--ord_w w` | For Score questions, adds `w ×` the ranked probability score: the mean squared gap between the predicted and the observed CDF over the ordered levels. |
| `--perm_kl w`, `--perm_frac f` | For a fraction `f` of records with a Choice of 3+ options, a second forward pass with the options shuffled, and `w ×` the symmetric KL between the two predictions (mapped back to the original order). |
| `--anchor file`, `--anchor_w w`, `--anchor_sources` | `w × KL(base ‖ model)` toward a frozen base model's zero-shot distribution per question (a JSON of `{record_id: {qid: {key: p}}}` targets; questions whose option set changed are skipped). Its builder, `d1a.anchors`, was already removed. |
| `--option_isolation 1` | Every option span is its own sub-branch: it sees the state, the instruction and itself only, all spans share position ids, and `<decide>` sits after the longest. Answers are permutation-invariant by construction. Packed mask only, so not on Qwen3.5, MLX or the media server. A checkpoint trained with it is now refused at load. |
| `--special_embeddings 1` | Also trains the 5 delimiter tokens' embeddings (PEFT `trainable_token_indices`). Such adapters still load on torch; MLX refuses them as before. |

## Full-weight training (removed October 2026)

`d1a.training.train --full_ft 1` trained every backbone weight instead of a LoRA. No D1A model used it: D1A trains LoRA adapters
on one GPU (an L4), and a whole-backbone run of E4B wants far more memory. It is in commit `2a9be28` and earlier:

```bash
git show 2a9be28:d1a/full_ft.py                      # MasterAdamW, FSDP2 sharding, resume and snapshot writers
git show 2a9be28:scripts/interpolate_checkpoint.py   # WiSE-FT blends of full-weight checkpoints
git show 2a9be28:scripts/merge_lora_checkpoint.py    # a LoRA merged into a full-weight checkpoint, to start full-weight training
```

- **Optimizer** (`MasterAdamW`): bf16 working weights, with fp32 master weights and AdamW moments kept in host memory
  on one GPU, streamed one tensor at a time, clipped by the global norm and refusing a non-finite one.
- **Several GPUs**: `torchrun` + FSDP2 over the decoder layers, each rank holding its shard's masters and moments, with
  the epoch split evenly across ranks (`rank_share`) and micro-batches balanced across them.
- **Snapshots** (`--snapshot_fractions`, `--snapshot_every_steps`, `--snapshot_dir`): loadable checkpoints at chosen
  steps, kept beside the run and completed on resume.
- **Checkpoints**: `save_pretrained` of the whole backbone plus `head.pt` with `weights: "full"`; such a checkpoint is now
  refused at load. LoRA scaling at load time (`LoadOptions.lora_scale`, `D1A_LORA_SCALE`) still gives WiSE-FT-style
  interpolation for adapters.

## Kev's training-history suites (removed October 2026)

Frozen suites Kev trained or screened rounds on, which nothing D1A trains or evaluates on depends on: `decision-v1`,
`public-pool-v5`, `public-pool-v6`, `round3`, `round4`, `round5`, `round6`, `round10`, `round15`, `sft-v1`, `sft-v2`,
`sft-v2-r21`, `sft-v2-r22`, `sft-v2-r25`, `sft-v2-r26`, `transfer-v1`, `v3`, `v5`, `v6`, `v8`. Their manifests, and the
builders that only wrote them (`build_long_states`, `build_soft_targets`, `freeze_calibration_audit`), are in commit
`47a8cc2` and earlier; most large partitions also stay on Kev's public mirror `jaredpalmer/kev-suites` at `cc4bac8`.

```bash
git checkout 47a8cc2 -- evals/round4 scripts/build_long_states.py   # for example
```

Kev's `external` (semif-v1, typesafe-v1, wanli-v1, wanli-v2) and `night2` were not dropped: they are D1A suites now,
byte for byte (`evals/d1a/external`, `evals/d1a/night2`), and their builders (`freeze_semif`, `freeze_semif_external`,
`build_night2_data`) left with the originals.

## Kev's suite builders (removed October 2026)

The scripts that generated the frozen suites D1A still evaluates on: `build_hard_v1.py` with `hard_v1_common.py`,
`hard_v1_families.py`, `hard_v1_numeric.py` and `hard_v1_policy.py` (hard-v1: rule-engine families with computed labels),
`build_breadth_v1.py` (breadth-v1), `build_devtools_v1.py` with `devtools_v1_licences.json` (devtools-v1),
`build_longdoc_v1.py` with `longdoc_v1_synthetic.py` (longdoc-v1), `build_documents_v1.py`, `build_documents_v2.py`,
`freeze_documents_v1.py` and `label_documents_v1.py` (documents-v1/v2: candidate collection, LLM adjudication, freezing),
`build_binding_diagnostic.py`, and `d1a/transfer_v9.py` (transfer-v9). About 6,900 lines, 93-100% Kev's, with their
builder tests. D1A never regenerates these suites: it loads them as data, and `tests/test_frozen_suites.py` checks every
partition in git against the sha256 and record count its manifest pins.

To rebuild a suite, use the builder that wrote it:

| Suite | Builder source |
|---|---|
| longdoc-v1 | Kev at `a0255cedcb`: its `build_longdoc_v1.py` and `longdoc_v1_synthetic.py` match the manifest's sha256 |
| hard-v1 | no public commit matches the manifest's builder hashes (frozen from uncommitted code); the nearest is Kev at `0fe8fc9` or D1A at `0c601b2a` |
| breadth-v1, devtools-v1, documents-v1/v2, transfer-v9 | the manifests do not pin builder hashes; Kev at `0fe8fc9` or D1A at `0c601b2a` |

```bash
git checkout 0c601b2a -- scripts/build_hard_v1.py scripts/hard_v1_common.py scripts/hard_v1_families.py \
    scripts/hard_v1_numeric.py scripts/hard_v1_policy.py tests/test_hard_v1.py      # for example
```

## `d1a.eval.suite`'s suite freezer and the policy generators (removed October 2026)

The last suite builder, inside the package, and the code only it (and the builders above) called:

- **`python -m d1a.eval.suite`** (`freeze`, `main`, with `select_unique`, `contrast_cases`, `case_copy` and
  `training_state_hashes`). It froze a suite from the public datasets in `d1a.training.data` (`--sources`, `--transfer`,
  `--exclude-states-from`): per source, a deterministic sample deduplicated on normalised state text, admitted only if it
  encodes under both pinned Qwen tokenizers with branch headroom, split train/calibration/development/test, plus three
  contrast variants per clean record with a Choice of 3+ options (`none_present`, `none_absent` with the label moved to a new "None of these"
  option, and `permuted` with the options shuffled), and a manifest with dataset and base revisions and code hashes. It
  wrote decision-v2, transfer-v2, public-pool-v4 and smoke-v1. `semantic_hash` (an order-insensitive state hash for policy
  records) was already unused.
- **`d1a/contrastive.py`**: eleven policy families (return windows, spend thresholds, authorization, age eligibility,
  quantity limits, and six Score-threshold families such as deadlines, warranty claims and late fees) that generate minimal pairs, the same case with one
  sentence changed so the label flips, each pair checked by code: removing an evidence sentence must make the label
  undetermined, and removing a filler sentence must not change it (`--contrastive-pairs`, `--contrastive-holdout`).
  Its `paired_flip` metric stays, now in `d1a/eval/benchmark.py`, because the benchmark still reports it on those suites.
- **`d1a/training/composition.py`'s generator**: random rule trees over typed atoms (thresholds, ranges, matches, elapsed days,
  flags), rendered as policy text in several styles, with labels computed from the facts and four-record groups checked
  for invariance. The study harness used it for decision-v4 and decision-v7. The rule shapes and structure keys stay,
  because `d1a.eval.suite.validate_training` uses them to refuse held-out compositional structures in training.

- **`d1a/training/data.py`'s other converters**: trec, dbpedia14, emotion, imdb, amazon, qnli, tweet_offensive, mmlu, paws,
  sciq, arc, openbookqa and csqa (`ALL_SOURCES`, `ALL_REPOS`, `TRAINABLE`, the `TRANSFER_*` subsets, and the
  parquet-branch pin `trec` needed). Only the freezer's `--sources`/`--transfer` read them; `d1a.training.train`'s `build()`
  default is the six `SOURCES`, unchanged. `EVAL_ONLY` stays: training still refuses those sources' records.
- **`scripts/longdoc_serving.py`**: long-document serving cost on CUDA (latency, peak and resident memory per state-length
  bucket on longdoc-v1 development records, one cached repeat per bucket), run on an H200 through the removed Modal app.

The suites stay as data, pinned by their manifests. Their `code_hashes` name Kev's files, so rebuild a suite with Kev's
code at the commit the table above names. To bring back D1A's last copy of these files:

```bash
git show aeaa97c0:d1a/eval/suite.py                      # freeze, main and their helpers
git checkout aeaa97c0 -- d1a/contrastive.py d1a/training/composition.py tests/test_generators.py
git checkout aeaa97c0 -- d1a/training/data.py scripts/longdoc_serving.py
```
