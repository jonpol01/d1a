"""The decision model: a causal language-model backbone reads one request in a single pass, and a pointer head scores
each question's options against the question.

A request is packed as [leading ids (Gemma's <bos>)] <state> state, then per question <q> instructions (<opt> option
</opt>)... <decide> (encode). Questions never see each other: under the block-causal mask every token attends to the
state and to earlier tokens of its own question only (branch_mask). The head reads the hidden state at <decide> and at
each option's </opt> and scores them by a scaled dot product (PointerHead); a softmax over a question's options gives
its probabilities.

The same answers come from three forms, chosen per backbone and length: the packed pass under the block-causal mask;
causal rows, one per question, each the state plus that question's branch (rows_of; hybrid backbones, whose recurrent
layers cannot honour the mask, always use them); and, when serving, the state run once into a cache that the questions
continue from (probs_and_prefix, probs_with_prefix).
"""
import copy, functools, math, os, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import LinearAttentionCacheLayerMixin
from .backbone import Attention, for_config

# The five delimiters (state, q, opt, /opt, decide) are existing, rarely used special tokens, so no embedding rows are
# added; LoRA adapts their meaning. Qwen's fill-in-the-middle and box tokens; Gemma 4 has none of those (they would be
# <unk>), but its reserved <unusedN> rows are distinct unit-norm embeddings caller text cannot produce.
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
GEMMA_SPECIAL = ["<unused0>", "<unused1>", "<unused2>", "<unused3>", "<unused4>"]
DELIMITERS = (SPECIAL, GEMMA_SPECIAL)
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

# The training context: state tokens, tokens per question branch, and the whole packed record. Frozen suites were admitted
# under it (d1a.suite) and training applies it to records built on the fly, so training and evaluation see the same population.
MAX_STATE, MAX_BRANCH, MAX_PACKED = 384, 1024, 2048
# The serving context (d1a.serve): a state of up to 64k tokens (twice Jev's 32k; the Qwen3.5 / Qwen3.8 bases' window is
# 262k) and a question row (the state plus its branch, the encoder's max_branch) of up to 8k tokens more.
SERVE_MAX_STATE = 65536
SERVE_MAX_BRANCH = SERVE_MAX_STATE + 8192
SERVE_MAX_PACKED = SERVE_MAX_STATE + SERVE_MAX_BRANCH
# The tokens one inference pass holds (rows_per_pass), and the longest request an attention-only backbone runs packed
# (rows_form: its block-causal mask is L x L); longer ones run as rows.
ROW_PASS_TOKENS = 16384
# The longest state a checkpoint may train on (d1a.train --max_state): the longest one served, which still leaves every
# question its training branch budget (training_context(MAX_TRAIN_STATE)["max_branch"] <= SERVE_MAX_BRANCH).
MAX_TRAIN_STATE = SERVE_MAX_STATE
# The limits before 64k states: the suites frozen until then were admitted under these and record them (d1a.suite.SERVING_CONTEXT_8K).
SERVE_MAX_STATE_8K, SERVE_MAX_BRANCH_8K, MAX_TRAIN_STATE_8K = 8192, 8192, 7552


def training_context(max_state=MAX_STATE):
    """The encoder limits for training with the state limit raised to `max_state`: the row (state + one branch) and the
    packed limits grow by the same amount, so every question keeps its budget. training_context() is the default context
    (d1a.suite.CONTEXT without `truncate`)."""
    if not MAX_STATE <= max_state <= MAX_TRAIN_STATE:
        raise ValueError(f"max_state must be in [{MAX_STATE}, {MAX_TRAIN_STATE}]")
    lift = max_state - MAX_STATE
    return {"max_state": max_state, "max_branch": MAX_BRANCH + lift, "max_packed": MAX_PACKED + lift}


def rows_per_pass(rows, prefix_len=0, budget=ROW_PASS_TOKENS):
    """How many causal rows one inference pass takes: as many as fit `budget` tokens, counting the cached state each row
    carries (prefix_len) and its own tokens; at least one. A pass's memory is then bounded by one maximal row however many
    questions a request has, and rows are independent, so the answers do not depend on the split (Kev-4B on MLX, a
    4.8k-token state with 64 questions: 24.7 GB peak in one pass, 8.9 GB a row at a time, and faster)."""
    return max(1, budget // (prefix_len + max(len(r) for r in rows)))


class ContextOverflow(ValueError):
    """A record does not encode within its context (state, branch or packed limit). d1a.serve answers it with a 422; the
    benchmark counts it as a rejected record for suites scored as published (skip_overlong)."""


def load_tokenizer(name, revision=None):
    return AutoTokenizer.from_pretrained(name, revision=revision)


# --- encoding -----------------------------------------------------------------------------------------------------------

@functools.cache
def layout(tok):
    """(leading ids, delimiter ids, escape) of a tokenizer: the ids it puts before any text (Gemma's <bos>, which its
    attention relies on; none for Qwen), the first DELIMITERS set whose five tokens all exist in its vocabulary, and a
    pattern for its special tokens not of the `<|name|>` form user_tokens already escapes (Gemma's <bos>, <pad>, <|turn>
    ...; None for Qwen)."""
    leading = tok("", add_special_tokens=True).input_ids
    others = sorted({t for t in getattr(tok, "all_special_tokens", []) if not _SPECIAL_RE.fullmatch(t)}, key=len, reverse=True)
    escape = re.compile("|".join(map(re.escape, others))) if others else None
    unk = getattr(tok, "unk_token_id", None)
    for names in DELIMITERS:
        ids = [tok.convert_tokens_to_ids(n) for n in names]
        if None not in ids and (unk is None or unk not in ids):
            return leading, ids, escape
    raise ValueError(f"no delimiter set in d1a.model.DELIMITERS exists in the {type(tok).__name__} vocabulary")


def delimiter_ids(tok):
    """The five delimiter ids (state, q, opt, /opt, decide) this tokenizer encodes with."""
    return layout(tok)[1]


def pad_id(tok):
    """The id rows are right-padded with (never attended to): the tokenizer's own, else 0."""
    return tok.pad_token_id if tok.pad_token_id is not None else 0


def user_tokens(tok, text):
    """Caller text tokenized so it can never produce a delimiter or control token, which keeps option boundaries
    unforgeable. The fast tokenizer ignores split_special_tokens, so `<|name|>` becomes `<¦name¦>` before tokenizing, and
    any other special token of this tokenizer (layout) gets a `¦` after its first character."""
    escape = layout(tok)[2]
    if escape is not None:
        text = escape.sub(lambda m: m[0][0] + "¦" + m[0][1:], text)
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False):
    """Pack one record: [leading ids] <state> state, then per question <q> instructions (<opt> option </opt>)... <decide>.

    -> {"ids", "seg" (0: the state, k: question k), "pos" (each question's positions continue from the state), "decide_idx"
    (each question's <decide>), "opt_idx" (each question's </opt> per option), "labels", "state_truncated"}. A state past
    max_state is cut (or refused when strict); a branch that does not fit its row (max_branch with the state) is refused."""
    state = user_tokens(tok, rec["state"])
    leading, (s_id, q_id, o_id, c_id, d_id), _ = layout(tok)
    head = leading + [s_id]
    if strict and len(state) + len(head) > max_state:
        raise ContextOverflow(f"state exceeds {max_state} tokens: {len(state) + len(head)}")
    prefix = head + state[: max_state - len(head)]
    ids, seg, pos = list(prefix), [0] * len(prefix), list(range(len(prefix)))
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        branch = instr + [t for span in spans for t in span] + [d_id]
        if len(branch) > max_branch - len(prefix):
            raise ContextOverflow(f"branch too long: {len(branch)} tokens with a {len(prefix)}-token state (row limit {max_branch})")
        start = len(ids)
        ends, cursor = [], len(instr)
        for span in spans:
            cursor += len(span)
            ends.append(cursor - 1)
        ids += branch
        seg += [k] * len(branch)
        pos += list(range(len(prefix), len(prefix) + len(branch)))
        decide_idx.append(start + len(branch) - 1)
        opt_idx.append([start + e for e in ends])
    return {"ids": ids, "seg": seg, "pos": pos, "decide_idx": decide_idx, "opt_idx": opt_idx,
            "labels": [q["label"] for q in rec["questions"]], "state_truncated": len(state) + len(head) > max_state}


def fits(rec, *tokenizers, max_state=MAX_STATE, max_branch=MAX_BRANCH, max_packed=MAX_PACKED):
    """Whether the record encodes strictly, without truncation, within the context under every tokenizer given (frozen
    suites were admitted against the tokenizers of all their pinned bases)."""
    try:
        return all(len(encode(tok, rec, max_state=max_state, max_branch=max_branch, strict=True)["ids"]) <= max_packed for tok in tokenizers)
    except ValueError:
        return False


# --- masks and rows -----------------------------------------------------------------------------------------------------

def branch_mask(seg, device, dtype=torch.float32):
    """The block-causal rule for one record, additive [1, 1, L, L]: i attends to j iff j <= i and (seg[j] == 0 or
    seg[j] == seg[i])."""
    return branch_mask_batch([seg], device, dtype)


def branch_mask_batch(segs, device, dtype=torch.float32, length=None):
    """The block-causal mask of right-padded records, additive [B, 1, L, L]. Pad keys are masked for every query; a pad
    query keeps its diagonal so no row is fully masked (finfo.min, not -inf, so softmax stays finite). Real tokens never
    see pads: pads come after them (causal) and belong to no segment (-1)."""
    L = max(max(len(s) for s in segs), length or 0)
    s = torch.full((len(segs), L), -1, device=device)
    for b, seg in enumerate(segs):
        s[b, : len(seg)] = torch.tensor(seg, device=device)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    valid_key = (s != -1)[:, None, :]
    allow = causal[None] & same & valid_key
    allow = allow | torch.eye(L, dtype=torch.bool, device=device)[None]
    return torch.zeros(len(segs), L, L, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[:, None]


def branch_masks(encs, device, dtype, window, length=None):
    """A backbone's packed mask: branch_mask_batch, and for one with sliding layers (`window`) a {layer type: mask} whose
    sliding entry also drops keys `window` or more positions back. Distances are in position ids, which restart per
    branch after the state, so each question sees exactly the keys it would as its own causal row (rows_of)."""
    mask = branch_mask_batch([e["seg"] for e in encs], device, dtype=dtype, length=length)
    if window is None:
        return mask
    L = mask.shape[-1]
    p = torch.zeros((len(encs), L), dtype=torch.long, device=device)
    for b, e in enumerate(encs):
        p[b, : len(e["pos"])] = torch.tensor(e["pos"], device=device)
    far = (p[:, :, None] - p[:, None, :] >= window) & ~torch.eye(L, dtype=torch.bool, device=device)[None]
    return {"full_attention": mask, "sliding_attention": mask.masked_fill(far[:, None], torch.finfo(dtype).min)}


def rows_of(enc):
    """A packed encoding as (state ids, state positions, rows): rows[k] = {"ids", "pos", "decide", "opts"} holds question
    k's branch tokens, their positions (continuing the state's) and the readout offsets within the branch. The state then
    rows[k] as one causal row equals the packed form for that question on any architecture: the row holds exactly the
    tokens the question may attend to, at the same positions."""
    seg = enc["seg"]
    Ls = seg.count(0)
    rows, start = [], Ls
    for k, (d, oi) in enumerate(zip(enc["decide_idx"], enc["opt_idx"]), start=1):
        end = d + 1   # <decide> ends its branch
        if seg[start] != k or seg[end - 1] != k:
            raise ValueError("branch layout mismatch")
        rows.append({"ids": enc["ids"][start:end], "pos": enc["pos"][start:end], "decide": d - start, "opts": [o - start for o in oi]})
        start = end
    return enc["ids"][:Ls], enc["pos"][:Ls], rows


# --- the pointer head ---------------------------------------------------------------------------------------------------

class PointerHead(nn.Module):
    """Scores a question's options: a scaled dot product of the projected <decide> state (q) with each projected </opt>
    state (k). In eval mode the logits are divided by `temperature` (1.0: raw), the calibration a checkpoint carries
    (d1a.calibrate); training always sees 1, so a fitted value stays meaningful, and the argmax never changes."""

    def __init__(self, d, dp=256):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)   # dp: the pointer dimension
        self.scale = 1 / math.sqrt(dp)
        self.temperature = 1.0

    def _tempered(self, z):
        return z if self.training or self.temperature == 1.0 else z / self.temperature

    def forward(self, h_decide, h_opts):   # [d], [K, d] -> logits [K]
        return self._tempered((self.k(h_opts) @ self.q(h_decide)) * self.scale)

    def many(self, h_decide, h_opts, owner):   # [Q, d], [sum K, d], each option's question [sum K] -> logits [sum K]
        """forward() for many questions in one pass (serving batches)."""
        return self._tempered((self.k(h_opts) * self.q(h_decide)[owner]).sum(-1) * self.scale)


# What a loaded model exposes to d1a.serve, d1a.predictors and the Space: the scoring interface DecisionModel (torch) and
# d1a.mlx_model.MLXDecisionModel both implement. tests/test_mlx.py checks the MLX class against this list.
SCORING_INTERFACE = ("encode", "forward", "probs", "probs_and_prefix", "probs_with_prefix", "probs_batch", "eval",
                     "head", "backend", "dtype", "device", "hybrid", "prefix_min_tokens")

EAGER_STATES = 4   # long new states (past the graphed state pass) whose eager prefixes one batched run holds at once: each
                   # holds its state's keys, values and DeltaNet states (~0.3 GB for 2,200 tokens on Kev-27B)

LORA_TARGETS = {"all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "dense": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],   # "all" minus the DeltaNet projections on hybrids
                "attn": ["q_proj", "k_proj", "v_proj", "o_proj"], "qv": ["q_proj", "v_proj"]}


def probs_one(model, enc, prefix, keep):
    """-> (probabilities, the prefix to keep) for one request on any backend: on a cache hit the question rows on the
    cached prefix; otherwise one pass that also returns the prefix when it is to be kept, else the plain pass."""
    if prefix is not None:
        return model.probs_with_prefix(enc, prefix), prefix
    return model.probs_and_prefix(enc) if keep else (model.probs(enc), None)


# --- the model ----------------------------------------------------------------------------------------------------------

class DecisionModel(nn.Module):
    def __init__(self, name, tok, device, lora=None, revision=None, attn=None, head_dim=256, lora_targets="all", dtype=torch.float32):
        super().__init__()
        # The backbone only: no vocabulary head, D1A never generates text. Eager attention on MPS and the CPU (known good
        # with the float 4D mask), SDPA on CUDA (which accepts arbitrary additive masks). fp32 trains and gives the exact
        # numbers; bf16 is a serving option for large backbones.
        attn = attn or ("sdpa" if str(device).startswith("cuda") else "eager")
        self.lm = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype, attn_implementation=attn).model
        # a multimodal checkpoint (Gemma 4) loads as a wrapper around its text model: keep the text model only, so the vision
        # and audio towers are neither in memory nor matched by the LoRA target names
        self.lm = Attention.unwrap(self.lm)
        self.backbone = for_config(self.lm.config)   # d1a.backbone: what this family needs (form, masks, cache, LoRA names)
        self.pad_id = pad_id(tok)
        # Hybrid backbones (Qwen3.5: recurrent Gated DeltaNet layers) cannot honour the block-causal mask, so each question
        # runs as its own causal row continuing the state; attention-only ones keep the packed form (the two agree to fp32
        # noise, tests/test_model.py::test_rows_match_packed).
        self.hybrid = self.backbone.hybrid
        self.sliding_window = self.backbone.sliding_window   # Gemma 4: the packed form carries a second mask (branch_masks)
        if lora:
            from peft import LoraConfig, get_peft_model
            targets = LORA_TARGETS[lora_targets]
            if lora_targets in ("all", "attn"):
                targets = targets + list(self.backbone.lora_extra)   # e.g. Qwen3.5's Gated DeltaNet projections
            self.lm = get_peft_model(self.lm, LoraConfig(task_type="FEATURE_EXTRACTION", r=lora, lora_alpha=2 * lora, lora_dropout=0.05, target_modules=targets))
        self.head = PointerHead(self.lm.config.hidden_size, dp=head_dim)
        self.device = device
        self.to(device)

    backend = "torch"   # d1a.mlx_model.MLXDecisionModel is the other implementation of the scoring interface
    graphs = None       # d1a.cuda_graphs.CudaGraphs for a hybrid backbone's serving passes on CUDA (LoadOptions.cuda_graphs)

    @property
    def prefix_min_tokens(self):
        """d1a.serve caches a state prefix from this many state tokens. Attention-only backbones: 384, below which the
        branch-only pass is no faster than one packed pass on MPS (per-op overhead). Hybrid backbones: always, since their
        miss path would otherwise run the state once per question (Kev-0.8B bf16 on MPS, 5 questions: 1011 -> 413 ms)."""
        return self.backbone.prefix_min_tokens

    @property
    def dtype(self):
        return str(next(self.lm.parameters()).dtype).removeprefix("torch.")

    def encode(self, tok, rec, **kw):
        """encode(), as part of the scoring interface (MLXDecisionModel has its own)."""
        return encode(tok, rec, **kw)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # --- hidden states ------------------------------------------------------------------------------------------------

    def hidden(self, enc):
        return self.hidden_batch([enc])[0, : len(enc["ids"])]

    SHAPE_BUCKET = int(os.environ.get("D1A_SHAPE_BUCKET", "64"))   # MPS: pad sequences to a multiple of this (kernels warm per shape); 1 disables

    def _pad_rows(self, rows):
        """(ids, pos) token rows right-padded to [N, L] id and position tensors, with an [N, L] attention mask (1: a real
        token). Pads follow every real token and are masked keys, so they never change a real token's hidden state (parity
        measured exact). On MPS in eval mode L rounds up to a SHAPE_BUCKET multiple, so kernels are warmed per bucket."""
        L = max(len(ids) for ids, _ in rows)
        if str(self.device) == "mps" and not self.training:
            L = -(-L // self.SHAPE_BUCKET) * self.SHAPE_BUCKET
        ids = torch.full((len(rows), L), self.pad_id, device=self.device)
        pos = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        att = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        for i, (row_ids, row_pos) in enumerate(rows):
            ids[i, : len(row_ids)] = torch.tensor(row_ids, device=self.device)
            pos[i, : len(row_pos)] = torch.tensor(row_pos, device=self.device)
            att[i, : len(row_ids)] = 1
        return ids, pos, att

    def hidden_batch(self, encs, picks=None):
        """[B, L, d] hidden states of right-padded records under the packed block-causal mask; with picks ([B, P]
        positions), [B, P, d] at those positions, which the backbone may compute more cheaply (Gemma 4)."""
        ids, pos, _ = self._pad_rows([(e["ids"], e["pos"]) for e in encs])
        mask = self._packed_mask(encs, length=ids.shape[1])
        if picks is not None and (h := self.backbone.picked_hidden(self.lm, ids, pos, mask, picks)) is not None:
            return h.float()
        h = self.lm(input_ids=ids, position_ids=pos, attention_mask=mask).last_hidden_state.float()   # the head stays fp32
        return h if picks is None else h.gather(1, picks[..., None].expand(-1, -1, h.shape[-1]))

    def _packed_mask(self, encs, length=None):
        return branch_masks(encs, self.device, next(self.lm.parameters()).dtype, self.sliding_window, length=length)

    def _new_cache(self):
        """An empty cache for a state prefix. Hybrid backbones need the layer types (each DeltaNet layer's recurrent and
        conv states); sliding-window ones must NOT get them: a sliding layer's cache keeps only the last window, but the
        packed branch pass hands those layers the full-length mask (branch_masks), so every layer caches the whole state."""
        return self.backbone.new_cache()

    def _readout(self, h, enc):
        return [self.head(h[d], h[torch.tensor(oi, device=self.device)]) for d, oi in zip(enc["decide_idx"], enc["opt_idx"])]

    def rows_form(self, encs):
        """Whether these records run as causal rows: always on a hybrid backbone (its recurrent layers cannot honour the
        packed mask), and on an attention-only one when a packed sequence would pass ROW_PASS_TOKENS (its L x L mask grows
        with the number of questions). The forms agree (tests/test_model.py::test_rows_match_packed)."""
        return self.hybrid or any(len(e["ids"]) > ROW_PASS_TOKENS for e in encs)

    def _rows_hidden(self, rows, cache=None, prefix_len=0):
        """Hidden states of causal token rows, one [L_i, d] tensor per row, rows_per_pass at a time in eval mode (training
        keeps one batch: its batches are small and autograd needs the whole graph). With `cache`, the rows are branches
        continuing the cached state: in eager mode the cache is copied once per chunk, leaving the caller's prefix as it
        was, and the cached tokens are marked real in the attention mask."""
        chunk = len(rows) if self.training else rows_per_pass([ids for ids, _ in rows], prefix_len)
        out = []
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            ids, pos, att = self._pad_rows(part)
            past = {}
            if cache is not None:
                past = {"past_key_values": self._replica(cache, len(part)), "use_cache": True}
                att = torch.cat([torch.ones((len(part), prefix_len), dtype=torch.long, device=self.device), att], 1)
            h = self.lm(input_ids=ids, position_ids=pos, attention_mask=att, **past).last_hidden_state.float()
            out += [h[i, : len(row_ids)] for i, (row_ids, _) in enumerate(part)]
        return out

    def _replica(self, cache, rows):
        """A copy of `cache` widened to `rows` rows, sharing nothing the rows will write (a DeltaNet layer's states are
        dicts the forward updates in place)."""
        replica = copy.copy(cache)
        replica.layers = [copy.copy(layer) for layer in cache.layers]
        for source, target in zip(cache.layers, replica.layers):
            if isinstance(source, LinearAttentionCacheLayerMixin):
                target.conv_states = source.conv_states.copy()
                target.recurrent_states = source.recurrent_states.copy()
                target.is_conv_states_initialized = source.is_conv_states_initialized.copy()
                target.is_recurrent_states_initialized = source.is_recurrent_states_initialized.copy()
                target.has_previous_state = source.has_previous_state.copy()
                target.conv_kernel_size = source.conv_kernel_size.copy()
        replica.reorder_cache(torch.zeros(rows, dtype=torch.long, device=self.device))
        return replica

    # --- logits (training and the exact path) -------------------------------------------------------------------------

    def forward_rows_batch(self, encs, shared_prefix=False):
        """The row form: each question of each record is one causal row, the state plus its branch, giving forward_batch's
        nested logits. Rows are independent, so isolation is exact; the state is computed again per row, which training
        accepts (serving and probs() use the prefix cache instead). shared_prefix (training, hybrid Qwen3.5): the same rows
        with each state run once and its branches continuing from it, gradients included (d1a.shared_prefix)."""
        if shared_prefix:
            from .shared_prefix import branch_hidden   # here, not at the top: the Space vendors model.py alone
            splits = [rows_of(e) for e in encs]
            return [[self.head(h[r["decide"]], h[torch.tensor(r["opts"], device=self.device)]) for h, r in zip(hs, rows)]
                    for hs, (_, _, rows) in zip(branch_hidden(self.lm, splits, self.pad_id, self.device), splits)]
        rows, readouts = [], []   # one causal row per question; readouts[i] = (record, <decide> offset, option offsets)
        for b, e in enumerate(encs):
            state, state_pos, branches = rows_of(e)
            for r in branches:
                rows.append((state + r["ids"], state_pos + r["pos"]))
                readouts.append((b, len(state) + r["decide"], [len(state) + o for o in r["opts"]]))
        out = [[] for _ in encs]
        for h, (b, d, oi) in zip(self._rows_hidden(rows), readouts):
            out[b].append(self.head(h[d], h[torch.tensor(oi, device=self.device)]))
        return out

    def forward(self, enc):
        """The logits of each question of one record."""
        return self.forward_batch([enc])[0]

    def forward_batch(self, encs, shared_prefix=False):
        """Per record, per question, the logits: by rows (rows_form) or under the packed block-causal mask, which already
        runs each state once (shared_prefix does that for a hybrid backbone's rows)."""
        if self.rows_form(encs):
            return self.forward_rows_batch(encs, shared_prefix and self.hybrid)
        # only the positions the head reads: each question's <decide> and its options' </opt>
        picked = [sorted({i for d, oi in zip(e["decide_idx"], e["opt_idx"]) for i in (d, *oi)}) for e in encs]
        width = max(map(len, picked))
        picks = torch.tensor([p + [0] * (width - len(p)) for p in picked], device=self.device)
        hs = self.hidden_batch(encs, picks)
        slot = [{i: j for j, i in enumerate(p)} for p in picked]
        return [[self.head(hs[b, slot[b][d]], hs[b, torch.tensor([slot[b][i] for i in oi], device=self.device)])
                 for d, oi in zip(e["decide_idx"], e["opt_idx"])] for b, e in enumerate(encs)]

    @torch.no_grad()
    def probs(self, enc):
        """Each question's probabilities. A record that runs as rows (every record on a hybrid backbone) takes the serving
        miss path, the state once and then the question rows from its cache; forward() keeps the plain row form, which runs
        the state per question; the two agree to fp32 rounding (#77)."""
        if self.rows_form([enc]):
            return self.probs_and_prefix(enc)[0]
        return [F.softmax(z, -1).cpu() for z in self.forward(enc)]

    # --- serving: the state once, the questions from its cache ---------------------------------------------------------
    # Exact by construction: branch tokens never attend across questions (block-causal) and the state never sees the
    # branches (causal), so the state's hidden states and keys/values are the same with or without them.

    def _branch_rows_from_prefix(self, enc, cache):
        """The branches as causal rows continuing the cached state (forward_rows_batch's layout without the state)."""
        state, _, rows = rows_of(enc)
        hs = self._rows_hidden([(r["ids"], r["pos"]) for r in rows], cache=cache, prefix_len=len(state))
        ps = [F.softmax(self.head(h[r["decide"]], h[torch.tensor(r["opts"], device=self.device)]), -1) for h, r in zip(hs, rows)]
        return list(torch.cat(ps).cpu().split([len(p) for p in ps]))   # one device sync per request, not per question

    @torch.no_grad()
    def prefix(self, enc):
        """Run the state alone. -> (state tokens, its cache, its hidden states [Ls, d] in fp32 or None). Only the packed pass
        (probs_with_prefix) reads the hidden states, so a hybrid backbone, which always runs rows, keeps None, as the graphed
        prefixes do (an fp32 copy is 160 MiB for 8k tokens on Kev-27B); an attention-only one keeps them even when this
        record runs as rows, since the same state with fewer questions may be packed."""
        Ls = enc["seg"].count(0)
        ids = torch.tensor([enc["ids"][:Ls]], device=self.device)
        pos = torch.tensor([enc["pos"][:Ls]], device=self.device)
        out = self.lm(input_ids=ids, position_ids=pos, past_key_values=self._new_cache(), use_cache=True)
        return Ls, out.past_key_values, None if self.hybrid else out.last_hidden_state[0].float()

    @torch.no_grad()
    def probs_and_prefix(self, enc):
        """One full pass that also returns the state prefix (the cache cropped to the state, and the state's hidden states
        or None, as prefix() keeps them), so a cache miss costs one pass, not two."""
        Ls = enc["seg"].count(0)
        if self.rows_form([enc]):
            # recurrent layers cannot be cropped back to the state (and an over-long packed pass is what rows avoid), so a
            # miss here is a state pass, kept as the prefix, plus the branch rows
            Ls, cache, h_state = self.prefix(enc)
            return self._branch_rows_from_prefix(enc, cache), (Ls, cache, h_state)
        ids = torch.tensor([enc["ids"]], device=self.device)
        pos = torch.tensor([enc["pos"]], device=self.device)
        out = self.lm(input_ids=ids, position_ids=pos, attention_mask=self._packed_mask([enc]), past_key_values=self._new_cache(), use_cache=True)
        h = out.last_hidden_state[0].float()
        out.past_key_values.crop(-(len(enc["ids"]) - Ls))   # keep the state only (a negative crop drops that many trailing tokens)
        return [F.softmax(z, -1).cpu() for z in self._readout(h, enc)], (Ls, out.past_key_values, h[:Ls].clone())

    @torch.no_grad()
    def probs_with_prefix(self, enc, prefix):
        """probs() for a record whose state equals the cached prefix's: only the branches run, and the cache is cropped
        back to the state afterwards so it can be used again."""
        Ls, cache, h_state = prefix
        if enc["seg"].count(0) != Ls:
            raise ValueError("prefix does not match this record's state")
        if self.rows_form([enc]):
            return self._branch_rows_from_prefix(enc, cache)
        ids = torch.tensor([enc["ids"][Ls:]], device=self.device)
        pos = torch.tensor([enc["pos"][Ls:]], device=self.device)
        mask = self._packed_mask([enc])
        mask = {k: m[:, :, Ls:, :] for k, m in mask.items()} if isinstance(mask, dict) else mask[:, :, Ls:, :]
        try:
            out = self.lm(input_ids=ids, position_ids=pos, past_key_values=cache, attention_mask=mask, use_cache=True)
            h = torch.cat([h_state, out.last_hidden_state[0].float()], 0)
        finally:
            cache.crop(-(len(enc["ids"]) - Ls))
        return [F.softmax(z, -1).cpu() for z in self._readout(h, enc)]

    @torch.no_grad()
    def probs_batch(self, encs, prefixes, keep):
        """d1a.serve's model thread, several requests at once. prefixes[i]: request i's cached prefix or None; keep[i]:
        whether to return its new prefix. -> (probabilities per request, prefix per request: the cached one, the new one if
        kept, else None). With CUDA graphs, the requests the graphed passes admit run together (d1a.cuda_graphs: shared state
        and row passes; a state too long for the graphed state pass gets its own eager pass first); the rest, and every
        other backend, one at a time. Rows are independent, so a request's answers do not depend on its batch."""
        splits = [rows_of(e) for e in encs]

        def admitted(i, cached):
            state, _, rows = splits[i]
            return self.graphs is not None and self.graphs.admits(len(state), [len(r["ids"]) for r in rows], cached)
        long = {i for i in range(len(encs)) if prefixes[i] is None and not admitted(i, False) and admitted(i, True)}
        batched = [i for i in range(len(encs)) if i in long or admitted(i, prefixes[i] is not None)]
        out = {i: probs_one(self, encs[i], prefixes[i], keep[i]) for i in sorted(set(range(len(encs))) - set(batched))}
        runs, held = [[]], 0   # graphed runs, each holding at most EAGER_STATES long states' eager prefixes at once
        for i in batched:
            if i in long and held == EAGER_STATES:
                runs.append([])
                held = 0
            runs[-1].append(i)
            held += i in long
        from .cuda_graphs import Request   # here, not at the top: the Space vendors model.py without cuda_graphs.py
        for run in filter(None, runs):
            pre = {i: self.prefix(encs[i]) if i in long else prefixes[i] for i in run}
            X, caches = self.graphs.run([Request(splits[i][0], splits[i][1], [(r["ids"], r["pos"]) for r in splits[i][2]],
                                                 None if pre[i] is None else pre[i][1], keep[i], [[r["decide"], *r["opts"]] for r in splits[i][2]])
                                         for i in run])   # each question's picks: its <decide> position, then its options
            ps = iter(self._readout_many(X, [len(r["opts"]) for i in run for r in splits[i][2]]))
            for i, cache in zip(run, caches):
                made = pre[i] if i in long else None if cache is None else (len(splits[i][0]), cache, None)
                out[i] = [next(ps) for _ in splits[i][2]], prefixes[i] or (made if keep[i] else None)
        return [out[i][0] for i in range(len(encs))], [out[i][1] for i in range(len(encs))]

    def _readout_many(self, X, ks):
        """Many questions' probabilities from their picked hidden states X, laid out per question as its <decide> then its
        ks[q] options, in a fixed number of kernels and one device sync. -> one tensor per question."""
        owner = [q for q, k in enumerate(ks) for _ in range(k)]
        slot = [j for k in ks for j in range(k)]
        starts = [0]
        for k in ks:
            starts.append(starts[-1] + 1 + k)
        dec = torch.tensor(starts[:-1]).to(self.device, non_blocking=True)
        opt, own, sl = torch.tensor([[starts[q] + 1 + j for q, j in zip(owner, slot)], owner, slot]).to(self.device, non_blocking=True)
        z = self.head.many(X[dec], X[opt], own)
        Z = torch.full((len(ks), max(ks)), float("-inf"), device=self.device).index_put_((own, sl), z)
        return torch.softmax(Z, -1)[own, sl].cpu().split(ks)
