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
import copy, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.cache_utils import LinearAttentionCacheLayerMixin
from d1a.backends.backbone import Attention, for_config
# Encoding and the head are the backend-independent core (#60); their names stay importable from here.
from d1a.core.encoding import (DELIMITERS, GEMMA_SPECIAL, MAX_BRANCH, MAX_PACKED, MAX_STATE, MAX_TRAIN_STATE, MAX_TRAIN_STATE_8K,  # noqa: F401
                               ROW_PASS_TOKENS, SERVE_MAX_BRANCH, SERVE_MAX_BRANCH_8K, SERVE_MAX_PACKED, SERVE_MAX_STATE,
                               SERVE_MAX_STATE_8K, SPECIAL, ContextOverflow, delimiter_ids, encode, fits, layout, load_tokenizer,
                               pad_id, rows_of, rows_per_pass, training_context, user_tokens)
from d1a.core.head import PointerHead  # noqa: F401

# --- masks -------------------------------------------------------------------------------------------------------------

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


# What a loaded model exposes to d1a.serving.serve, d1a.eval.predictors and the Space: the scoring interface DecisionModel (torch) and
# d1a.backends.mlx.MLXDecisionModel both implement. tests/test_mlx.py checks the MLX class against this list.
SCORING_INTERFACE = ("encode", "forward", "probs", "probs_and_prefix", "probs_with_prefix", "probs_batch", "eval",
                     "head", "backend", "dtype", "device", "hybrid", "prefix_min_tokens")

EAGER_STATES = 4   # long new states (past the graphed state pass) whose eager prefixes one batched run holds at once: each
                   # holds its state's keys, values and DeltaNet states (~0.3 GB for 2,200 tokens on Kev-27B)

LORA_TARGETS = {"all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "dense": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],   # "all" minus the DeltaNet projections on hybrids
                "attn": ["q_proj", "k_proj", "v_proj", "o_proj"], "qv": ["q_proj", "v_proj"]}


def temperature_of(model, enc):
    """The temperature `enc`'s probabilities are read at: the head's for the use case d1a.serving.serve put on it
    (enc["use_case"], PointerHead.temperature_for), the checkpoint's when it names none. Shared by both backends."""
    return model.head.temperature_for(enc.get("use_case"))


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
        self.backbone = for_config(self.lm.config)   # d1a.backends.backbone: what this family needs (form, masks, cache, LoRA names)
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

    backend = "torch"   # d1a.backends.mlx.MLXDecisionModel is the other implementation of the scoring interface
    graphs = None       # d1a.backends.cuda_graphs.CudaGraphs for a hybrid backbone's serving passes on CUDA (LoadOptions.cuda_graphs)

    @property
    def prefix_min_tokens(self):
        """d1a.serving.serve caches a state prefix from this many state tokens. Attention-only backbones: 384, below which the
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
        t = temperature_of(self, enc)
        return [self.head(h[d], h[torch.tensor(oi, device=self.device)], t) for d, oi in zip(enc["decide_idx"], enc["opt_idx"])]

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
        with each state run once and its branches continuing from it, gradients included (d1a.backends.shared_prefix)."""
        if shared_prefix:
            from d1a.backends.shared_prefix import branch_hidden   # here, not at the top: the Space vendors model.py alone
            splits = [rows_of(e) for e in encs]
            return [[self.head(h[r["decide"]], h[torch.tensor(r["opts"], device=self.device)], temperature_of(self, e)) for h, r in zip(hs, rows)]
                    for e, hs, (_, _, rows) in zip(encs, branch_hidden(self.lm, splits, self.pad_id, self.device), splits)]
        rows, readouts = [], []   # one causal row per question; readouts[i] = (record, <decide> offset, option offsets)
        for b, e in enumerate(encs):
            state, state_pos, branches = rows_of(e)
            for r in branches:
                rows.append((state + r["ids"], state_pos + r["pos"]))
                readouts.append((b, len(state) + r["decide"], [len(state) + o for o in r["opts"]]))
        out = [[] for _ in encs]
        for h, (b, d, oi) in zip(self._rows_hidden(rows), readouts):
            out[b].append(self.head(h[d], h[torch.tensor(oi, device=self.device)], temperature_of(self, encs[b])))
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
        return [[self.head(hs[b, slot[b][d]], hs[b, torch.tensor([slot[b][i] for i in oi], device=self.device)], temperature_of(self, e))
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
        t = temperature_of(self, enc)
        ps = [F.softmax(self.head(h[r["decide"]], h[torch.tensor(r["opts"], device=self.device)], t), -1) for h, r in zip(hs, rows)]
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
        """d1a.serving.serve's model thread, several requests at once. prefixes[i]: request i's cached prefix or None; keep[i]:
        whether to return its new prefix. -> (probabilities per request, prefix per request: the cached one, the new one if
        kept, else None). With CUDA graphs, the requests the graphed passes admit run together (d1a.backends.cuda_graphs: shared state
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
        from d1a.backends.cuda_graphs import Request   # here, not at the top: the Space vendors model.py without cuda_graphs.py
        for run in filter(None, runs):
            pre = {i: self.prefix(encs[i]) if i in long else prefixes[i] for i in run}
            X, caches = self.graphs.run([Request(splits[i][0], splits[i][1], [(r["ids"], r["pos"]) for r in splits[i][2]],
                                                 None if pre[i] is None else pre[i][1], keep[i], [[r["decide"], *r["opts"]] for r in splits[i][2]])
                                         for i in run])   # each question's picks: its <decide> position, then its options
            ps = iter(self._readout_many(X, [len(r["opts"]) for i in run for r in splits[i][2]],
                                         [temperature_of(self, encs[i]) for i in run for _ in splits[i][2]]))
            for i, cache in zip(run, caches):
                made = pre[i] if i in long else None if cache is None else (len(splits[i][0]), cache, None)
                out[i] = [next(ps) for _ in splits[i][2]], prefixes[i] or (made if keep[i] else None)
        return [out[i][0] for i in range(len(encs))], [out[i][1] for i in range(len(encs))]

    def _readout_many(self, X, ks, temperatures=None):
        """Many questions' probabilities from their picked hidden states X, laid out per question as its <decide> then its
        ks[q] options, in a fixed number of kernels and one device sync. temperatures: each question's (its request's,
        temperature_of); when they all equal the head's own, the batch is read exactly as one without use cases.
        -> one tensor per question."""
        owner = [q for q, k in enumerate(ks) for _ in range(k)]
        slot = [j for k in ks for j in range(k)]
        starts = [0]
        for k in ks:
            starts.append(starts[-1] + 1 + k)
        dec = torch.tensor(starts[:-1]).to(self.device, non_blocking=True)
        opt, own, sl = torch.tensor([[starts[q] + 1 + j for q, j in zip(owner, slot)], owner, slot]).to(self.device, non_blocking=True)
        t = None
        if temperatures is not None and any(x != self.head.temperature for x in temperatures):   # a batch mixing temperatures (#209)
            t = torch.tensor([temperatures[q] for q in owner], dtype=torch.float32).to(self.device, non_blocking=True)
        z = self.head.many(X[dec], X[opt], own, t)
        Z = torch.full((len(ks), max(ks)), float("-inf"), device=self.device).index_put_((own, sl), z)
        return torch.softmax(Z, -1)[own, sl].cpu().split(ks)
