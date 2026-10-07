"""Training through a shared state prefix, for the hybrid Qwen3.5 backbones (d1a.training.train --shared_prefix).

The row form (d1a.backends.torch.forward_rows_batch) trains a record with q questions as q causal rows of state + one
branch, so its state runs q times. Here the state runs once and every branch continues from it: state + the branches
instead of q x (state + branch). It is the same computation. A branch token sees the same tokens at the same positions,
and it reaches the state through what each layer leaves at the state's end, read the way the layer's own cache path reads
it when serving (d1a.backends.torch._branch_rows_from_prefix): attention keys and values, and for the Gated DeltaNet
layers the short convolution's window and the recurrent state (fla's chunked delta rule takes it as `initial_state` and
differentiates through it).

Two things differ from serving:
- Prefix keeps that per-layer state as plain tensors (transformers' cache layers copy into static buffers in place, which
  autograd cannot follow), so every branch's gradient reaches the state and adds up there;
- one call per decoder layer runs its state pass and then its branch pass, so gradient checkpointing (when the backbone
  has it on) recomputes both together and no cache outlives its layer or is written twice on replay.

A batch's states are padded on the left (padded positions are zeroed and stay zero: a zero input changes neither the
DeltaNet state, nor the conv window, nor the residual stream), its branches on the right.

Exactness: in fp32 with transformers' reference kernels it matches the row form to ~1e-5, logits and every gradient
(a tiny random Qwen3.5 in tests/test_unit.py, Qwen3.5-0.8B-Base in tests/test_model.py). On CUDA, fla's Triton kernels
round fp32 dot products like TF32 and states are left-padded here, so it rounds differently from the row form: as far
from the exact answer as the row form's own batchings are, with no systematic shift.
"""
import types

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class Prefix:
    """One decoder layer's view of the shared state. It answers the cache calls a Qwen3.5 layer makes: during the state
    pass it stores what the layer leaves (keys and values, or the conv window and the recurrent state), `branch` gathers
    that once per branch row, and the branch pass reads it. Nothing is changed in place."""
    record_past = True   # a one-token pass must not take the in-place decoding kernel

    def __init__(self):
        self.keys = self.values = self.conv = None
        self.recurrent_states = {}

    @property
    def layers(self):   # layers read cache.layers[layer_idx].recurrent_states[0]: every index is this prefix
        return self

    def __getitem__(self, layer_idx):
        return self

    def has_previous_state(self, layer_idx=None, state_idx=0):
        return self.conv is not None

    def update_conv_state(self, conv_input, layer_idx, conv_kernel_size, **kwargs):
        if self.conv is not None:   # the branch pass: its inputs follow the state's last ones
            return torch.cat([self.conv, conv_input], -1)
        missing = max(0, conv_kernel_size - conv_input.shape[-1])
        self.conv = F.pad(conv_input, (missing, 0))[..., -conv_kernel_size:]
        return conv_input

    def update_recurrent_state(self, recurrent_state, layer_idx, **kwargs):
        self.recurrent_states.setdefault(0, recurrent_state)   # the state's final one; a branch's own is not kept
        return recurrent_state

    def update(self, keys, values, layer_idx, *args, **kwargs):
        if self.keys is None:   # the state pass
            self.keys, self.values = keys, values
            return keys, values
        return torch.cat([self.keys, keys], -2), torch.cat([self.values, values], -2)

    def branch(self, owner):
        """Repeat the stored state for each branch row: owner[i] is the record of branch i."""
        def per_row(t):
            return None if t is None else t.index_select(0, owner)
        self.keys, self.values, self.conv = per_row(self.keys), per_row(self.values), per_row(self.conv)
        self.recurrent_states = {k: per_row(v) for k, v in self.recurrent_states.items()}


def _masks(allow, dtype, attn):
    """The attention mask from `allow` [B, L, L]: boolean for SDPA (whose float masks must be in the query's dtype, which
    autocast changes), additive in `dtype` for eager attention."""
    allow = allow[:, None]
    if attn == "sdpa":
        return allow
    return torch.zeros(allow.shape, dtype=dtype, device=allow.device).masked_fill(~allow, torch.finfo(dtype).min)


def _padded(seqs, length, pad_id, device, left):
    """(ids, positions, real) [len(seqs), length] for (ids, positions) pairs padded to `length` on the left or the right."""
    ids = torch.full((len(seqs), length), pad_id, device=device)
    pos, real = torch.zeros_like(ids), torch.zeros_like(ids, dtype=torch.bool)
    for b, (row_ids, row_pos) in enumerate(seqs):
        span = slice(length - len(row_ids), None) if left else slice(None, len(row_ids))
        ids[b, span] = torch.tensor(row_ids, device=device)
        pos[b, span] = torch.tensor(row_pos, device=device)
        real[b, span] = True
    return ids, pos, real


def branch_hidden(lm, splits, pad_id, device):
    """splits[b] = d1a.core.encoding.rows_of(record b's encoding). -> per record, per question, the final hidden states of
    its branch [branch length, d] in fp32: what the pointer head reads (the state's own are never needed)."""
    base = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
    owner = [b for b, (_, _, rows) in enumerate(splits) for _ in rows]
    branches = [r for _, _, rows in splits for r in rows]
    states = _padded([(ids, pos) for ids, pos, _ in splits], max(len(s) for s, _, _ in splits), pad_id, device, left=True)
    rows = _padded([(r["ids"], r["pos"]) for r in branches], max(len(r["ids"]) for r in branches), pad_id, device, left=False)
    run = _method(base, "d1a_shared_prefix", _forward)
    hidden = run(*states, *rows, torch.tensor(owner, device=device))
    out = [[] for _ in splits]
    for i, b in enumerate(owner):
        out[b].append(hidden[i, :len(branches[i]["ids"])])
    return out


def _forward(base, s_ids, s_pos, s_real, b_ids, b_pos, b_real, owner):
    """The decoder stack over left-padded states and right-padded branches (owner[i] = the record of branch i). Who sees
    what: a state token sees the real state tokens up to itself; a branch token its record's real state and its own branch
    so far; a padded query keeps its diagonal so no softmax row is empty (its output is zeroed or never read)."""
    device, Ls, Lb = s_ids.device, s_ids.shape[1], b_ids.shape[1]
    h_s = base.embed_tokens(s_ids) * s_real[..., None]
    h_b = base.embed_tokens(b_ids) * b_real[..., None]
    dtype, attn = h_s.dtype, base.config._attn_implementation

    def causal(n):
        return torch.ones(n, n, dtype=torch.bool, device=device).tril()

    def diagonal(n):
        return torch.eye(n, dtype=torch.bool, device=device)
    # Unpadded states (one per micro-batch, or all of one length) need no state mask under SDPA: None is causal, the flash
    # kernel's case. A 64k-token state's explicit mask would take 4 GB, kept through the backward, and rule flash out.
    plain = attn == "sdpa" and bool(s_real.all())
    branch_allow = torch.cat([s_real[owner][:, None, :].expand(-1, Lb, -1), (causal(Lb)[None] & b_real[:, None, :]) | diagonal(Lb)[None]], -1)
    state_mask = None if plain else _masks((causal(Ls)[None] & s_real[:, None, :]) | diagonal(Ls)[None], dtype, attn)   # never built when unused
    state_masks = {"full_attention": state_mask, "linear_attention": s_real.to(dtype)}
    branch_masks = {"full_attention": _masks(branch_allow, dtype, attn), "linear_attention": b_real.to(dtype)}
    ropes = (base.rotary_emb(h_s, s_pos[None].expand(3, -1, -1)), base.rotary_emb(h_b, b_pos[None].expand(3, -1, -1)))

    recompute = base.gradient_checkpointing and base.training and torch.is_grad_enabled()
    for layer in base.layers:
        step = _method(layer, "d1a_shared_prefix_step", _step)
        masks = (state_masks[layer.block_type], branch_masks[layer.block_type])
        args = (h_s, h_b, ropes, masks, (s_pos, b_pos), s_real, owner)
        h_s, h_b = checkpoint(step, *args, use_reentrant=False) if recompute else step(*args)
    return base.norm(h_b).float()


def _step(layer, h_s, h_b, ropes, masks, positions, s_real, owner):
    """One decoder layer: the state pass, then the branch pass reading what it left behind (Prefix). One call on the layer
    (both passes call its plain forward), so checkpointing replays the two together."""
    prefix = Prefix()
    h_s = layer.forward(h_s, position_embeddings=ropes[0], attention_mask=masks[0], position_ids=positions[0], past_key_values=prefix)
    prefix.branch(owner)
    h_b = layer.forward(h_b, position_embeddings=ropes[1], attention_mask=masks[1], position_ids=positions[1], past_key_values=prefix)
    return h_s * s_real[..., None], h_b


def _method(module, name, fn):
    """`fn` bound to `module` as its method `name`, bound once and then reused."""
    if not hasattr(module, name):
        setattr(module, name, types.MethodType(fn, module))
    return getattr(module, name)
