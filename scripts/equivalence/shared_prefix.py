"""d1a.backends.shared_prefix against the module at --ref, bit for bit: branch_hidden's outputs and every parameter's
gradient on a tiny random Qwen3.5 (one Gated DeltaNet layer, one attention layer), over generated batches of records
(states of mixed and equal lengths, 1-4 questions each), eager and SDPA attention, gradient checkpointing on and off;
and Prefix's cache calls on their own.

    uv run python scripts/equivalence/shared_prefix.py                 # against origin/main (before the rewrite)

The module binds its forward onto the backbone once (d1a_shared_prefix, d1a_shared_prefix_step); those bindings are
removed before every call, so each version runs its own code on the same weights.
"""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Checker, arguments, module_at  # noqa: E402

import d1a.backends.shared_prefix as new  # noqa: E402


def backbone(attn):
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    config = Qwen3_5TextConfig(vocab_size=40, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                               head_dim=16, linear_num_value_heads=2, linear_num_key_heads=1, linear_key_head_dim=8, linear_value_head_dim=8,
                               layer_types=["linear_attention", "full_attention"], pad_token_id=1)
    config._attn_implementation = attn
    torch.manual_seed(0)
    return Qwen3_5TextModel(config).train()


def splits(rng, equal):
    n_state = rng.randint(3, 24)
    out = []
    for _ in range(rng.randint(1, 3)):
        ls = n_state if equal else rng.randint(3, 24)
        state = [rng.randint(2, 39) for _ in range(ls)]
        rows = []
        for _ in range(rng.randint(1, 4)):
            lb = rng.randint(2, 9)
            rows.append({"ids": [rng.randint(2, 39) for _ in range(lb)], "pos": list(range(ls, ls + lb)), "decide": lb - 1, "opts": [0]})
        out.append((state, list(range(ls)), rows))
    return out


def unbind(lm):
    for m in (lm, *lm.layers):
        for name in ("d1a_shared_prefix", "d1a_shared_prefix_step"):
            if hasattr(m, name):
                delattr(m, name)


def run(module, lm, batch, weights):
    """-> (the branch hidden states, every parameter's gradient) of one forward + backward."""
    unbind(lm)
    lm.zero_grad(set_to_none=True)
    hidden = module.branch_hidden(lm, batch, 1, "cpu")
    loss = sum((h * w).sum() for hs, ws in zip(hidden, weights) for h, w in zip(hs, ws))
    loss.backward()
    grads = [(n, None if p.grad is None else p.grad.clone()) for n, p in lm.named_parameters()]
    return [[h.detach() for h in hs] for hs in hidden], grads


def main():
    a = arguments(__doc__.split("\n")[0], "origin/main", 40)
    old, rng, check = module_at(a.ref, "d1a/backends/shared_prefix.py"), random.Random(a.seed), Checker()
    for attn in ("eager", "sdpa"):
        lm = backbone(attn)
        for checkpointing in (False, True):
            lm.gradient_checkpointing = checkpointing
            for i in range(a.iterations):
                batch = splits(rng, equal=(i % 3 == 0))   # equal states: no state padding (SDPA's unmasked case)
                gen = torch.Generator().manual_seed(i)
                weights = [[torch.randn(len(r["ids"]), 32, generator=gen) for r in rows] for _, _, rows in batch]
                check.same(f"{attn} checkpointing={checkpointing} batch {i}", lambda: run(old, lm, batch, weights), lambda: run(new, lm, batch, weights))
    # Prefix on its own: the cache calls a layer makes, in a state pass then a branch pass
    for i in range(a.iterations):
        gen = torch.Generator().manual_seed(1000 + i)
        k = rng.randint(1, 5)

        def calls(module):
            p = module.Prefix()
            out = [p.record_past, p.layers is p, p[3] is p, p.has_previous_state()]
            out.append(p.update_conv_state(torch.randn(2, 4, k, generator=torch.Generator().manual_seed(i)), 0, 4))
            out.append(p.update_recurrent_state(torch.randn(2, 3, generator=torch.Generator().manual_seed(i + 1)), 0))
            out.append(p.update(torch.randn(2, 1, k, 8, generator=torch.Generator().manual_seed(i + 2)).squeeze(1), torch.ones(2, k, 8), 0))
            p.branch(torch.tensor([0, 1, 1, 0]))
            out += [p.has_previous_state(), p.conv, p.recurrent_states, p.keys]
            out.append(p.update_conv_state(torch.ones(4, 4, 2), 0, 4))
            out.append(p.update_recurrent_state(torch.zeros(4, 3), 0))
            out.append(p.update(torch.ones(4, 2, 8), torch.zeros(4, 2, 8), 0))
            return out
        check.same(f"Prefix {i}", lambda: calls(old), lambda: calls(new))
        del gen
    print(f"d1a.backends.shared_prefix: identical to {a.ref} on {check.count} comparisons")


if __name__ == "__main__":
    main()
