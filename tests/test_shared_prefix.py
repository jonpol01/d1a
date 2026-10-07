"""d1a.backends.shared_prefix: training a hybrid Qwen3.5 through each record's state once, its question branches continuing
from it, is the row form's computation (one causal row of state + branch per question): the same logits and gradients in
fp32 on the CPU, on tests/conftest.py's tiny hybrid base."""
import random

import pytest
import torch

import d1a.backends.shared_prefix as shared_prefix
from d1a.backends.torch import DecisionModel
from d1a.core.encoding import load_tokenizer


def records(tok, sizes, seed=0):
    """One record per (state words, questions) in sizes: random words of the tokenizer's vocabulary, 2 to 4 options a
    question."""
    rng, vocabulary = random.Random(seed), sorted(w for w in tok.get_vocab() if w.isalpha())
    words = lambda n: " ".join(rng.choices(vocabulary, k=n))
    return [{"state": words(n), "questions": [{"instr": words(rng.randint(1, 5)), "options": [words(rng.randint(1, 3)) for _ in range(rng.randint(2, 4))],
                                               "label": 0} for _ in range(questions)]} for n, questions in sizes]


def both_forms(hybrid_base, sizes, lora=None, checkpointing=False, attn=None):
    """-> {False: (logits, gradients) of the row form, True: of the shared prefix} for one model (training mode, LoRA dropout
    off so both forms run one function) and one loss that weighs every question differently."""
    tok = load_tokenizer(str(hybrid_base / "base"))
    torch.manual_seed(0)
    model = DecisionModel(str(hybrid_base / "base"), tok, "cpu", lora=lora, attn=attn)
    with torch.no_grad():   # at its random init the DeltaNet's recurrent state is ~1e-3, under the tolerance: a branch that
        for name, p in model.lm.named_parameters():   # lost it would still pass; x10 carries it into the logits
            if "linear_attn" in name and p.dim() > 1:
                p.mul_(10)
    model.train()
    for module in model.modules():
        for name in getattr(module, "lora_dropout", {}):
            module.lora_dropout[name] = torch.nn.Identity()
    if checkpointing:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    encs = [model.encode(tok, r) for r in records(tok, sizes)]
    out = {}
    for shared in (False, True):
        model.zero_grad()
        logits = [z for per_record in model.forward_batch(encs, shared) for z in per_record]
        sum((k + 1) * torch.log_softmax(z, -1)[0] for k, z in enumerate(logits)).backward()
        out[shared] = ([z.detach() for z in logits], {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    return out


def assert_same(forms, questions, lora):
    (rows, row_grads), (prefix, prefix_grads) = forms[False], forms[True]
    assert len(rows) == len(prefix) == questions and all(torch.allclose(a, b, atol=1e-5) for a, b in zip(rows, prefix))
    assert row_grads.keys() == prefix_grads.keys() and any("lora_" in name for name in row_grads) == bool(lora)
    scale = max(g.abs().max() for g in row_grads.values())
    for name, g in row_grads.items():
        assert torch.allclose(g, prefix_grads[name], atol=1e-5 * scale), name


@pytest.mark.parametrize("lora, checkpointing", [(None, False), (None, True), (4, True)])
def test_the_shared_prefix_is_the_row_form(hybrid_base, lora, checkpointing):
    """States of unequal length (padded on the left), one shorter than the DeltaNet convolution window, 1 to 4 questions;
    the base's own weights, then a LoRA's; with gradient checkpointing each layer's state and branch passes recompute
    together."""
    sizes = [(5, 3), (17, 4), (1, 2), (40, 1)]
    assert_same(both_forms(hybrid_base, sizes, lora, checkpointing), sum(q for _, q in sizes), lora)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_under_sdpa_states_of_one_length_need_no_state_mask(hybrid_base, checkpointing, monkeypatch):
    """Under SDPA, a batch whose states are all one length (a long record alone in its micro-batch) runs its state pass
    causal with no explicit mask (a 64k-token state's would take 4 GB and rule out the flash kernel), and still matches
    the row form; states of mixed lengths build the padded state mask."""
    built = []
    real = shared_prefix._masks
    monkeypatch.setattr(shared_prefix, "_masks", lambda allow, dtype, attn: built.append(tuple(allow.shape[1:])) or real(allow, dtype, attn))
    for second, state_masks in ((23, []), (9, [(24, 24)])):   # 23 words and <state>: 24 tokens
        built.clear()
        assert_same(both_forms(hybrid_base, [(23, 3), (second, 1)], checkpointing=checkpointing, attn="sdpa"), 4, None)
        assert built and [shape for shape in built if shape[0] == shape[1]] == state_masks   # a branch mask is [branch, state + branch]
