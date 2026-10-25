"""Chat on PyTorch: a DecisionModel loaded with its adapter unmerged switches it off for chat and is then its base exactly,
and pieces_torch's greedy text equals transformers' own generate on that base (the tied embeddings and the final soft-cap
stand in for the generation head DecisionModel drops). On the committed tiny Gemma 4, no download."""
from pathlib import Path

import pytest
import torch

FIXTURE = Path(__file__).resolve().parent / "golden" / "tiny-gemma4"


@pytest.fixture(scope="module")
def loaded():
    from d1a.backends.checkpoint import Checkpoint, LoadOptions
    return Checkpoint(str(FIXTURE / "checkpoint")).load("cpu", LoadOptions(merge=False))


def test_the_adapter_switches_off_to_the_base_exactly_and_nests(loaded):
    from transformers import AutoModelForCausalLM
    from d1a.backends.checkpoint import Checkpoint
    tok, model = loaded
    ids = torch.tensor([[2, 10, 11, 12, 13, 14]])
    on = model.lm(input_ids=ids).last_hidden_state
    base = AutoModelForCausalLM.from_pretrained(str(FIXTURE / "base"), attn_implementation="eager").model(input_ids=ids).last_hidden_state
    with model.adapter(False):
        assert torch.equal(model.lm(input_ids=ids).last_hidden_state, base)
        with model.adapter(True):   # a decision in the middle of a chat
            assert torch.equal(model.lm(input_ids=ids).last_hidden_state, on)
        assert torch.equal(model.lm(input_ids=ids).last_hidden_state, base)
    assert torch.equal(model.lm(input_ids=ids).last_hidden_state, on)
    merged = Checkpoint(str(FIXTURE / "checkpoint")).load("cpu")[1]
    with pytest.raises(ValueError, match="merged"), merged.adapter(False):
        pass


def test_greedy_chat_text_is_the_bases_own_generation(loaded):
    from transformers import AutoModelForCausalLM
    from d1a.serving.openai_compat import pieces_torch
    tok, model = loaded
    prompt = "Decide whether the parcel arrived damaged and say why"
    base = AutoModelForCausalLM.from_pretrained(str(FIXTURE / "base"), attn_implementation="eager")
    ids = tok(prompt, return_tensors="pt").input_ids
    ref = base.generate(ids, max_new_tokens=12, min_new_tokens=12, do_sample=False)[0, ids.shape[1]:]
    with model.adapter(False):
        steps = list(pieces_torch(model, tok, prompt, 12, 0.0, 1.0, eos=set()))
    assert "".join(p for p, _, _ in steps) == tok.decode(ref)
    assert steps[-1][1:] == (12, "length")


def test_the_logits_are_gemma4s_own_generation_head_with_its_soft_cap():
    from transformers import AutoConfig, AutoModelForCausalLM
    from d1a.serving.openai_compat import next_logits
    config = AutoConfig.from_pretrained(str(FIXTURE / "base"))
    cap = 0.1   # below the tiny model's raw logits (~0.36), so the cap shapes them; Gemma 4 E2B-it's is 30
    config.get_text_config().final_logit_softcapping = cap
    torch.manual_seed(0)
    causal = AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
    ids = torch.tensor([[2, 10, 11, 12, 13, 14]])
    with torch.no_grad():
        ref = causal(input_ids=ids).logits[0, -1].float()
        ours = next_logits(causal.model, causal.model(input_ids=ids).last_hidden_state[0, -1])
        raw = causal.model(input_ids=ids).last_hidden_state[0, -1] @ causal.model.get_input_embeddings().weight.T
    assert raw.abs().max() > 2 * cap   # the cap is not a no-op here
    assert torch.allclose(ours, ref, atol=1e-6)
