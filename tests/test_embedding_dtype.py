"""--embedding_dtype bf16: the base loads in bf16 and everything but the embedding tables goes to fp32, so the forward is the
fp32 model's for a base stored in bf16 (every weight on bf16's grid) while the tables keep half the memory. On the committed
tiny Gemma 4 fixture, on the CPU."""
from pathlib import Path

import torch

BASE = str(Path(__file__).resolve().parent / "golden" / "tiny-gemma4" / "base")
REC = {"state": "Order 4411 arrived two weeks late and the box was crushed. Two charges appear on the card.",
       "questions": [{"instr": "Is there a billing problem?", "options": ["yes", "no"], "label": 0},
                     {"instr": "Which team handles it?", "options": ["billing", "shipping", "support"], "label": 0}]}


def test_tables_stay_bf16_and_the_forward_is_the_fp32_models():
    from d1a.backends.torch import DecisionModel, load_tokenizer
    tok = load_tokenizer(BASE)
    torch.manual_seed(0)
    ref = DecisionModel(BASE, tok, "cpu", lora=4, head_dim=16)
    with torch.no_grad():                                   # the base as a bf16 checkpoint stores it
        for n, p in ref.lm.named_parameters():
            if "lora_" not in n:
                p.copy_(p.bfloat16().float())
    torch.manual_seed(0)
    kept = DecisionModel(BASE, tok, "cpu", lora=4, head_dim=16, tables_dtype=torch.bfloat16)
    kept.load_state_dict({k: v for k, v in ref.state_dict().items() if "lora_" in k or k.startswith("head.")}, strict=False)
    tables = {n for n, m in kept.lm.named_modules() if isinstance(m, torch.nn.Embedding)}
    assert len(tables) == 2
    for n, p in kept.lm.named_parameters():
        assert p.dtype == (torch.bfloat16 if n.rsplit(".", 1)[0] in tables else torch.float32), n
    assert kept.dtype == "float32"
    ref.eval(), kept.eval()
    enc = ref.encode(tok, REC)
    with torch.no_grad():
        assert torch.equal(kept.hidden_batch([enc]), ref.hidden_batch([enc]))
        assert all(torch.equal(a, b) for a, b in zip(kept.forward_batch([enc])[0], ref.forward_batch([enc])[0]))
