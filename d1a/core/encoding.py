"""Encoding, the backend-independent core of D1A (#60): how a request is packed into tokens, the context limits, and the
split of a packed encoding into causal rows. Every backend (d1a.backends.torch, d1a.backends.mlx, d1a.serving.media) encodes
through here, so they read the same tokens.

A request is packed as [leading ids (Gemma's <bos>)] <state> state, then per question <q> instructions (<opt> option
</opt>)... <decide> (encode). rows_of splits that into the state and one causal row per question.
"""
import functools
import re

from transformers import AutoTokenizer

# The five delimiters (state, q, opt, /opt, decide) are existing, rarely used special tokens, so no embedding rows are
# added; LoRA adapts their meaning. Qwen's fill-in-the-middle and box tokens; Gemma 4 has none of those (they would be
# <unk>), but its reserved <unusedN> rows are distinct unit-norm embeddings caller text cannot produce.
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
GEMMA_SPECIAL = ["<unused0>", "<unused1>", "<unused2>", "<unused3>", "<unused4>"]
DELIMITERS = (SPECIAL, GEMMA_SPECIAL)
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

# The training context: state tokens, tokens per question branch, and the whole packed record. Frozen suites were admitted
# under it (d1a.eval.suite) and training applies it to records built on the fly, so training and evaluation see the same population.
MAX_STATE, MAX_BRANCH, MAX_PACKED = 384, 1024, 2048
# The serving context (d1a.serving.serve): a state of up to 64k tokens (twice Jev's 32k; the Qwen3.5 / Qwen3.8 bases' window is
# 262k) and a question row (the state plus its branch, the encoder's max_branch) of up to 8k tokens more.
SERVE_MAX_STATE = 65536
SERVE_MAX_BRANCH = SERVE_MAX_STATE + 8192
SERVE_MAX_PACKED = SERVE_MAX_STATE + SERVE_MAX_BRANCH
# The tokens one inference pass holds (rows_per_pass), and the longest request an attention-only backbone runs packed
# (rows_form: its block-causal mask is L x L); longer ones run as rows.
ROW_PASS_TOKENS = 16384
# The longest state a checkpoint may train on (d1a.training.train --max_state): the longest one served, which still leaves every
# question its training branch budget (training_context(MAX_TRAIN_STATE)["max_branch"] <= SERVE_MAX_BRANCH).
MAX_TRAIN_STATE = SERVE_MAX_STATE
# The limits before 64k states: the suites frozen until then were admitted under these and record them (d1a.eval.suite.SERVING_CONTEXT_8K).
SERVE_MAX_STATE_8K, SERVE_MAX_BRANCH_8K, MAX_TRAIN_STATE_8K = 8192, 8192, 7552


def training_context(max_state=MAX_STATE):
    """The encoder limits for training with the state limit raised to `max_state`: the row (state + one branch) and the
    packed limits grow by the same amount, so every question keeps its budget. training_context() is the default context
    (d1a.eval.suite.CONTEXT without `truncate`)."""
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
    """A record does not encode within its context (state, branch or packed limit). d1a.serving.serve answers it with a 422; the
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
    raise ValueError(f"no delimiter set in d1a.core.encoding.DELIMITERS exists in the {type(tok).__name__} vocabulary")


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


# --- rows ----------------------------------------------------------------------------------------------------------------

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
