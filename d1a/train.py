"""Fine-tune the decision model: LoRA on the backbone and a pointer head trained from scratch, on labelled requests (a
frozen suite's training partition, records built from the public sources, or your own JSONL).

    uv run python -m d1a.train --suite evals/v7/decision-v7 --out runs/<name>          # Gemma 4 E2B (the default base)
    uv run python -m d1a.train --base Qwen/Qwen2.5-0.5B --n_per_source 40 --accum 4 --out runs/smoke   # ~1 min smoke test
    uv run python -m d1a.train --data mine.jsonl --init_from JohnP1/kev-gemma4-e2b --lr 2e-5 --out runs/mine   # delta

Records vary in length and carry their own masks, so a forward pass holds few of them (--batch) and gradients accumulate
over --accum micro-batches: an optimizer step sees accum x batch records. Training is reproducible from --seed: the
shuffle, each record's augmentation and none pairs, and the LoRA and head initialisation all draw from it, and a run cut
off and resumed (--resume) continues exactly where it stopped.
"""
import argparse, contextlib, json, math, random, resource, shutil, sys, time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from . import resume, suites
from .checkpoint import Checkpoint, Meta, write_meta
from .device import allocated_bytes, default_device, empty_cache, sync
from .data import EVAL_ONLY, build, augment, load_records, materialize, none_pair, source_seed
from .suite import ADMISSION_BRANCH_HEADROOM, SYNTHETIC_SOURCES, digest, load_split, read_manifest, validate_training, write_json
from .model import MAX_STATE, MAX_TRAIN_STATE, SPECIAL, DecisionModel, delimiter_ids, fits, layout, load_tokenizer, rows_of, training_context, user_tokens


# --- the loss -----------------------------------------------------------------------------------------------------------

def question_loss(z, q, dev):
    """One question's loss from its logits: cross-entropy on its label, or against its soft target when it has one."""
    if q.get("target") is not None:
        target = torch.tensor(q["target"], device=dev, dtype=z.dtype)
        return -(target * F.log_softmax(z, -1)).sum()
    return F.cross_entropy(z[None], torch.tensor([q["label"]], device=dev))


def accumulation_records(n, batch, accum, microbatch):
    """How many of n records the optimizer step holding micro-batch `microbatch` sees (fewer in a short last step)."""
    first = (microbatch // accum) * accum * batch
    return min(accum * batch, n - first)


# --- what a run trains on -----------------------------------------------------------------------------------------------

def base_requests(a, manifest, holdout):
    """--data (with --replay records from the suite), the suite's training partition, or records built from the public
    sources; each suite's rules (declared trainable sources, no held-out structures) apply to the records taken from it."""
    if a.data:
        requests = load_records(suites.resolve(a.data, purpose="train"))   # a D1A suite partition: pinned, verified, train role only
        if a.replay:
            pool = load_split(a.suite, "train")
            replay = random.Random(f"replay:{a.seed}").sample(pool, min(a.replay, len(pool)))
            validate_training(replay, manifest)
            print(f"replay: {len(replay)} of {len(pool)} suite training records mixed with {len(requests)} from {a.data}", flush=True)
            requests = requests + replay
        return requests
    if manifest:
        requests = load_split(a.suite, "train")
        validate_training(requests, manifest)
        return requests
    return build(a.n_per_source, "train", a.seed, exclude=holdout)


def training_requests(a, tok, manifest, holdout):
    """The labelled requests one run trains on: base_requests plus --extra_suites, kept to the training context where
    they were not admitted to it, checked against the eval-only sources, then the mix options (--train_sources,
    --public_frac, --synthetic_repeat)."""
    requests = base_requests(a, manifest, holdout)
    extra_eval_only = set()
    for suite in filter(None, a.extra_suites.split(",")):   # each extra suite's own rules apply to its records
        extra_manifest, extra = read_manifest(suite), load_split(suite, "train")
        validate_training(extra, extra_manifest)
        extra_eval_only |= set(extra_manifest.get("eval_only_sources", []))
        print(f"extra suite {Path(suite).name}: {len(extra)} training records", flush=True)
        requests = requests + extra
    # Frozen suites were admitted under a Qwen tokenizer; a base from another family (Gemma 4: its own delimiters, <bos>,
    # vocabulary) never was. Records not admitted to the training context are filtered to it here rather than aborting the
    # run in the strict encoder (issue #5); a foreign tokenizer also keeps the suites' branch headroom, because
    # augmentation (a none option or a distractor, <= 18 Gemma tokens) grows a branch after this check (Gemma 4 on
    # decision-v7: 70 of 12,576 dropped; without the headroom a run died at step 2490 of 3144).
    foreign = bool(manifest) and not a.data and delimiter_ids(tok) != [tok.convert_tokens_to_ids(t) for t in SPECIAL]
    if not manifest or a.data or foreign or a.extra_suites:
        context = training_context(a.max_state)
        limits = {**context, "max_branch": context["max_branch"] - ADMISSION_BRANCH_HEADROOM} if foreign else context
        kept = [r for r in requests if fits(materialize(r), tok, **limits)]
        if len(kept) < len(requests):
            print(f"dropped {len(requests) - len(kept)} of {len(requests)} records that exceed the training context "
                  f"({context['max_state']} state / {limits['max_branch']} branch / {context['max_packed']} packed tokens)", flush=True)
        requests = kept
    if not requests:
        raise ValueError("empty training set")
    eval_only = set(EVAL_ONLY) | set(manifest.get("eval_only_sources", []) if manifest else []) | extra_eval_only
    forbidden = {r["_meta"]["source"] for r in requests} & eval_only
    if forbidden:
        raise ValueError(f"training data contains eval-only sources: {sorted(forbidden)}")
    return mixed(a, requests)


def mixed(a, requests):
    """The mix options, in order: --train_sources (a subset of sources), --public_frac (a seeded share of the public
    records, the synthetic ones all kept), --synthetic_repeat (the synthetic records that many times)."""
    if a.train_sources:
        wanted = set(a.train_sources.split(","))
        unknown = wanted - {r["_meta"]["source"] for r in requests}
        if unknown:
            raise ValueError(f"--train_sources not in the training partition: {sorted(unknown)}")
        requests = [r for r in requests if r["_meta"]["source"] in wanted]
        print(f"ablation: training on {sorted(wanted)} -> {len(requests)} records", flush=True)
    synthetic = lambda r: r["_meta"]["source"] in SYNTHETIC_SOURCES
    if a.public_frac < 1:
        draw = random.Random(source_seed(a.seed, "public_frac"))
        public, synth = [r for r in requests if not synthetic(r)], [r for r in requests if synthetic(r)]
        keep = sorted(draw.sample(range(len(public)), int(round(a.public_frac * len(public)))))
        requests = [public[i] for i in keep] + synth
        print(f"mix: public_frac {a.public_frac} -> {len(keep)} public + {len(synth)} synthetic records", flush=True)
    if a.synthetic_repeat > 1:
        extra = [r for r in requests if synthetic(r)] * (a.synthetic_repeat - 1)
        requests = requests + extra
        print(f"mix: synthetic_repeat {a.synthetic_repeat} -> +{len(extra)} records", flush=True)
    return requests


# --- encoded examples and their passes ----------------------------------------------------------------------------------

@dataclass(eq=False)   # compared by identity, so batch.index(v) finds this very variant
class Variant:
    """One encoded training example: an augmented copy of a request, and the request's id."""
    rec: dict
    enc: dict
    request_id: str
    share: float = 1.0              # the part of its variant this is, when --row_budget split its questions (question_parts)

    @property
    def tokens(self):
        return len(self.enc["ids"])


def shape(enc):
    """(state tokens, [branch tokens of each question]) of an encoding (d1a.model.rows_of)."""
    state, _, rows = rows_of(enc)
    return len(state), [len(r["ids"]) for r in rows]


def pass_tokens(shapes, shared):
    """The padded tokens of one forward pass over records of these shapes. Rows: rows x the longest row (the state again
    in every question's row). A shared prefix (d1a.shared_prefix): states x the longest state + branches x the longest
    branch."""
    if shared:
        return len(shapes) * max(s for s, _ in shapes) + sum(len(b) for _, b in shapes) * max(x for _, b in shapes for x in b)
    rows = [s + x for s, branches in shapes for x in branches]
    return len(rows) * max(rows)


def question_parts(enc, budget, shared):
    """--row_budget: a record's questions in consecutive groups whose pass fits `budget` tokens (one question always fits:
    its row is at most d1a.model.MAX_TRAIN_STATE plus a branch). One group without a budget."""
    state, branches = shape(enc)
    if not budget:
        return [list(range(len(branches)))]
    parts = [[]]
    for q in range(len(branches)):
        if parts[-1] and pass_tokens([(state, [branches[i] for i in parts[-1] + [q]])], shared) > budget:
            parts.append([])
        parts[-1].append(q)
    return parts


NONE_OPTION_CHARS = 64   # a none-pair sibling's extra option, in the characters microbatch_plan measures (d1a.data.NONE_OPTIONS' longest is shorter)


def character_shapes(reqs, pairs):
    """{id(record): [(state characters, [characters of each question])]}: the cost microbatch_plan cuts on before anything
    is encoded. A record pairing this epoch (pairs) also counts its two siblings: its state again, each with its longest
    Choice question and one more option."""
    chars = lambda value: len(json.dumps(value, ensure_ascii=False))
    shapes = {id(r): [(chars(r["state"]), [chars(q) for q in r["questions"].values()])] for r in reqs}
    for r in reqs if pairs else ():
        if id(r) in pairs:
            longest_choice = max(chars(q) for q in r["questions"].values() if q["type"] == "choice")
            shapes[id(r)] += [(shapes[id(r)][0][0], [longest_choice + NONE_OPTION_CHARS])] * 2
    return shapes


def microbatch_plan(reqs, a, pairs=None, shapes=None):
    """One epoch's micro-batches of `reqs` (already shuffled), in order: (records, records in its optimizer step, whether
    the step ends after it). Each record is seen once per epoch.

    Plain: --batch consecutive records per micro-batch, --accum micro-batches per step.
    --length_sort 1: each step's batch x accum records are cut, in cost order, into accum runs of neighbours whose costliest
    padded pass (pass_tokens) is as cheap as possible (one long record, or many short ones), so micro-batches pad little;
    a step still sees the same records (the same gradient up to summation order). The cost is characters, known before
    encoding (character_shapes), with each pairing record's none-pair siblings counted, so a micro-batch's real padded
    tokens stay within what its run was cut to (without them, --p_none_pair's siblings on long states tripled a pass past
    the GPU: round 21's projection).
    shapes (--pass_tokens_max, from plan_shapes): the runs are cut on those exact token shapes instead, and a step whose
    costliest run is over --pass_tokens_max gets one more micro-batch until none is (round 21 ran out of memory on a
    characters plan: a token-dense state, ~2 characters per token, padded 16 states to 5,877 tokens, 98.7k padded tokens
    where the other passes, costed alike in characters, held 33-50k)."""
    if not a.length_sort:
        count = math.ceil(len(reqs) / a.batch)
        return [(reqs[mb * a.batch:(mb + 1) * a.batch], accumulation_records(len(reqs), a.batch, a.accum, mb), (mb + 1) % a.accum == 0 or mb + 1 == count)
                for mb in range(count)]
    ceiling = a.pass_tokens_max if shapes is not None else 0
    if shapes is None:
        shapes = character_shapes(reqs, pairs)
    cost = lambda rs: pass_tokens([s for r in rs for s in shapes[id(r)]], a.shared_prefix)
    if ceiling and (over := [r for r in reqs if cost([r]) > ceiling]):
        worst = max(over, key=lambda r: cost([r]))
        raise ValueError(f"{len(over)} record(s) need more than --pass_tokens_max {ceiling} padded tokens on their own (the largest, "
                         f"{worst['_meta']['id']}, {cost([worst])}): raise the ceiling or lower --max_state")
    plan, per_step = [], a.batch * a.accum
    for start in range(0, len(reqs), per_step):
        step = reqs[start:start + per_step]
        by_cost = sorted(step, key=lambda r: cost([r]), reverse=True)
        count = math.ceil(len(step) / a.batch)   # accum micro-batches, fewer in a short last step
        runs = balanced_runs(by_cost, count, cost)
        while ceiling and max(map(cost, runs)) > ceiling:   # ends: count runs of one record each are all within it
            count += 1
            runs = balanced_runs(by_cost, count, cost)
        runs = sorted(runs, key=cost, reverse=True)
        plan += [(runs[k], len(step), k == count - 1) for k in range(count)]
    return plan


def balanced_runs(items, n, cost):
    """`items` cut into exactly n consecutive non-empty runs with the costliest as cheap as a greedy cut allows (a binary
    search on the cap); when the cap leaves fewer runs, the costliest splittable run is halved until there are n."""
    def cut(cap):
        runs = [[]]
        for item in items:
            if runs[-1] and cost(runs[-1] + [item]) > cap:
                runs.append([])
            runs[-1].append(item)
        return runs
    low, high = max(cost([item]) for item in items), cost(items)
    while low < high:
        middle = (low + high) // 2
        if len(cut(middle)) <= n:
            high = middle
        else:
            low = middle + 1
    runs = cut(low)
    while len(runs) < n:
        run = max((r for r in runs if len(r) > 1), key=cost)
        at = runs.index(run)
        runs[at:at + 1] = [run[:len(run) // 2], run[len(run) // 2:]]
    return runs


def row_passes(batch, budget, shared):
    """--row_budget: a micro-batch's variants in forward/backward passes of at most `budget` padded tokens (pass_tokens),
    longest first; [batch] without a budget. The loss is a sum over variants (a split record's parts carry their share of
    its mean), so the gradient is the same, but only one pass's activations are alive at once: on one H200, a 27B's bf16
    weights and gradients leave ~35 GB, and the first probe, with passes of up to 16k tokens, ran out of memory."""
    if not budget:
        return [batch]
    passes = []   # [variants, their shapes]
    for v in sorted(batch, key=lambda v: pass_tokens([shape(v.enc)], shared), reverse=True):
        if passes and pass_tokens(passes[-1][1] + [shape(v.enc)], shared) <= budget:
            passes[-1][0].append(v)
            passes[-1][1].append(shape(v.enc))
        else:
            passes.append([[v], [shape(v.enc)]])
    return [variants for variants, _ in passes]


def state_token_counts(tok, reqs):
    """{id(record): state tokens} as d1a.model.encode counts them: the <state> token (after the tokenizer's leading ids:
    Gemma's <bos>, none for Qwen) plus user_tokens of the materialised state; the length --none_pair_max_state gates on."""
    head = len(layout(tok)[0]) + 1
    return {id(r): head + len(user_tokens(tok, materialize(r)["state"])) for r in reqs}


def none_pairs(a, reqs, epoch, state_tokens):
    """--none_pair_max_state: {id(record)} of the records that train their none-pair siblings this epoch. Each record draws
    from its own stream (seed, epoch, record id, "none_pair"), so microbatch_plan knows the siblings before anything is
    encoded: a record pairs with probability --p_none_pair when its state has at most none_pair_max_state tokens and it has
    an eligible Choice (d1a.data.none_pair gives it a pair)."""
    return {id(r) for r in reqs if state_tokens[id(r)] <= a.none_pair_max_state
            and random.Random(source_seed(a.seed, f"{epoch}:{r['_meta']['id']}:none_pair")).random() < a.p_none_pair
            and none_pair(r, random.Random(0))}


def record_variants(req, a, epoch, pairs=None):
    """(the requests one record trains this epoch, its item stream afterwards): the record augmented afresh (a new option
    order, perhaps a none option or a distractor), then its none-pair siblings. pairs None: the record draws its pair from
    its own item stream with probability --p_none_pair (every recipe before --none_pair_max_state); otherwise exactly the
    records in pairs (by id(), none_pairs) get theirs. encode_batch trains them; plan_shapes measures them."""
    item_rng = random.Random(source_seed(a.seed, f"{epoch}:{req['_meta']['id']}"))
    variants = [augment(req, item_rng, p_none=a.p_none, p_none_distract=a.p_none_distract, p_distract=a.p_distract)]
    if pairs is None:
        if a.p_none_pair > 0 and item_rng.random() < a.p_none_pair:
            variants += none_pair(req, item_rng)
    elif id(req) in pairs:
        variants += none_pair(req, item_rng)
    return variants, item_rng


def plan_shapes(model, tok, a, reqs, epoch, pairs, state_tokens):
    """--pass_tokens_max: {id(record): [(state tokens, [branch tokens of each question]) of every variant it trains this
    epoch, siblings included]}, exactly the shapes encode_batch will build. A branch's tokens do not depend on the state
    (d1a.model.encode tokenizes each part on its own), so branches are encoded under an empty state and the state's count
    comes from state_token_counts, without tokenizing the states again. Branches are tokenized again each epoch
    (augmentation draws per epoch): a few minutes on the SFT corpus. A branch that fits a one-token state but not its real
    one still raises ContextOverflow in encode_batch, mid-epoch, as without this flag (frozen suites are admitted with
    branch headroom, so it does not arise on them)."""
    context = training_context(a.max_state)
    branches = lambda v: [len(r["ids"]) for r in rows_of(model.encode(tok, {**materialize(v), "state": ""}, max_state=context["max_state"], max_branch=context["max_branch"]))[2]]
    return {id(r): [(state_tokens[id(r)], branches(v)) for v in record_variants(r, a, epoch, pairs)[0]] for r in reqs}


def encode_batch(model, tok, a, chunk, epoch, pairs=None):
    """Each request's variants this epoch (record_variants), encoded strictly; a record --row_budget splits by question
    becomes one Variant per part."""
    out, context = [], training_context(a.max_state)
    limits = {"max_state": context["max_state"], "max_branch": context["max_branch"]}
    for req in chunk:
        for variant in record_variants(req, a, epoch, pairs)[0]:
            rec = materialize(variant)
            enc = model.encode(tok, rec, strict=True, **limits)
            if len(enc["ids"]) > context["max_packed"]:
                raise ValueError(f"training request exceeds {context['max_packed']} packed tokens")
            parts = question_parts(enc, a.row_budget, a.shared_prefix)
            for part in parts:
                sub = rec if len(parts) == 1 else {**rec, "questions": [rec["questions"][q] for q in part]}
                out.append(Variant(sub, enc if sub is rec else model.encode(tok, sub, strict=True, **limits), req["_meta"]["id"],
                                   share=len(part) / len(rec["questions"])))
    return out


def batch_loss(model, a, batch, dev, autocast):
    """Forward the variants and sum their mean question losses (each times its share). -> (loss, Counter(ce=loss))."""
    with autocast:
        logits_b = model.forward_batch([v.enc for v in batch], a.shared_prefix)
    loss = sum(sum(question_loss(z.float(), q, dev) for z, q in zip(logits, v.rec["questions"])) / len(logits) * v.share
               for v, logits in zip(batch, logits_b))
    if not torch.isfinite(loss):
        raise NonFinite("non-finite training loss")
    return loss, Counter(ce=loss.item())


# --- arguments ----------------------------------------------------------------------------------------------------------

DEFAULT_BASE = ("google/gemma-4-E2B", "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f")   # D1A's base and its pinned commit


def parse_args():
    ap = argparse.ArgumentParser()
    add = ap.add_argument
    add("--base", default=DEFAULT_BASE[0], help="Hugging Face base model (default: Gemma 4 E2B, pinned to DEFAULT_BASE's commit)")
    add("--n_per_source", type=int, default=1000)
    add("--epochs", type=int, default=1)
    add("--lr", type=float, default=2e-4)
    add("--head_lr", type=float, default=0.0, help="separate learning rate for the pointer head (0 = same as --lr); the head trains from scratch")
    add("--weight_decay", type=float, default=0.01, help="AdamW weight decay on LoRA and head parameters")
    add("--lora", type=int, default=16)
    add("--accum", type=int, default=8)
    add("--holdout", default="", help="comma-separated sources excluded from training (evaluated as out-of-source)")
    add("--suite", help="frozen suite directory; train only on its training partition")
    add("--train_sources", default="", help="comma-separated subset of the suite's trainable sources (ablations); default all")
    add("--device", choices=["cpu", "mps", "cuda"], default=None)
    add("--batch", type=int, default=1, help="records per forward pass (padded batch); optimizer step every --accum micro-batches")
    add("--dtype", choices=["fp32", "bf16"], default="fp32", help="bf16 = autocast forward with fp32 master weights (CUDA only)")
    add("--weights_dtype", choices=["fp32", "bf16"], default="fp32", help="dtype of the frozen backbone weights. bf16 halves memory and is required by the fused MoE experts "
                                                                        "(torch._grouped_mm wants bf16); LoRA and head stay fp32 (peft upcasts adapters). The checkpoint records it and is loaded the same way.")
    add("--checkpointing", type=int, choices=[0, 1], default=0)
    add("--head_dim", type=int, default=256, help="pointer head dimension")
    add("--lora_targets", choices=["all", "dense", "attn", "qv"], default="all", help="LoRA module set; fewer modules = less drift from the base; dense = all minus the DeltaNet projections on hybrid bases")
    add("--base_revision", default="", help="pin the base commit when the suite manifest does not pin this base")
    add("--p_none", type=float, default=0.1)
    add("--p_none_distract", type=float, default=0.12)
    add("--p_distract", type=float, default=0.15)
    add("--p_none_pair", type=float, default=0.0, help="fraction of Choice records that additionally emit a none-present/none-absent minimal pair")
    add("--none_pair_max_state", type=int, default=None, help="only records whose state has at most this many tokens emit none pairs (each pair repeats "
                                                              "the state twice); their siblings count in the --length_sort cost (default: every record, as before)")
    add("--synthetic_repeat", type=int, default=1, help="oversample synthetic policy sources (legacy_policy, compositional, contrastive) this many times per epoch")
    add("--public_frac", type=float, default=1.0, help="deterministic subsample of public-source training records (mix ablations)")
    add("--out", default="runs/d1a")
    add("--data", default="", help="your own labelled requests, one JSON object per line (see d1a.data.load_records), or a D1A suite partition (evals/d1a/<suite>:<partition>, d1a.suites); an alternative to --suite for fine-tuning, or combined with --suite and --replay")
    add("--max_state", type=int, default=MAX_STATE, help=f"state tokens per training record (default {MAX_STATE}); raising it admits long-state --data records, the packed limit grows by the same amount")
    add("--extra_suites", default="", help="comma-separated frozen suite directories whose training partitions are added to this run (each checked against its own manifest), e.g. evals/hard-v1,evals/devtools-v1")
    add("--replay", type=int, default=0, help="with --data and --suite: mix in this many records sampled (by --seed) from the suite's training partition, so a delta fine-tune does not forget the released recipe")
    add("--init_from", default="", help="delta mode: warm-start LoRA and the pointer head from an existing run "
                                        "(local directory or hub id) instead of starting from the base model; keeps the "
                                        "released model's in-domain skill while adapting to a new domain")
    add("--row_budget", type=int, default=0, help="padded row tokens per forward/backward pass (0 = the whole micro-batch at once); a micro-batch over it "
                                                  "runs in several passes, a record whose rows do not fit is split by question (long states x many questions)")
    add("--shared_prefix", type=int, choices=[0, 1], default=0, help="hybrid backbones: run each record's state once and its question branches from it "
                                                                   "(d1a.shared_prefix; exact) instead of one row per question")
    add("--length_sort", type=int, choices=[0, 1], default=0, help="deal each optimizer step's records into micro-batches balanced by padded length "
                                                                   "(--batch becomes the average; same records per step; see microbatch_plan)")
    add("--pass_tokens_max", type=int, default=0, help="with --length_sort 1: padded tokens (pass_tokens: states x the longest state + branches x the longest "
                                                       "branch, none-pair siblings included) no forward/backward pass may exceed; the plan cuts on exact "
                                                       "token shapes and gives a step more micro-batches until none does (0 = off)")
    add("--max_steps", type=int, default=0, help="stop after this many optimizer steps (0 = every epoch); the lr schedule spans them")
    add("--save_every_steps", type=int, default=0, help="write a resume point (<out>/resume) every N optimizer steps")
    add("--save_every_minutes", type=float, default=0, help="write a resume point once this many minutes have passed since the last")
    add("--resume", type=int, choices=[0, 1], default=0, help="continue from <out>/resume if it holds a resume point (same arguments), else start")
    add("--stop_after", type=int, default=0, help="exit after this optimizer step without saving the checkpoint (a run split across containers; tests)")
    add("--seed", type=int, default=0)
    a = ap.parse_args()
    if min(a.epochs, a.accum, a.n_per_source, a.lora, a.batch, a.synthetic_repeat) < 1 or not 0 < a.public_frac <= 1:
        ap.error("epochs, accum, n_per_source, lora, batch and synthetic_repeat must be positive; 0 < public_frac <= 1")
    if a.dtype == "bf16" and a.device != "cuda":
        ap.error("--dtype bf16 requires --device cuda")
    if a.lr <= 0 or a.head_lr < 0 or a.weight_decay < 0:
        ap.error("invalid learning rate or weight decay")
    if not MAX_STATE <= a.max_state <= MAX_TRAIN_STATE:
        ap.error(f"--max_state must be in [{MAX_STATE}, {MAX_TRAIN_STATE}]")
    if a.none_pair_max_state is not None and (a.none_pair_max_state < 1 or a.p_none_pair <= 0):
        ap.error("--none_pair_max_state is a positive token count and needs --p_none_pair > 0")
    if a.replay and not (a.data and a.suite):
        ap.error("--replay needs both --data and --suite")
    if a.row_budget < 0 or a.max_steps < 0 or a.pass_tokens_max < 0:
        ap.error("--row_budget, --pass_tokens_max and --max_steps are >= 0")
    if a.pass_tokens_max and (not a.length_sort or a.row_budget):
        ap.error("--pass_tokens_max caps the passes --length_sort 1 plans: not without it, nor with --row_budget (which splits them again)")
    if Path(a.out).exists() and not a.resume:
        ap.error("refusing to overwrite an existing run")
    return a


RESUME_KNOBS = ("resume", "save_every_steps", "save_every_minutes", "stop_after")   # may differ between a run and its continuation
RESUMED = ("step", "seen", "tokens_seen", "peak_mem", "optimizer_seconds", "step_seconds", "elapsed", "epoch", "microbatch", "grad_norms")   # counters a resume point carries


class NonFinite(ValueError):
    """A training loss that is not finite: its micro-batch is skipped (see MAX_NONFINITE)."""


MAX_NONFINITE = 3   # a non-finite loss or gradient skips its micro-batch or step; this many in a row end the run (the weights themselves are broken)
MAX_GRAD_NORM = 1.0   # the global gradient norm is clipped to this


def grad_norm_summary(grad_norms):
    """Per epoch: the mean and max of its steps' global gradient norms before clipping, and how many steps were clipped."""
    return [{"epoch": ep, "steps": len(norms), "mean": sum(norms) / len(norms), "max": max(norms), "clipped_steps": sum(n > MAX_GRAD_NORM for n in norms)}
            for ep, norms in enumerate(grad_norms) if norms]


def pinned_revision(a, manifest):
    """The base commit a run trains against: the suite's pin, else --base_revision (which must agree with a pin)."""
    revision = manifest["base_revisions"].get(a.base) if manifest else None
    if not revision and not a.base_revision and a.base == DEFAULT_BASE[0]:
        revision = DEFAULT_BASE[1]   # frozen suites pin Qwen bases only; the default base carries its own pin
    if a.base_revision:
        if revision and revision != a.base_revision:
            raise ValueError("--base_revision conflicts with the suite's pinned revision")
        revision = a.base_revision
    if manifest and not revision:
        raise ValueError("base not pinned by the suite; pass --base_revision")
    return revision


# --- the run ------------------------------------------------------------------------------------------------------------

@dataclass
class Progress:
    """The counters a run carries, and that a resume point saves (RESUMED) and restores."""
    step: int = 0
    seen: int = 0
    tokens_seen: int = 0
    peak_mem: int = 0
    optimizer_seconds: float = 0
    step_seconds: list = field(default_factory=list)
    elapsed: float = 0
    epoch: int = 0
    microbatch: int = 0
    grad_norms: list = field(default_factory=list)   # per epoch, each step's global gradient norm before clipping


def build_model(a, dev, tok, revision, holdout):
    """-> (the model with LoRA, the Meta this run will save, the warm-start provenance or None)."""
    model = DecisionModel(a.base, tok, dev, lora=a.lora, revision=revision, head_dim=a.head_dim, lora_targets=a.lora_targets,
                          dtype=torch.bfloat16 if a.weights_dtype == "bf16" else torch.float32)
    if a.checkpointing:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    # what this run will save as head.pt, and the architecture a warm start must match
    meta = Meta(base=a.base, base_revision=revision, lora=a.lora, head_dim=a.head_dim, weights_dtype=a.weights_dtype, holdout=holdout)
    init_source = None
    if a.init_from:
        # delta mode (PR #9, Radexito): start from a trained adapter and pointer head instead of the base, so a fine-tune
        # on new data keeps what the released checkpoint knows
        init_source = Checkpoint(a.init_from).warm_start(model, meta)
        print(f"delta: warm start from {init_source['resolved']}: {init_source['tensors']} {meta.weights} tensors and the pointer head loaded", flush=True)
    if a.pass_tokens_max and not model.hybrid:
        # pass_tokens is exact for rows and the shared prefix, which a hybrid backbone always runs; an attention-only one
        # runs the packed mask (rows_form) unless a record is over ROW_PASS_TOKENS, a cost pass_tokens does not measure
        raise SystemExit("d1a.train: --pass_tokens_max needs a hybrid backbone (Gated DeltaNet: every pass runs as rows or a shared "
                         f"prefix, which pass_tokens measures); {a.base} is attention-only and runs the packed mask")
    print(f"device={dev} trainable params={sum(p.numel() for p in model.trainable_parameters())/1e6:.1f}M", flush=True)
    return model, meta, init_source


def optimizer_and_schedule(a, model, reqs):
    """AdamW over LoRA (--lr) and the head (--head_lr or --lr), and a one-cycle schedule over the run's steps. -> (opt,
    sched, steps). The step count depends on len(reqs) only, not the shuffle (--length_sort: one step per batch x accum
    records, however many micro-batches the ceiling gives it)."""
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    groups = [{"params": [p for p in model.trainable_parameters() if id(p) not in head_ids], "lr": a.lr},
              {"params": head_params, "lr": a.head_lr or a.lr}]
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.weight_decay)
    per_epoch = math.ceil(len(reqs) / (a.batch * a.accum)) if a.pass_tokens_max else sum(ends for _, _, ends in microbatch_plan(reqs, a))
    steps = a.epochs * per_epoch
    steps = min(steps, a.max_steps) if a.max_steps else steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr or a.lr], total_steps=max(steps, 1), pct_start=0.1)
    return opt, sched, steps


def epoch_plan(a, model, tok, reqs, ep, state_tokens):
    """-> (this epoch's none pairs or None, its micro-batch plan)."""
    pairs = none_pairs(a, reqs, ep, state_tokens) if a.none_pair_max_state is not None else None
    if pairs is not None:
        print(f"none pairs: {len(pairs)} of {len(reqs)} records (states of at most {a.none_pair_max_state} tokens, p {a.p_none_pair})", flush=True)
    shapes = plan_shapes(model, tok, a, reqs, ep, pairs, state_tokens) if a.pass_tokens_max else None
    plan = microbatch_plan(reqs, a, pairs, shapes)   # every record once per epoch
    if shapes is not None:
        largest = max(pass_tokens([s for r in chunk for s in shapes[id(r)]], a.shared_prefix) for chunk, _, _ in plan)
        print(f"plan: {len(plan)} micro-batches for {sum(ends for _, _, ends in plan)} steps (--accum {a.accum}); "
              f"the plan's largest pass {largest} of --pass_tokens_max {a.pass_tokens_max} padded tokens", flush=True)
    return pairs, plan


def main():
    a = parse_args()
    if a.data:
        suites.resolve(a.data, purpose="train")   # before the model loads: a refused or mismatched suite partition fails at once
    dev = a.device or default_device()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=bool(a.resume))
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)   # the per-epoch shuffle
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if a.dtype == "bf16" else contextlib.nullcontext()
    manifest = read_manifest(a.suite) if a.suite else None
    revision = pinned_revision(a, manifest)
    holdout = manifest["holdout_sources"] if manifest else [s for s in a.holdout.split(",") if s]

    tok = load_tokenizer(a.base, revision=revision)
    model, meta, init_source = build_model(a, dev, tok, revision, holdout)
    reqs = training_requests(a, tok, manifest, holdout)
    # the none-pair gate (none_pairs) and the ceiling's token shapes (plan_shapes)
    state_tokens = state_token_counts(tok, reqs) if a.none_pair_max_state is not None or a.pass_tokens_max else None
    suite_hash = digest(Path(a.suite) / "manifest.json") if manifest else None
    write_json(out_dir / "training_config.json", {"args": vars(a), "suite_sha256": suite_hash, "base_revision": revision, "init_source": init_source,
                                                "holdout": holdout})
    print(f"{len(reqs)} training requests (holdout={holdout}), questions by type "
          f"{dict(Counter(q['qtype'] for r in reqs for q in materialize(r)['questions']))}")

    opt, sched, steps = optimizer_and_schedule(a, model, reqs)
    progress, run = Progress(), Counter()   # run: the loss since the last log line
    resume_seconds, write_seconds = [], []
    resume_dir, resume_args = out_dir / "resume", {k: v for k, v in vars(a).items() if k not in RESUME_KNOBS}
    position = resume.load(resume_dir, model.trainable_parameters(), opt, sched, resume_args) if a.resume else None
    if position:
        progress = Progress(**{k: position[k] for k in RESUMED})
        run = Counter(position["run"])
        print(f"resumed from {resume_dir / position['dir']}: step {progress.step}, epoch {progress.epoch}, micro-batch {progress.microbatch}", flush=True)
    start_epoch, start_mb = progress.epoch, progress.microbatch
    nonfinite, skipped = 0, Counter()   # non-finite losses / gradients in a row, and the micro-batches and steps skipped in all
    model.train()
    t0, last, saved_at, stopped = time.time() - progress.elapsed, time.time(), time.time(), False
    for ep in range(a.epochs):
        rng.shuffle(reqs)
        if ep < start_epoch:
            continue   # the finished epochs' shuffles are replayed, so the interrupted epoch's order comes back
        pairs, plan = epoch_plan(a, model, tok, reqs, ep, state_tokens)
        for mb in range(start_mb if ep == start_epoch else 0, len(plan)):
            chunk, step_records, ends_step = plan[mb]
            batch = encode_batch(model, tok, a, chunk, ep, pairs)
            variants = sum(v.share for v in batch)   # a record --row_budget split counts once
            # weighted by the source records in the step, so none-pair siblings do not inflate a record's share
            group_records = step_records * (variants / len(chunk))
            try:
                for part in row_passes(batch, a.row_budget, a.shared_prefix):
                    loss, terms = batch_loss(model, a, part, dev, autocast)
                    (loss / group_records).backward()
                    run += terms
            except NonFinite:
                # the passes before it keep their gradients in the step (a share of the micro-batch, as a record --row_budget split)
                nonfinite += 1
                skipped["microbatches"] += 1
                print(f"!!! ep{ep} step {progress.step}: non-finite loss, micro-batch {mb} skipped ({nonfinite} in a row)", flush=True)
                if nonfinite >= MAX_NONFINITE:
                    raise
                if not ends_step:
                    continue
            else:
                nonfinite = 0
            run["n"] += variants
            progress.seen += round(variants)
            progress.tokens_seen += sum(v.tokens for v in batch)
            progress.peak_mem = max(progress.peak_mem, allocated_bytes(dev))
            if not ends_step:
                continue
            norm = float(torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), MAX_GRAD_NORM))
            if not math.isfinite(norm):   # a NaN gradient from a finite loss: the step's update is dropped, the schedule moves on
                nonfinite += 1
                skipped["steps"] += 1
                print(f"!!! ep{ep} step {progress.step}: non-finite gradient norm, optimizer step skipped ({nonfinite} in a row)", flush=True)
                if nonfinite >= MAX_NONFINITE:
                    raise NonFinite(f"{nonfinite} non-finite losses or gradients in a row")
            else:
                started = time.time()
                opt.step()
                sync(dev)
                progress.optimizer_seconds += time.time() - started
                progress.grad_norms += [[] for _ in range(ep + 1 - len(progress.grad_norms))]
                progress.grad_norms[ep].append(round(norm, 6))
            sched.step()
            opt.zero_grad()
            progress.step += 1
            progress.step_seconds.append(round(time.time() - last, 3))
            last = time.time()
            if dev == "mps":
                empty_cache(dev)   # MPS only: releasing the cache each step keeps the unified-memory footprint down (on CUDA it only slows the step)
            if progress.step % 10 == 0:
                print(f"ep{ep} step {progress.step}/{steps} loss {run['ce']/run['n']:.3f} {(time.time()-t0)/progress.seen:.3f}s/rec", flush=True)
                run = Counter()
            if progress.step == steps:
                break
            if resume.due(progress.step, a.save_every_steps, a.save_every_minutes, saved_at, max(resume_seconds, default=0)):
                progress.elapsed, progress.epoch, progress.microbatch = time.time() - t0, ep, mb + 1
                values = {k: getattr(progress, k) for k in RESUMED}
                write_seconds.append(resume.save(resume_dir, progress.step, model.trainable_parameters(), opt, sched,
                                                 {**values, "run": dict(run), "args": resume_args}))
                resume_seconds.append(round(time.time() - last, 3))
                saved_at = last = time.time()   # the time training blocked, not part of the next step's
            if progress.step == a.stop_after:
                stopped = True
                break
        if progress.step == steps or stopped:
            break
    if stopped:
        print(f"stopped after step {progress.step}; continue with --resume 1", flush=True)
        return
    save_run(a, model, tok, meta, progress, out_dir, resume_dir, t0, reqs, suite_hash, init_source, skipped, resume_seconds, write_seconds, dev)


def save_run(a, model, tok, meta, progress, out_dir, resume_dir, t0, reqs, suite_hash, init_source, skipped, resume_seconds, write_seconds, dev):
    """The checkpoint (adapter, head.pt, tokenizer) and training_metrics.json; the resume points it supersedes go."""
    wall = time.time() - t0
    model.lm.save_pretrained(a.out)
    backbone_seconds = time.time() - t0 - wall
    shutil.rmtree(resume_dir, ignore_errors=True)
    meta.head, meta.extra = model.head.state_dict(), {"args": vars(a), "suite_sha256": suite_hash, "init_source": init_source}
    write_meta(a.out, meta)
    tok.save_pretrained(a.out)
    write_json(out_dir / "training_metrics.json", {"wall_seconds": wall, "records_seen": round(progress.seen),
               "requested_records": a.epochs * len(reqs), "truncated_records": 0, "rejected_records": 0,
               "optimizer_steps": progress.step, "forward_tokens": round(progress.tokens_seen), "step_seconds": progress.step_seconds,
               "optimizer_seconds": progress.optimizer_seconds, "resume_seconds": resume_seconds, "resume_write_seconds": write_seconds,
               "backbone_save_seconds": round(backbone_seconds, 1),
               "grad_norm": grad_norm_summary(progress.grad_norms), "nonfinite_skipped": dict(skipped),
               "weights": meta.weights, "peak_device_bytes": progress.peak_mem, "device": dev, "dtype": a.dtype, "batch": a.batch,
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)})
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
