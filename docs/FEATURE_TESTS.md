# Feature tests

Every feature D1A claims, the tests behind the claim, and where each stands on the build that is actually served. A
feature is not claimed, demoed or deployed until its gate passes:

1. **Pass bar written first**, before the run (a pre-registration with the data, the build and the numbers that pass).
2. **Independent review** of the bar and the method.
3. **Pinned build:** the served code commit and model files, recorded with the result.
4. **Run behind the resource guard** (`d1a_preflight`), never against a live serving endpoint.
5. **Chart and per-item results** kept with the run.
6. **Only then the wording:** README, docs, playground and model cards describe what passed, in words that match the
   code ("routes" means the request is forwarded; "logs" means logging is on by default).

Status: **Pass** · **Building** · **Next** · **Reword** (the claim says more than the code does) · **Needs hardware**.

| Feature | Test cases | Pass bar | Status |
|---|---|---|---|
| Model routing (decide and forward) | Unit tests against a fake large model and a fake review service; quality gate with routing off; 30 general prompts (10 small, 10 medium, 10 large; English and Japanese; some with tools) sampled from the held-out routing set; a tool loop; "continue" after a large request; streaming; large model down (unit-tested); three concurrent large requests (cap 2); a prompt longer than D1A's context; review traps (a fenced memo, an essay, a PR link asking for a fix); 7 pull-request reviews, 3 code reviews, 1 with the reviewer down | ≥ 28/30 sorted small vs not-small, no large prompt kept by D1A, every forward shown in the target's log; reviews returned verbatim within 180 s | Pass (2026-11-26, c1d58fca): routing 29/30 (s10 small→large), 0/10 large and 0/10 medium kept; forwarding proven 32/32 against the 31B log; T1–T3, T5, R1–R4 pass; reviews 10/10 + u1. Known limit: over-context prompt → 400 (T6). T4 unit-tested only; 270-row held-out report pending. |
| Agent decisions (gate, judge, intake, tier, route) | The agent evaluation (259 held-out items) re-run on the served build | Not significantly below each claimed number; dangerous calls allowed 0/33; tasks closed too early 0/33 | Pass (gate 53/54, judge 52/52 and 50/52, route 45/45) |
| Skill suites (hard, devtools, documents, decision-v7, transfer) | The served build's scorecard (`docs/reports/2026-10-28-e4b-it-qat-scorecard.md`) | Level with or above the claimed numbers | Pass; README numbers to be relabelled with the served build's |
| PR labeler | The 487 pull requests newer than all training, on the served build | Paired against v0.6 on the same 487 PRs: 95% CI lower bound above −1 point | Severity holds: 81.7% (398/487) vs v0.6's 79.1%, +2.7 [0.0, +5.3]. Change type not shown: 91.6% (446) vs 90.6%, +1.0 [−1.2, +3.3]; the bound misses −1 by 0.2 (pre-registered label: lower), no drop measured. Finding: P1 caught exactly 5/23 vs 8/23 (P0 5/7 both); the served build predicts P0/P1 half as often (n = 30, direction only). Served build, 2026-11-26. |
| Photo, voice and video | The playground's 6 photos, 4 voice clips and 4 videos with the demos' questions; then a labelled set of about 20 clips | Demo examples right; a "Proven at" claim needs the labelled set | Works on demo samples (22/24); labelled set needed |
| Chat | Real answers in English and Japanese, streaming, the model name returned | Works end to end on the served build | Pass (functional smoke, single run, 2026-11-26, c1d58fca, 4-bit MLX, routes unset): 10/10 pre-registered prompts answered correctly (5 EN, 3 JA, 2 streamed) on /v1/chat/completions; all HTTP 200 with model d1a-chat; both streams end with [DONE]. Caveats: finish_reason verified on the 8 non-stream replies only; E3 hit max_tokens 200 and passed on a substring match. Not a quality measure: no multi-turn, system-prompt, tool-call or incremental-streaming check. |
| Playground demos (14) | The right answer hand-labelled for every example; all 14 scored on the served build | Each demo's examples are right, or the demo is reworded | On the playground's own text examples, the served build matches a blind expert label on 213 of 236 answers (90.3%); 9 of 11 demos ≥ 80%; below: PR labeler 7/12, refund gate 2/4. Known misses are marked. Note (post hoc, after seeing D1A's answers, not blind): 10 labels withdrawn gives 213/226; the PR labeler (7/10) still fails, and the refund gate passes on its human-review lane. |
| Japanese dates (`ja_dates`) | The urgency gain (75.0 → 81.7) re-run on the served build | The gain holds, or the number is dropped | Next |
| Self-learning (outcome calibrator, shadow gate) | Evidence in the serving host's logs of calibrator updates and of the gate running | Evidence found, or the wording changes | Next |
| Wording-only claims | Embeddings come from a separate model; decision logging is off by default; no code turns a retrain into the next version; the gate catches 60% of prompt injections | Text matches the code | Reworded in d1a (#424) and the playground (#140); Hugging Face cards waiting for the owner |
| iPhone / Neural Engine, ONNX on Android | A fixed set run on the device | Measured on the device, or marked untested | Needs hardware |
