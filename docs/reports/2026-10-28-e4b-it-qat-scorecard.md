# D1A-E4B-it QAT: scorecard against D1A-E4B-it v0.6 (2026-10-28)

[JohnP1/d1a-e4b-it-qat-mlx-q4](https://huggingface.co/JohnP1/d1a-e4b-it-qat-mlx-q4) is D1A-E4B-it v0.6's adapter, retrained
for one epoch of v0.6's all-skills mix on Google's QAT int4 trunk (`google/gemma-4-E4B-it-qat-w4a16-ct`), kept exact as MLX 4-bit
(#341). It does decisions, chat, photos, voice and video. Unlike v0.6, it was trained with Gemma 4's per-layer embeddings read
correctly (#353). It was released on 2026-10-27 by the owner's decision, before these gates finished; both gates were
pre-registered before training.

**Method:**
- `scripts/quality_gate.py --all-suites`, then `scripts/decide.py`, on an Apple M1 Max;
- base and head served through `d1a.serving.serve` (MLX, adapter unmerged, media encoders loaded) and interleaved;
- Δ is in points, with paired 95% bootstrap intervals; **bold** where the interval excludes 0;
- ECE is the expected calibration error at each build's temperature (QAT 1.66, fitted on its own answers; v0.6 1.55).

## Against D1A-E4B-it v0.6 (bf16 trunk, MLX 8-bit)

Pooled over every line: **-0.16 [-0.88, +0.51]**, level. No veto: the five card suites, latency (ratio 1.011 against a
base/base floor of 1.000) and every large suite pass. Significant wins: PR labels on the test set +1.6, PR severity with the
per-class offsets +6.0 [+3.8, +8.3], and both routing sets (+1.7, +1.5). Significant loss: **external typesafe-v1, −7.8**
(n = 102). Calibration is looser on transfer-v4 (ECE 0.081 → 0.105, the one rise whose interval excludes 0).

On PR severity the retrain catches fewer P1s with the offsets on (29 of 47 against 39 of 47) at a higher precision (0.30
against 0.22). Blast radius on the test set: +7.2 [+1.4, +13.7].

| Suite | n | v0.6 (MLX 8-bit) | QAT | Δ pts [95% CI] | ECE |
|---|---|---|---|---|---|
| devtools-v1 (card) | 1,074 | 74.4 | 74.6 | +0.2 [-1.1, +1.5] | 0.020 → 0.028 |
| documents-v1 (card) | 920 | 88.6 | 88.9 | +0.3 [-1.1, +1.7] | 0.037 → 0.049 |
| external semif-v1 | 144 | 91.0 | 88.9 | -2.1 [-5.6, +0.7] | 0.029 → 0.058 |
| external typesafe-v1 | 102 | 87.3 | 79.4 | **-7.8 [-11.7, -3.6]** | 0.104 → 0.081 |
| external wanli-v1 | 256 | 73.8 | 72.3 | -1.6 [-3.9, +0.8] | 0.143 → 0.161 |
| external wanli-v2 | 1,002 | 72.6 | 72.1 | -0.5 [-1.9, +0.9] | 0.139 → 0.152 |
| hard-v1 (card) | 1,083 | 75.6 | 76.8 | +1.2 [-0.4, +2.9] | 0.043 → 0.023 |
| ja-jglue development | 1,500 | 80.7 | 80.7 | -0.1 [-1.3, +1.1] | 0.027 → 0.036 |
| ja-jglue test | 1,500 | 78.5 | 79.7 | +1.2 [+0.0, +2.4] | 0.034 → 0.040 |
| night2 assertion | 1,020 | 93.2 | 94.1 | +0.9 [-0.3, +2.0] | 0.043 → 0.033 |
| night2 dates | 900 | 91.2 | 90.0 | -1.2 [-2.7, +0.2] | 0.029 → 0.018 |
| night2 unknowable | 525 | 66.7 | 67.2 | +0.6 [-0.8, +1.9] | 0.089 → 0.081 |
| pr-labels development | 2,030 | 79.7 | 80.5 | +0.9 [-0.1, +1.9] | 0.023 → 0.026 |
| pr-labels development-ja | 198 | 75.3 | 78.8 | +3.5 [-0.5, +7.2] | 0.054 → 0.066 |
| pr-labels real17 | 51 | 68.6 | 64.7 | -3.9 [-11.8, +3.9] | 0.169 → 0.126 |
| pr-labels test | 2,045 | 79.4 | 81.0 | **+1.6 [+0.5, +2.7]** | 0.026 → 0.032 |
| pr-labels test-ja | 193 | 76.7 | 76.2 | -0.5 [-2.7, +1.6] | 0.070 → 0.053 |
| routing factory-development | 640 | 93.4 | 95.2 | **+1.7 [+0.3, +3.0]** | 0.059 → 0.043 |
| routing generic-development | 270 | 97.4 | 98.9 | **+1.5 [+0.4, +3.0]** | 0.023 → 0.009 |
| routing handlabelled-45 | 45 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] | 0.007 → 0.006 |
| transfer-v4 (card) | 656 | 77.3 | 76.1 | -1.2 [-3.4, +0.8] | 0.081 → 0.105 |
| decision-v7 (card) | 1,264 | 85.8 | 86.0 | +0.2 [-0.8, +1.3] | 0.038 → 0.052 |

## Against the v0.6 adapter on the same QAT trunk (what the retrain did)

Pooled: **+1.19 [+0.50, +1.87]**, with 6 significant wins (hard-v1 +3.4, PR labels test +4.4 and development +1.7, PR severity
with offsets +8.6, JGLUE test +1.7, wanli-v2 +1.6) and no significant loss. The base is v0.6's adapter moved onto the exact QAT
trunk unchanged, at v0.6's temperature. typesafe-v1 is level here (−1.0 [−5.8, +4.3]), so its loss against v0.6 comes
from the 4-bit trunk, not the retrain. One target missed: the routing factory set's coverage at 90% confidence (0.78 → 0.77).

| Suite | n | v0.6 adapter on QAT | QAT (retrained) | Δ pts [95% CI] | ECE |
|---|---|---|---|---|---|
| devtools-v1 (card) | 1,074 | 74.3 | 74.6 | +0.3 [-1.0, +1.6] | 0.043 → 0.028 |
| documents-v1 (card) | 920 | 87.8 | 88.9 | +1.1 [-0.2, +2.4] | 0.071 → 0.049 |
| external semif-v1 | 144 | 90.3 | 88.9 | -1.4 [-4.9, +2.1] | 0.042 → 0.058 |
| external typesafe-v1 | 102 | 80.4 | 79.4 | -1.0 [-5.8, +4.3] | 0.106 → 0.081 |
| external wanli-v1 | 256 | 71.9 | 72.3 | +0.4 [-2.3, +3.1] | 0.166 → 0.161 |
| external wanli-v2 | 1,002 | 70.5 | 72.1 | **+1.6 [+0.1, +3.1]** | 0.176 → 0.152 |
| hard-v1 (card) | 1,083 | 73.4 | 76.8 | **+3.4 [+1.7, +5.4]** | 0.040 → 0.023 |
| ja-jglue development | 1,500 | 79.7 | 80.7 | +1.0 [-0.2, +2.3] | 0.028 → 0.036 |
| ja-jglue test | 1,500 | 78.0 | 79.7 | **+1.7 [+0.4, +3.0]** | 0.045 → 0.040 |
| night2 assertion | 1,020 | 92.5 | 94.1 | **+1.7 [+0.7, +2.7]** | 0.019 → 0.033 |
| night2 dates | 900 | 89.1 | 90.0 | +0.9 [-0.2, +2.0] | 0.020 → 0.018 |
| night2 unknowable | 525 | 66.7 | 67.2 | +0.6 [-1.0, +2.3] | 0.115 → 0.081 |
| pr-labels development | 2,030 | 78.9 | 80.5 | **+1.7 [+0.4, +3.0]** | 0.065 → 0.026 |
| pr-labels development-ja | 198 | 79.3 | 78.8 | -0.5 [-5.0, +4.1] | 0.070 → 0.066 |
| pr-labels real17 | 51 | 64.7 | 64.7 | +0.0 [-7.8, +7.8] | 0.194 → 0.126 |
| pr-labels test | 2,045 | 76.5 | 81.0 | **+4.4 [+3.0, +5.9]** | 0.072 → 0.032 |
| pr-labels test-ja | 193 | 73.1 | 76.2 | +3.1 [-1.0, +7.4] | 0.104 → 0.053 |
| routing factory-development | 640 | 94.5 | 95.2 | +0.6 [-0.9, +2.2] | 0.026 → 0.043 |
| routing generic-development | 270 | 97.8 | 98.9 | +1.1 [+0.0, +2.6] | 0.017 → 0.009 |
| routing handlabelled-45 | 45 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] | 0.005 → 0.006 |
| transfer-v4 (card) | 656 | 76.1 | 76.1 | +0.0 [-1.8, +1.7] | 0.119 → 0.105 |
| decision-v7 (card) | 1,264 | 85.3 | 86.0 | +0.7 [-0.2, +1.6] | 0.071 → 0.052 |

## The live PR labeler and the owner's labels

| Set | n | v0.6 | QAT | Δ pts [95% CI] |
|---|---|---|---|---|
| labeler replay (recent PRs) | 193 | 71.0 | 72.0 | +1.0 [-4.2, +6.2] |
| owner's labels | 109 | 70.6 | 71.6 | +0.9 [-4.4, +6.5] |

## Photos, voice and video

The playground's 14 media requests (6 photos, 4 voice clips, 4 videos): all answered; 26 of 28 answers are the same as
D1A-E4B-it v0.6's. The 2 differences: an intact parcel in a mailbox read as "in a delivery locker", and a voice clip about a
damaged parcel marked urgent.

## Follow-ups

- typesafe-v1's loss against v0.6 (the 4-bit trunk, not the retrain).
- transfer-v4's calibration (ECE +0.023).
