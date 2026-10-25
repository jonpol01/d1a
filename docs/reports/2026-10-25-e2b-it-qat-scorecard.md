# D1A-E2B-it QAT: scorecard against D1A-E2B-it v0.6 (2026-10-25)

[JohnP1/d1a-e2b-it-qat-mlx-q4](https://huggingface.co/JohnP1/d1a-e2b-it-qat-mlx-q4) is D1A-E2B-it v0.6's adapter, retrained
for one epoch of v0.6's all-skills mix on Google's QAT int4 trunk (`google/gemma-4-E2B-it-qat-w4a16-ct`), kept exact as MLX 4-bit
(#341). It does decisions, chat, photos, voice and video.

**Method:**
- `scripts/quality_gate.py --all-suites`, then `scripts/decide.py`, on an Apple M1 Max;
- base and head served as published (MLX, adapter unmerged) and interleaved;
- Δ is in points, with paired 95% bootstrap intervals; **bold** where the interval excludes 0;
- ECE is the expected calibration error at each build's temperature.

## Against D1A-E2B-it v0.6 (bf16 trunk), the build it replaces

Pooled over every line: **+0.40 [-0.21, +1.02]**. Significant wins: PR severity on newer PRs +3.5 [+1.4, +5.7] and PR labels on the test set.
Significant loss: **transfer-v4**, task types outside training. D1A's release rule vetoes it there (V1, V4). It replaces v0.6 as
the recommended E2B-it build by the owner's decision of 2026-10-25, a release exception like D1A-E4B v0.6's. Latency: 11% faster
per decision (ratio 0.89).

| Suite | n | v0.6 (bf16) | QAT | Δ pts [95% CI] | ECE |
|---|---|---|---|---|---|
| devtools-v1 (card) | 1,074 | 70.9 | 70.6 | -0.3 [-1.6, +1.0] | 0.051 → 0.069 |
| documents-v1 (card) | 920 | 86.2 | 86.8 | +0.7 [-0.8, +2.1] | 0.043 → 0.051 |
| external semif-v1 | 144 | 88.2 | 87.5 | -0.7 [-4.2, +2.8] | 0.038 → 0.030 |
| external typesafe-v1 | 102 | 70.6 | 74.5 | +3.9 [-3.5, +11.5] | 0.115 → 0.070 |
| external wanli-v1 | 256 | 68.8 | 69.5 | +0.8 [-3.1, +4.7] | 0.186 → 0.179 |
| external wanli-v2 | 1,002 | 69.0 | 67.7 | -1.3 [-3.0, +0.3] | 0.176 → 0.190 |
| hard-v1 (card) | 1,083 | 68.9 | 71.1 | +2.2 [-0.1, +4.5] | 0.040 → 0.028 |
| ja-jglue development | 1,500 | 78.5 | 78.0 | -0.5 [-1.7, +0.9] | 0.028 → 0.022 |
| ja-jglue test | 1,500 | 76.7 | 77.5 | +0.7 [-0.7, +2.1] | 0.023 → 0.033 |
| night2 assertion | 1,020 | 92.1 | 92.2 | +0.1 [-1.1, +1.3] | 0.035 → 0.028 |
| night2 dates | 900 | 86.9 | 85.8 | -1.1 [-3.0, +0.7] | 0.036 → 0.038 |
| night2 unknowable | 525 | 69.0 | 67.6 | -1.3 [-3.6, +1.0] | 0.096 → 0.160 |
| pr-labels development | 2,030 | 78.4 | 79.1 | +0.6 [-0.3, +1.6] | 0.021 → 0.035 |
| pr-labels development-ja | 198 | 74.2 | 76.8 | +2.5 [-0.5, +5.6] | 0.051 → 0.042 |
| pr-labels real17 | 51 | 64.7 | 66.7 | +2.0 [+0.0, +5.9] | 0.170 → 0.129 |
| pr-labels test | 2,045 | 77.0 | 78.7 | **+1.7 [+0.4, +2.9]** | 0.019 → 0.038 |
| pr-labels test-ja | 193 | 75.1 | 74.6 | -0.5 [-3.7, +3.0] | 0.063 → 0.045 |
| routing factory-development | 640 | 92.0 | 93.6 | +1.6 [-0.2, +3.4] | 0.046 → 0.033 |
| routing generic-development | 270 | 97.0 | 97.4 | +0.4 [-0.7, +1.9] | 0.015 → 0.017 |
| routing handlabelled-45 | 45 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] | 0.024 → 0.017 |
| transfer-v4 (card) | 656 | 74.8 | 69.7 | **-5.2 [-7.9, -2.6]** | 0.057 → 0.124 |
| decision-v7 (card) | 1,264 | 84.9 | 84.4 | -0.5 [-1.7, +0.6] | 0.031 → 0.044 |

## Against the v0.6 adapter on the same QAT trunk (what the retrain did)

Pooled: **+2.13 [+1.36, +2.89]**, with 8 significant wins and no significant loss. transfer-v4 is level (−0.2), so its loss against bf16 comes from
the 4-bit trunk, not the retrain. One weakness the retrain added: held-out rule compositions inside transfer-v4, with 4 fixes
against 13 breaks (sign p 0.049).

| Suite | n | v0.6 adapter on QAT | QAT (retrained) | Δ pts [95% CI] | ECE |
|---|---|---|---|---|---|
| devtools-v1 (card) | 1,074 | 68.3 | 70.6 | **+2.2 [+0.6, +4.0]** | 0.098 → 0.069 |
| documents-v1 (card) | 920 | 85.7 | 86.8 | +1.2 [-0.4, +2.9] | 0.078 → 0.051 |
| external semif-v1 | 144 | 82.6 | 87.5 | **+4.9 [+0.7, +9.0]** | 0.059 → 0.030 |
| external typesafe-v1 | 102 | 74.5 | 74.5 | +0.0 [-5.1, +4.7] | 0.125 → 0.070 |
| external wanli-v1 | 256 | 68.8 | 69.5 | +0.8 [-2.3, +3.9] | 0.207 → 0.179 |
| external wanli-v2 | 1,002 | 66.5 | 67.7 | +1.2 [-0.4, +2.9] | 0.225 → 0.190 |
| hard-v1 (card) | 1,083 | 66.9 | 71.1 | **+4.2 [+2.1, +6.5]** | 0.061 → 0.028 |
| ja-jglue development | 1,500 | 76.8 | 78.0 | +1.2 [-0.1, +2.5] | 0.051 → 0.022 |
| ja-jglue test | 1,500 | 76.7 | 77.5 | +0.8 [-0.5, +2.1] | 0.051 → 0.033 |
| night2 assertion | 1,020 | 91.7 | 92.2 | +0.5 [-0.7, +1.7] | 0.013 → 0.028 |
| night2 dates | 900 | 85.3 | 85.8 | +0.4 [-0.9, +1.8] | 0.018 → 0.038 |
| night2 unknowable | 525 | 68.0 | 67.6 | -0.4 [-1.7, +1.0] | 0.168 → 0.160 |
| pr-labels development | 2,030 | 77.5 | 79.1 | **+1.5 [+0.4, +2.7]** | 0.052 → 0.035 |
| pr-labels development-ja | 198 | 76.8 | 76.8 | +0.0 [-4.1, +4.1] | 0.057 → 0.042 |
| pr-labels real17 | 51 | 62.7 | 66.7 | +3.9 [-2.0, +11.8] | 0.315 → 0.129 |
| pr-labels test | 2,045 | 75.3 | 78.7 | **+3.4 [+2.1, +4.7]** | 0.069 → 0.038 |
| pr-labels test-ja | 193 | 73.1 | 74.6 | +1.6 [-2.6, +6.2] | 0.100 → 0.045 |
| routing factory-development | 640 | 89.8 | 93.6 | **+3.7 [+1.8, +5.8]** | 0.034 → 0.033 |
| routing generic-development | 270 | 94.4 | 97.4 | **+3.0 [+0.7, +5.2]** | 0.033 → 0.017 |
| routing handlabelled-45 | 45 | 97.8 | 100.0 | +2.2 [+0.0, +6.7] | 0.024 → 0.017 |
| transfer-v4 (card) | 656 | 69.8 | 69.7 | -0.2 [-2.6, +2.1] | 0.150 → 0.124 |
| decision-v7 (card) | 1,264 | 84.3 | 84.4 | +0.1 [-0.8, +1.0] | 0.056 → 0.044 |

## Photos, voice and video

The playground's 14 media requests (6 photos, 4 voice clips, 4 videos): all answered; 25 of 28 answers are the same as
D1A-E2B-it v0.6's. All 3 differences are on voice clips: two `urgent` flags, and one intent ("the customer is not home" read as
"report a damaged parcel").
