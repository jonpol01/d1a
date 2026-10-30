<p><a href="../MODELS.md">English</a> · <a href="../ja/MODELS.md">日本語</a> · 한국어</p>

> 이 문서는 영어판 [MODELS.md](../MODELS.md)(커밋 9eb86269 기준)의 번역입니다. 내용이 다를 경우 영어판이 정확한 정보입니다.

<a id="models"></a>

# 모델

어떤 D1A 빌드가 있는지, 어디에 공개되어 있는지, 각 버전이 무엇을 위한 것인지 설명합니다.

<a id="what-each-build-can-and-cannot-do-2026-10-25"></a>

## 각 빌드가 할 수 있는 것과 할 수 없는 것(2026-10-25)

공개된 모든 D1A 빌드를 Hub의 파일과 대조해 확인했습니다. **의사결정**: System One API(`POST /v1/systemone`).
**채팅**: OpenAI 형식의 `POST /v1/chat/completions`이며, 어댑터를 따로 유지해야 합니다(*병합되지 않은* 빌드). 어댑터를
끄면 모델은 인스트럭션 튜닝된 Gemma 4 그 자체입니다. **사진, 음성, 영상**: `POST /v1/systemone/media`이며,
모든 빌드에서 제로샷입니다(학습에 미디어가 없음).

| 빌드 | 형식 | 의사결정 | 채팅 | 사진 | 음성 | 영상 | 제한 |
|---|---|---|---|---|---|---|---|
| [d1a-e4b](https://huggingface.co/JohnP1/d1a-e4b) `v0.6.1` | PyTorch (어댑터) | ✅ | ❌ 베이스 트렁크 | ✅¹ | ✅¹ | ✅¹ | |
| [d1a-e4b-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-mlx-q8) `v0.6.1` | MLX 8-bit, 병합됨 | ✅ | ❌ 베이스 트렁크 | ✅ | ✅ | ✅ | 7.5 GB |
| [d1a-e4b-coreml](https://huggingface.co/JohnP1/d1a-e4b-coreml) `v0.6` | Core ML int8 | ✅ | ❌ | ❌ | ❌ | ❌ | 프리뷰: 텍스트 전용, 입력 ≤ 512 토큰 |
| [d1a-e2b](https://huggingface.co/JohnP1/d1a-e2b) `v0.6.1` | PyTorch (어댑터) | ✅ | ❌ 베이스 트렁크 | ✅¹ | ✅¹ | ✅¹ | |
| [d1a-e2b-mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-mlx-q8) `v0.6.1` | MLX 8-bit, 병합됨 | ✅ | ❌ 베이스 트렁크 | ✅ | ✅ | ✅ | 4.8 GB |
| [d1a-e2b-coreml](https://huggingface.co/JohnP1/d1a-e2b-coreml) `v0.6` | Core ML int8 | ✅ | ❌ | ❌ | ❌ | ❌ | 프리뷰: 텍스트 전용, 입력 ≤ 512 토큰 |
| [d1a-e4b-it](https://huggingface.co/JohnP1/d1a-e4b-it) `v0.6` | PyTorch (어댑터) | ✅ | ✅² | ✅¹ | ✅¹ | ✅¹ | |
| [d1a-e4b-it-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-it-mlx-q8) `v0.6` | MLX 8-bit, 병합됨 | ✅ | ❌ 병합됨³ | ✅ | ✅ | ✅ | 7.5 GB |
| [d1a-e4b-it-mlx-q4](https://huggingface.co/JohnP1/d1a-e4b-it-mlx-q4) `v0.6` | MLX 4-bit, 병합됨 | ✅ | ❌ 병합됨³ | ✅ | ✅ | ✅ | 5.2 GB |
| [d1a-e4b-it-coreml-4bit](https://huggingface.co/JohnP1/d1a-e4b-it-coreml-4bit) `v0.6` | Core ML 4-bit | ✅ | ❌ | ❌ | ❌ | ❌ | 프리뷰: 텍스트 전용, ≤ 512 토큰; 테스트 답변 42개 중 3개가 바뀜 |
| [d1a-e2b-it](https://huggingface.co/JohnP1/d1a-e2b-it) `v0.6` | PyTorch (어댑터) | ✅ | ✅² | ✅¹ | ✅¹ | ✅¹ | |
| [d1a-e2b-it-mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-it-mlx-q8) `v0.6` | MLX 8-bit, 병합됨 | ✅ | ❌ 병합됨³ | ✅ | ✅ | ✅ | 4.8 GB |
| [d1a-e2b-it-mlx-q4](https://huggingface.co/JohnP1/d1a-e2b-it-mlx-q4) `v0.6` | MLX 4-bit, 병합됨 | ✅ | ❌ 병합됨³ | ✅ | ✅ | ✅ | 3.6 GB; 스위트로 평가하지 않음 |
| [d1a-e2b-it-coreml](https://huggingface.co/JohnP1/d1a-e2b-it-coreml) `v0.6` | Core ML int8 | ✅ | ❌ | ❌ | ❌ | ❌ | 프리뷰: 텍스트 전용, ≤ 512 토큰 |
| [d1a-e2b-it-coreml-4bit](https://huggingface.co/JohnP1/d1a-e2b-it-coreml-4bit) `v0.6` | Core ML 4-bit | ✅ | ❌ | ❌ | ❌ | ❌ | 프리뷰: 텍스트 전용, ≤ 512 토큰 |
| [d1a-e2b-it-qat-mlx-q4](https://huggingface.co/JohnP1/d1a-e2b-it-qat-mlx-q4) | MLX 4-bit (Google의 QAT int4, 그대로 유지), **병합 안 됨** | ✅ | ✅ | ✅ | ✅ | ✅ | 권장 E2B-it 빌드, **v0.7**(2026-10-30): 영어와 일본어 대화 행위 판단이 향상되었고 인텐트 분류는 그대로임. 첫 버전(태그 `v0.6`)은 transfer-v4 −5.2점에도 불구하고 소유자의 2026-10-25 결정에 따라 D1A-E2B-it v0.6을 대체함; 3.7 GB |
| [d1a-e2b-it-qat](https://huggingface.co/JohnP1/d1a-e2b-it-qat) | PyTorch (어댑터; `base_ct`: Google의 QAT int4 트렁크, 로드 시 재구성) | ✅ | ✅² | ✅¹ | ✅¹ | ✅¹ | PyTorch용 권장 E2B-it, 아직 v0.6(v0.7은 현재 MLX 전용); Gemma 4 E2B-it와 QAT 체크포인트를 로드함 |
| [d1a-e2b-it-qat-coreml](https://huggingface.co/JohnP1/d1a-e2b-it-qat-coreml) | Core ML (Google의 QAT int4, 그대로 유지; 어댑터는 스위치로 제공) | ✅⁴ | ✅⁴ | ✅⁴ | ✅⁴ | ✅⁴ | Core ML용 권장 E2B-it, 아직 v0.6(v0.7은 현재 MLX 전용); 4.6 GB; Mac(M1 Max, GPU)에서 검증했으며 iPhone에서는 아직 검증하지 않음 |
| [d1a-e2b-it-qat-onnx](https://huggingface.co/JohnP1/d1a-e2b-it-qat-onnx) | ONNX Runtime 1.22 (Android / CPU; Google의 QAT int4, 그대로 유지; 어댑터는 스위치로 제공) | ✅ | ✅ | ❌ | ❌ | ❌ | ONNX Runtime의 D1A-E2B-it QAT v0.7: 의사결정과 채팅을 하나의 그래프와 하나의 트렁크로; 텍스트 전용; Pixel 10 Pro에서 채팅 약 12.9 토큰/초, 의사결정 약 0.4초 |
| [d1a-e4b-it-qat-mlx-q4](https://huggingface.co/JohnP1/d1a-e4b-it-qat-mlx-q4) | MLX 4-bit (Google의 QAT int4, 그대로 유지), **병합 안 됨** | ✅ | ✅ | ✅ | ✅ | ✅ | QAT 트렁크에서 다시 학습시킨 D1A-E4B-it v0.6; v0.6과 전체적으로 동등, typesafe −7.8([스코어카드](../reports/2026-10-28-e4b-it-qat-scorecard.md)); 5.3 GB |
| [d1a-e4b-it-qat](https://huggingface.co/JohnP1/d1a-e4b-it-qat) | PyTorch (어댑터; `base_ct`: Google의 QAT int4 트렁크, 로드 시 재구성) | ✅ | ✅² | ✅¹ | ✅¹ | ✅¹ | PyTorch용 같은 어댑터; Gemma 4 E4B-it와 QAT 체크포인트를 로드합니다 |

¹ 별도의 PyTorch 미디어 서버 `python -m d1a.serving.media --run <repo>`를 통해 사용하며, 이 서버는 전체 Gemma 4
모델(bf16, E2B 기준 약 10 GB)을 로드합니다([사진, 음성, 영상](USAGE.md)).
² `D1A_MERGE=0`으로 서빙합니다. 이 설정은 어댑터를 병합하지 않은 채로 유지하므로 채팅 시 어댑터를 끌 수 있습니다([D1A와 채팅하기](USAGE.md#chat-with-d1a)).
³ 채팅에는 병합되지 않은 빌드가 필요합니다. `scripts/export_mlx.py --run JohnP1/d1a-e2b-it --unmerged --media`로 만들 수 있습니다
([D1A와 채팅하기](USAGE.md#chat-with-d1a)).

⁴ 저장소의 Python 레퍼런스 러너(`reference/d1a_coreml_qat.py`, 서버 없음. `io.json`에 포팅을 위한 모든 모델
입력과 출력이 나열되어 있습니다)를 통해 사용합니다. 모든 모델이 GPU에서 실행되며 아직 Neural Engine에서 실행되는 것은 없습니다. 의사결정은 최대 1,536 토큰, 채팅은 최대
2,048 포지션까지 가능합니다([APPLE.md](APPLE.md#d1a-e2b-it-qat-on-core-ml)).

D1A-E2B-it QAT는 채팅, 사진, 음성, 영상을 모두 지원하는 유일한 Core ML 빌드이며, v0.6 Core ML 빌드는 텍스트 전용입니다.

<a id="capabilities-2026-10-17"></a>

## 기능(2026-10-17)

각 D1A 모델 계열이 현재 할 수 있는 일입니다. "의사결정"은 D1A의 System One API(`POST /v1/systemone`)를 뜻하며, 문서에 대한 예/아니오, 선택, 점수 질문에 보정된 확률로 한 번의 패스에 답합니다. 수치는 각 모델의 카드에서 가져온 것이며, "측정하지 않음"은 카드에 아직 수치가 없다는 뜻입니다.

| | **D1A-E4B-it v0.6** | **D1A-E4B v0.6.1** | **D1A-E2B v0.6.1** | **D1A-E2B-it v0.6** | D1A-12B |
|---|---|---|---|---|---|
| 베이스 모델(트렁크) | Gemma 4 E4B-it (인스트럭션 튜닝) | Gemma 4 E4B (사전 학습된 베이스) | Gemma 4 E2B (사전 학습된 베이스) | Gemma 4 E2B-it (인스트럭션 튜닝) | Gemma 4 12B-it QAT; **아직 학습되지 않음**(jonpol01/d1a#270, 보류) |
| 의사결정(`/v1/systemone`) | 예 | 예 | 예 | 예(실험적) | — |
| PR 라벨링, 모든 학습 데이터보다 최신인 PR 487건: 심각도 / 변경 유형 | 이 세트에서는 측정하지 않음; PR-labels 테스트(PR 953건): 72.9% / 88.4% (E2B-it 70.5% / 87.8%) | **79.1% / 90.6%**; P0/P1 30건 중 13건 찾음 | 77.2% / 87.5%; P0/P1 30건 중 0건 | 72.9% / 87.3%; P0/P1 30건 중 3건("PR 라벨러가 아님") | — |
| 스킬 스위트(어려운 의사결정, 개발자 도구, 긴 문서, 전이, 일본어 JGLUE) | 2026-10-17에 E2B-it 대비 측정(카드): 22개 스위트 중 21개에서 향상, 예: 어려운 의사결정 75.7%, 개발자 도구 74.3%, 긴 문서 88.6%, JGLUE 78.6% | 측정함(카드) | 측정함(카드) | 2026-10-15에 E4B v0.6.1 대비 측정(미공개) | — |
| 사진, 음성, 영상(`/v1/systemone/media`, `media/`가 있는 MLX 빌드) | 예, 제로샷; 데모 답변 28개 중 7개가 E2B-it와 다름(직접 읽어 보면 6개가 더 나음) | 예, 제로샷 | 예; 데모 답변 28개 중 25개가 E4B와 일치 | 예; 28개 중 21개가 E4B와 일치 | — |
| 라우팅 온도(`"use_case": "routing"`) | 아니요(피팅했으나 배포하지 않음: 일반 T보다 나을 것이 없음) | 예, T 1.18(v0.6.1부터) | 예, T 0.95(v0.6.1부터) | 아니요 | — |
| 자기 학습(결과 로그, 재보정, 학습된 플래그) | 예 | 예, `d1a.serving.serve`로 서빙할 때 | 예 | 예 | — |
| **채팅 / 텍스트 생성** | **예**: 병합되지 않은 빌드(`scripts/export_mlx.py --unmerged`); 어댑터를 끄면 Gemma 4 E4B-it입니다([D1A와 채팅하기](USAGE.md#chat-with-d1a)) | **아니요**: 베이스 트렁크; MLX 빌드는 어댑터가 병합되어 있음 | **아니요**(동일) | **예**: `scripts/export_mlx.py --unmerged`로 빌드하여 서빙합니다. 어댑터를 끄면 Gemma 4 E2B-it입니다([D1A와 채팅하기](USAGE.md#chat-with-d1a)). Mieru iPhone 앱에서도 병합되지 않은 어댑터로 실행됩니다 | 계획됨: 어댑터를 끄면 Gemma 4 12B-it |
| 임베딩 | 함께 서빙하는 GGUF 임베더로 제공(`D1A_EMBED_GGUF`) | — | — | 함께 서빙하는 GGUF 임베더로 제공(`D1A_EMBED_GGUF`) | — |
| 빌드 | PyTorch 어댑터 `JohnP1/d1a-e4b-it`; MLX `JohnP1/d1a-e4b-it-mlx-q8`(미디어 포함 7.5 GB)과 `-mlx-q4`(5.2 GB), 둘 다 병합됨; Core ML 4-bit `JohnP1/d1a-e4b-it-coreml-4bit`(텍스트 전용, ≤512 토큰; 테스트 답변 42개 중 3개가 바뀜, int8은 빌드하지 않음) | PyTorch `JohnP1/d1a-e4b`; MLX 8-bit `JohnP1/d1a-e4b-mlx-q8`(미디어 인코더 1.0 GB 포함 7.5 GB)과 미러 `mlx-community/d1a-e4b-8bit`; Core ML int8 프리뷰 `JohnP1/d1a-e4b-coreml`(텍스트 전용, ≤512 토큰) | PyTorch `JohnP1/d1a-e2b`; MLX 8-bit `JohnP1/d1a-e2b-mlx-q8`(4.8 GB)과 미러 `mlx-community/d1a-e2b-8bit`; Core ML int8 `JohnP1/d1a-e2b-coreml`(텍스트 전용, ≤512 토큰) | PyTorch 어댑터 `JohnP1/d1a-e2b-it`; MLX `JohnP1/d1a-e2b-it-mlx-q8`(4.8 GB)과 `-mlx-q4`(3.6 GB), 둘 다 병합됨; Core ML `JohnP1/d1a-e2b-it-coreml`과 `-coreml-4bit` | 4-bit Q4_0 내보내기 준비됨(`scripts/export_mlx.py --q4-0`, #272) |
| 보정 온도 | 1.55 | 1.95 | 1.45 | 1.41 | — |

PR 라벨링과 라우팅 전용의 이전 단일 목적 빌드는 D1A-E4B v0.6.1로 대체되었습니다.

<a id="builds"></a>

## 빌드

| | |
|---|---|
| Gemma 4 E2B / E4B(및 Qwen)에서의 학습과 서빙 | 예, 이 저장소 |
| Gemma 4 12B(통합 모델, 예: `google/gemma-4-12B-it-qat-q4_0-unquantized`)를 베이스로 사용 | 두 백엔드 모두 로드할 수 있음(#270: PyTorch는 transformers를 통해, MLX는 mlx-lm의 Gemma 4 클래스를 통해); 아직 학습된 D1A-12B는 없음 |
| D1A-E2B v0.6(Gemma 4 E2B: 이후의 모든 D1A 스킬을 한 번의 학습에 담음, 온디바이스용) | [JohnP1/d1a-e2b](https://huggingface.co/JohnP1/d1a-e2b) `v0.6`: v0.2 대비 어려운 의사결정 31% → 62%, 개발자 도구 51% → 65%, 긴 문서 73% → 86%, PR 변경 유형 40% → 85%; 학습에 쓰인 어떤 데이터보다도 최신인 PR 487건에서 심각도 77%, 변경 유형 88%(E4B v0.6: 79%, 91%)이지만 여기서 P0나 P1은 한 번도 예측하지 않으므로 PR 라벨러는 계속 E4B입니다. 명시적 예외로 릴리스됨: 일반 의사결정(decision-v7)이 기준치 −1에 대해 −1.4(#229). `v0.6.1`은 같은 가중치에 라우팅 온도(T 0.95)를 추가합니다. `v0.2`(일반 의사결정, 2 에폭, 보정됨)와 `v0.1`은 계속 사용할 수 있습니다 |
| Apple Silicon용 D1A-E2B(MLX, 8-bit) | [JohnP1/d1a-e2b-mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-mlx-q8) `v0.6`과 `v0.6.1`(3.8 GB, 그리고 사진, 음성, 영상 인코더 1.0 GB 추가) |
| D1A-E4B v0.6(Gemma 4 E4B: 이전의 모든 학습 데이터보다 최신인 풀 리퀘스트로 v0.5를 갱신하고, 긴 문서와 라우팅을 더함) | [JohnP1/d1a-e4b](https://huggingface.co/JohnP1/d1a-e4b) `v0.6`: 학습에 쓰인 어떤 데이터보다도 최신인 PR 487건에서 심각도 79%(v0.5: 75%), 변경 유형 91%(87%). 릴리스 규칙의 명시적 예외로 릴리스됨: v0.5 대비 어려운 의사결정에서 2.7점, 이전 PR 테스트 세트의 변경 유형에서 4.1점 떨어지므로(#197, #198) 그 용도에는 `v0.5`를 고정해서 쓰십시오. `v0.6.1`은 같은 가중치에 라우팅 온도(T 1.18)를 추가합니다. `v0.4`, `v0.3`, `v0.2`(Kev의 후반 단계 + 일본어 + 에이전트 라우팅)와 `v0.1`은 계속 사용할 수 있습니다 |
| D1A-E4B-it v0.6(인스트럭션 튜닝된 Gemma 4 E4B-it에 E2B v0.6의 레시피를 적용: 하나의 모델로 의사결정과 채팅) | [JohnP1/d1a-e4b-it](https://huggingface.co/JohnP1/d1a-e4b-it) `v0.6`: 모든 스위트에서 D1A-E2B-it v0.6 대비 정확도가 22개 중 21개에서 향상(통합 +2.5점; 어려운 의사결정 +6.6, 개발자 도구 +3.2, 긴 문서 +2.3), 의사결정당 약 2.9× 느림; 어댑터를 끄면 Gemma 4 E4B-it입니다([D1A와 채팅하기](USAGE.md#chat-with-d1a)). MLX 빌드, 어댑터 병합됨(의사결정 전용): [JohnP1/d1a-e4b-it-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-it-mlx-q8)과 [-mlx-q4](https://huggingface.co/JohnP1/d1a-e4b-it-mlx-q4) |
| Apple Neural Engine용 D1A-E4B-it(Core ML, 4-bit) | [JohnP1/d1a-e4b-it-coreml-4bit](https://huggingface.co/JohnP1/d1a-e4b-it-coreml-4bit) `v0.6`, 프리뷰: 텍스트 전용, 입력은 최대 512 토큰. 28개 요청에서 fp32 레퍼런스와 비교하면 답변 42개 중 3개가 바뀌며(최대 \|Δp\| 0.46), M1 Max GPU에서 p50 6.6 s입니다. 답이 바뀌면 안 되는 곳에는 MLX 8-bit 빌드를 사용하십시오. int8 빌드는 없음 |
| Apple Silicon용 D1A-E2B-it QAT(MLX, 4-bit) | [JohnP1/d1a-e2b-it-qat-mlx-q4](https://huggingface.co/JohnP1/d1a-e2b-it-qat-mlx-q4): Google의 QAT int4 트렁크(`google/gemma-4-E2B-it-qat-w4a16-ct`)에서 D1A-E2B-it v0.6을 다시 학습시켰으며, 그대로 유지됩니다(2.9 GB, 그리고 QAT 체크포인트의 사진, 음성, 영상 인코더 0.9 GB 추가); 의사결정, 채팅, 사진, 음성, 영상을 지원합니다. 같은 트렁크 위의 v0.6 어댑터 대비 통합 +2.1점이고, bf16의 D1A-E2B-it v0.6 대비로는 통합 +0.4, PR 심각도 +3.5, PR 라벨 +1.7이지만 transfer-v4는 −5.2입니다. 소유자의 2026-10-25 결정에 따라 v0.6을 대체하는 권장 E2B-it 빌드이며, D1A-E4B v0.6의 경우와 같은 릴리스 예외입니다(#341). 모든 스위트: [스코어카드](../reports/2026-10-25-e2b-it-qat-scorecard.md). **v0.7**(2026-10-30; v0.6은 태그 `v0.6`): 영어와 일본어 대화 행위와 MASSIVE 1.1 인텐트로 추가 학습했으며, 이전의 모든 학습 소스도 다시 학습했습니다. 대화 행위는 보류된 498문항에서 +15.9점 [+12.2, +19.7], 인텐트 분류는 그대로(MASSIVE −1.2, 유의하지 않음), 나머지 스위트는 통합 −0.38(유의하지 않음), 일부 스위트에서 보정이 약간 나빠졌고 속도는 같습니다. 지연 시간 거부권이 측정 설정 때문으로 확인된 뒤 소유자의 2026-10-30 결정으로 릴리스했습니다(수치는 [모델 카드](https://huggingface.co/JohnP1/d1a-e2b-it-qat-mlx-q4)) |
| Apple Silicon용 D1A-E4B-it QAT(MLX, 4-bit) | [JohnP1/d1a-e4b-it-qat-mlx-q4](https://huggingface.co/JohnP1/d1a-e4b-it-qat-mlx-q4): Google의 QAT int4 트렁크(`google/gemma-4-E4B-it-qat-w4a16-ct`)에서 D1A-E4B-it v0.6을 다시 학습시켰으며, 그대로 유지됩니다(4.4 GB, 그리고 QAT 체크포인트의 사진, 음성, 영상 인코더 0.9 GB 추가); 의사결정, 채팅, 사진, 음성, 영상을 지원합니다. v0.6과 달리 Gemma 4의 레이어별 임베딩을 올바르게 읽으며 학습했습니다(#353). D1A-E4B-it v0.6 대비 통합 동등(−0.2), PR 라벨 +1.6, PR 심각도 +6.0, 라우팅 +1.7과 +1.5이지만 외부 typesafe는 −7.8입니다(#361). 같은 트렁크 위의 v0.6 어댑터 대비 통합 +1.2이며 유의한 손실은 없습니다. 소유자의 2026-10-27 결정에 따라 릴리스되었습니다(#341). 모든 스위트: [스코어카드](../reports/2026-10-28-e4b-it-qat-scorecard.md); PyTorch: [JohnP1/d1a-e4b-it-qat](https://huggingface.co/JohnP1/d1a-e4b-it-qat) |
| Apple Silicon용 D1A-E4B(MLX, 8-bit) | [JohnP1/d1a-e4b-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-mlx-q8) `v0.6`과 `v0.6.1`(6.6 GB, 그리고 사진, 음성, 영상 인코더 1.0 GB 추가) |
| Apple Neural Engine용 D1A-E4B(Core ML, int8) | [JohnP1/d1a-e4b-coreml](https://huggingface.co/JohnP1/d1a-e4b-coreml) `v0.6`, 프리뷰: 텍스트 전용, 입력은 최대 512 토큰([Apple Neural Engine (Core ML)](APPLE.md#apple-neural-engine-core-ml)) |
| Apple Neural Engine용 D1A-E2B(Core ML, int8) | [JohnP1/d1a-e2b-coreml](https://huggingface.co/JohnP1/d1a-e2b-coreml) `v0.6`: 텍스트 전용, 입력은 최대 512 토큰([Apple Neural Engine (Core ML)](APPLE.md#apple-neural-engine-core-ml)) |
| 경량 클라이언트(Python, JS) | [`clients/`](../../clients) |

모델 버전: 크기와 형식마다 Hugging Face 저장소 하나, 버전마다 태그 하나(`v0.1`, `v0.2`, `v0.3`, `v0.4`, `v0.5`, `v0.6`, `v0.6.1`, ...)를 둡니다. 각 버전은 직전 버전의 학습을 이어 가며, 모델 카드에 무엇으로 학습했는지 나와 있습니다. 버전은 `JohnP1/d1a-e4b@v0.3`처럼 로드합니다. 이전 태그 이름(`v0.2-hybrid`, `v0.2-2epoch`, ...)도 계속 동작합니다.
