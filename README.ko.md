<p align="center"><a href="README.md">English</a> · <a href="README.ja.md">日本語</a> · 한국어</p>

> 이 문서는 영어판 [README.md](README.md)(커밋 9eb86269 기준)의 번역입니다. 내용이 다를 경우 영어판이 정확한 정보입니다.

<p align="center">
  <img src="docs/d1a-avatar.png" width="220" alt="D1A: 은백색과 짙은 보라색 머리카락에 둥근 안경과 헤드폰을 한 애니메이션 소녀가 #D1A라고 적힌 둥근 에나멜 배지에 그려져 있습니다">
</p>

<h1 align="center">D1A</h1>

<p align="center"><b>Gemma 4 기반의 작은 의사결정 모델입니다.</b> 문서 하나와 유형이 지정된 질문 묶음을 넣으면, 한 번의 순전파(forward pass)로 모든 선택지에 대한 보정된 확률이 나옵니다. 텍스트 생성은 하지 않습니다.</p>

<p align="center">
  <a href="https://github.com/jonpol01/d1a/actions/workflows/ci.yml"><img src="https://github.com/jonpol01/d1a/actions/workflows/ci.yml/badge.svg?branch=main" alt="ci"></a>
  <a href="https://github.com/jonpol01/d1a/actions/workflows/license.yml"><img src="https://github.com/jonpol01/d1a/actions/workflows/license.yml/badge.svg?branch=main" alt="license check"></a>
  <a href="https://github.com/jonpol01/d1a/releases"><img src="https://img.shields.io/github/v/release/jonpol01/d1a" alt="release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="license"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue" alt="python"></a>
</p>
<p align="center">
  <a href="https://huggingface.co/google/gemma-4-E4B"><img src="https://img.shields.io/badge/base-Gemma%204%20E2B%20%7C%20E4B-4285F4" alt="base model: Gemma 4 E2B and E4B"></a>
  <a href="https://huggingface.co/JohnP1/d1a-e4b"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20model-JohnP1%2Fd1a--e4b-yellow" alt="model"></a>
  <a href="https://huggingface.co/collections/JohnP1/d1a-for-pytorch-cuda-cpu-non-mac"><img src="https://img.shields.io/badge/PyTorch-CUDA%20%7C%20CPU-ee4c2c" alt="PyTorch collection"></a>
  <a href="https://huggingface.co/collections/JohnP1/d1a-for-mlx-apple-silicon-gpu"><img src="https://img.shields.io/badge/Apple%20Silicon-MLX-black" alt="MLX collection"></a>
  <a href="https://huggingface.co/collections/JohnP1/d1a-for-core-ml-apple-neural-engine"><img src="https://img.shields.io/badge/Neural%20Engine-Core%20ML%20preview-555555" alt="Core ML collection"></a>
  <a href="https://huggingface.co/mlx-community/d1a-e4b-8bit"><img src="https://img.shields.io/badge/mlx--community-8bit-black" alt="mlx-community mirror"></a>
  <a href="https://github.com/jonpol01/d1a-playground"><img src="https://img.shields.io/badge/demos-d1a--playground-7c3aed" alt="demos"></a>
</p>

> **Built on Kev.** D1A는 Jared Palmer의 [Kev](https://github.com/jaredpalmer/kev)를 기반으로 하며, [Apache License 2.0](LICENSE)에 따라 라이선스됩니다. 이 저장소의 모델 코드, 트레이너, 벤치마크, 고정된 평가 스위트는 Kev를 복사한 것에서 시작했습니다(업스트림 커밋 `0fe8fc9`). [docs/UPSTREAM.md](docs/UPSTREAM.md)에 Kev에서 가져온 모든 파일과 D1A가 바꾼 내용이 나와 있습니다. D1A는 독립 프로젝트입니다. Jared Palmer나 Kev 저자들과 제휴 관계에 있지 않으며, 이들의 후원이나 보증을 받지 않았습니다.

<a id="proven-at"></a>

## 검증 결과

- **에이전트 의사결정**(D1A-E4B-it v0.6, 홀드아웃 에이전트 항목): 도구 호출 게이트 98%, 작업 판정 92%, 모델 라우팅은 사람이 라벨링한 프롬프트 45개 중 45개. 페일세이프 권고를 적용하면, 질문(ask)하거나 거부(deny)해야 하는 호출 33건 중 0건이 허용되었습니다
  ([docs/ko/AGENTS_GUIDE.md](docs/ko/AGENTS_GUIDE.md)).
- **어려운 의사결정, 개발자 도구, 긴 문서, 일본어**(D1A-E4B-it v0.6, D1A의 스킬 스위트): 75.7%, 74.3%,
  88.6%, 78.6%([docs/ko/MODELS.md](docs/ko/MODELS.md)).
- **하나의 모델로 의사결정과 채팅:** 어댑터를 끄면 D1A-E4B-it는 OpenAI 형식 엔드포인트 뒤의 Gemma 4 E4B-it입니다. 프롬프트 캐시 덕분에 20k 토큰 프롬프트에서 에이전트 한 단계가 33 s에서 1–1.7 s로 줄어듭니다(M1 Max).
- **일상적으로 쓰는 PR 라벨링**(D1A-E4B v0.6): M4에서 PR당 0.63 s. 학습에 쓰인 어떤 데이터보다도 최신인 PR 487건에서
  심각도 79%, 변경 유형 91%([recipes/pr-labeler](recipes/pr-labeler/README.md)).
- **사진, 음성, 영상**을 Gemma 4 자체 인코더로 제로샷 처리합니다([docs/ko/DEMOS.md](docs/ko/DEMOS.md)).

<a id="quick-start"></a>

## 빠른 시작

Python 3.12 또는 3.13과 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
git clone https://github.com/jonpol01/d1a.git && cd d1a
uv sync --extra serve
uv run --extra serve python -m d1a.serving.serve --run JohnP1/d1a-e4b-mlx-q8@v0.6.1 --port 8009
```

이 명령은 Apple Silicon에서 MLX 빌드를 서빙합니다. 다른 환경에서는 PyTorch 빌드 `--run JohnP1/d1a-e4b@v0.6.1`를 지정합니다(GPU가 있으면 CUDA를 사용하며, `D1A_BACKEND=torch`는 MPS에서 PyTorch를 실행합니다. [Apple Silicon (MLX)](docs/ko/APPLE.md#apple-silicon-mlx) 참고). 처음 실행할 때 모델을 다운로드합니다. `--run`에는 로컬 체크포인트 디렉터리나 Hub 리비전(`repo@rev`)도 지정할 수 있습니다.

모든 설정(서버 플래그, `D1A_*` 변수, 자기 학습, 사진과 음성, 에이전트)은 기본값 및 변경해야 하는 경우와 함께
[docs/CONFIGURATION.md](docs/CONFIGURATION.md)에 정리되어 있습니다.

티켓을 하나 보내 봅니다:

```bash
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
  "model": "d1a-latest",
  "questions": {
    "team":   {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"returns": null, "shipping": null, "billing": null}},
    "urgent": {"type": "noul", "instructions": "Does this need a reply today?"}
  }
}'
```

<a id="pick-a-build"></a>

## 빌드 선택하기

공개된 모든 빌드가 할 수 있는 것과 할 수 없는 것입니다. 제한 사항과 각주: [docs/ko/MODELS.md](docs/ko/MODELS.md#what-each-build-can-and-cannot-do-2026-10-25).

<a id="d1a-e2b-smaller-and-faster-phones-and-small-machines"></a>

### D1A-E2B (더 작고 빠름: 휴대폰과 소형 기기)

| 빌드 | 형식 | 의사결정 | 채팅 | 사진 | 음성 | 영상 | 크기, 비고 |
|---|---|---|---|---|---|---|---|
| **D1A-E2B-it QAT** ([mlx-q4](https://huggingface.co/JohnP1/d1a-e2b-it-qat-mlx-q4)) | MLX 4-bit, 병합 안 됨 | ✅ | ✅ | ✅ | ✅ | ✅ | 3.7 GB; **권장 E2B-it**; v0.7 |
| **D1A-E2B-it QAT** ([d1a-e2b-it-qat](https://huggingface.co/JohnP1/d1a-e2b-it-qat)) | PyTorch | ✅ | ✅⁴ | ✅¹ | ✅¹ | ✅¹ | Gemma 4 E2B-it와 Google의 QAT 체크포인트 위에 얹는 126 MB 어댑터; v0.6 |
| **D1A-E2B-it QAT** ([coreml](https://huggingface.co/JohnP1/d1a-e2b-it-qat-coreml)) | Core ML (Google의 QAT int4, 그대로 유지) | ✅⁵ | ✅⁵ | ✅⁵ | ✅⁵ | ✅⁵ | 4.6 GB; Mac에서 검증했으며 iPhone에서는 아직 검증하지 않음; v0.6 |
| D1A-E2B-it v0.6 ([d1a-e2b-it](https://huggingface.co/JohnP1/d1a-e2b-it)) | PyTorch | ✅ | ✅⁴ | ✅¹ | ✅¹ | ✅¹ | |
| D1A-E2B-it v0.6 ([mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-it-mlx-q8), [mlx-q4](https://huggingface.co/JohnP1/d1a-e2b-it-mlx-q4)) | MLX, 병합됨 | ✅ | ❌³ | ✅ | ✅ | ✅ | 4.8 / 3.6 GB |
| D1A-E2B-it v0.6 ([coreml](https://huggingface.co/JohnP1/d1a-e2b-it-coreml), [coreml-4bit](https://huggingface.co/JohnP1/d1a-e2b-it-coreml-4bit)) | Core ML | ✅² | ❌ | ❌ | ❌ | ❌ | 3.8 / 2.9 GB, 프리뷰 |
| D1A-E2B ([d1a-e2b](https://huggingface.co/JohnP1/d1a-e2b)) | PyTorch | ✅ | ❌ | ✅¹ | ✅¹ | ✅¹ | |
| D1A-E2B ([mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-mlx-q8)) | MLX 8-bit | ✅ | ❌ | ✅ | ✅ | ✅ | 4.8 GB |
| D1A-E2B ([coreml](https://huggingface.co/JohnP1/d1a-e2b-coreml)) | Core ML int8 | ✅² | ❌ | ❌ | ❌ | ❌ | 3.8 GB, 프리뷰 |

QAT 빌드는 낯선 작업 유형(transfer-v4)에서 5.2점을 내주는 대신, D1A 고유의 작업, 속도, 그리고 다섯 가지 기능 모두를 얻습니다
([스코어카드](docs/reports/2026-10-25-e2b-it-qat-scorecard.md)).
그 v0.7(2026-10-30, 현재 MLX 전용)은 영어와 일본어 대화 행위 판단이 향상되었습니다. 사용자가 동의했는지, 감사 인사뿐인지 요청인지,
명령인지 잡담인지(보류된 문항에서 +15.9점). 인텐트 분류와 나머지 스위트는 그대로입니다([모델 카드](https://huggingface.co/JohnP1/d1a-e2b-it-qat-mlx-q4)).

<a id="d1a-e4b-more-accurate-servers-and-macs"></a>

### D1A-E4B (더 정확함: 서버와 Mac)

| 빌드 | 형식 | 의사결정 | 채팅 | 사진 | 음성 | 영상 | 크기, 비고 |
|---|---|---|---|---|---|---|---|
| **D1A-E4B-it QAT** ([mlx-q4](https://huggingface.co/JohnP1/d1a-e4b-it-qat-mlx-q4)) | MLX 4-bit, 병합 안 됨 | ✅ | ✅ | ✅ | ✅ | ✅ | 5.3 GB; v0.6과 동등, typesafe −7.8 |
| **D1A-E4B-it QAT** ([d1a-e4b-it-qat](https://huggingface.co/JohnP1/d1a-e4b-it-qat)) | PyTorch | ✅ | ✅⁴ | ✅¹ | ✅¹ | ✅¹ | Gemma 4 E4B-it와 Google의 QAT 체크포인트 위에 얹는 133 MB 어댑터 |
| D1A-E4B-it ([d1a-e4b-it](https://huggingface.co/JohnP1/d1a-e4b-it)) | PyTorch | ✅ | ✅⁴ | ✅¹ | ✅¹ | ✅¹ | |
| D1A-E4B-it ([mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-it-mlx-q8), [mlx-q4](https://huggingface.co/JohnP1/d1a-e4b-it-mlx-q4)) | MLX, 병합됨 | ✅ | ❌³ | ✅ | ✅ | ✅ | 7.5 / 5.2 GB |
| D1A-E4B-it ([coreml-4bit](https://huggingface.co/JohnP1/d1a-e4b-it-coreml-4bit)) | Core ML 4-bit | ✅² | ❌ | ❌ | ❌ | ❌ | 4.6 GB, 프리뷰 |
| D1A-E4B ([d1a-e4b](https://huggingface.co/JohnP1/d1a-e4b)) | PyTorch | ✅ | ❌ | ✅¹ | ✅¹ | ✅¹ | 서버 기본값 |
| D1A-E4B ([mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-mlx-q8)) | MLX 8-bit | ✅ | ❌ | ✅ | ✅ | ✅ | 7.5 GB |
| D1A-E4B ([coreml](https://huggingface.co/JohnP1/d1a-e4b-coreml)) | Core ML int8 | ✅² | ❌ | ❌ | ❌ | ❌ | 6.5 GB, 프리뷰 |

D1A-E4B-it QAT는 Google의 QAT int4 트렁크에서 D1A-E4B-it v0.6을 다시 학습시킨 것입니다. v0.6 대비 전체적으로 동등하며(통합 −0.2점 [−0.9, +0.5]), PR 라벨 +1.6, PR 심각도 +6.0, 라우팅 +1.7과 +1.5이지만 외부 typesafe 스위트는 −7.8입니다. 같은 트렁크로 옮긴 v0.6 어댑터 대비로는 통합 +1.2이며 유의한 손실은 없습니다([스코어카드](docs/reports/2026-10-28-e4b-it-qat-scorecard.md)).

¹ 별도의 PyTorch 미디어 서버(`python -m d1a.serving.media`)를 통해 사용하며, 이 서버는 전체 모델을 로드합니다(E2B 기준 약 10 GB).
² 텍스트 전용이며 입력은 최대 512 토큰입니다.
³ 공개된 상태로는 병합되어 있습니다. 채팅에 쓰려면 `--unmerged`로 내보내십시오([D1A와 채팅하기](docs/ko/USAGE.md#chat-with-d1a)).
⁴ `D1A_MERGE=0`으로 서빙합니다. 이 설정은 어댑터를 병합하지 않은 채로 유지하므로 채팅 시 어댑터를 끌 수 있습니다.
⁵ 저장소의 Python 레퍼런스 러너(`reference/d1a_coreml_qat.py`, 서버 없음)를 통해 사용합니다. Neural Engine이 아니라 GPU에서 실행되며, 의사결정은 최대 1,536 토큰까지 가능합니다.

D1A-E2B-it QAT는 채팅, 사진, 음성, 영상을 모두 지원하는 유일한 Core ML 빌드이며, 다른 Core ML 빌드는 텍스트 전용입니다.

E4B는 더 정확한 크기이고, E2B는 더 작고 빠르며 휴대폰과 소형 기기용입니다. `-it` 모델은 인스트럭션 튜닝된 Gemma 4 트렁크 위에서 같은 방식으로 학습되므로 하나의 모델로 채팅도 합니다. D1A 어댑터를 끄면 Gemma 4 E4B-it / E2B-it가 됩니다. 채팅에는 병합되지 않은 빌드가 필요하며(`scripts/export_mlx.py --unmerged`, [D1A와 채팅하기](docs/ko/USAGE.md#chat-with-d1a)), 공개된 `-mlx-q8` / `-mlx-q4` 저장소는 어댑터가 병합되어 있어 의사결정에만 답합니다. D1A-E4B-it는 의사결정당 D1A-E2B-it보다 약 2.9× 느리고, 22개 스위트 중 21개에서 더 정확합니다. D1A-E4B와 D1A-E2B의 PyTorch 및 MLX 빌드는 `v0.6.1`(v0.6에 라우팅 온도를 더한 것)이고, `-it` 및 Core ML 빌드는 `v0.6`입니다. Core ML은 iPhone/iPad용 프리뷰이며 Mac에서는 MLX가 더 빠릅니다([docs/ko/APPLE.md](docs/ko/APPLE.md)). v0.6 Core ML 빌드는 텍스트 전용이며 입력은 최대 512 토큰이고, D1A-E2B-it QAT의 빌드는 다섯 가지를 모두 지원하며 Mac에서 검증했지만 아직 iPhone에서는 검증하지 않았습니다. 4-bit E4B-it 빌드는 테스트 답변 42개 중 3개를 바꾸므로, 답이 바뀌면 안 되는 곳에서는 MLX 8-bit를 사용하십시오. 크기, 예외, 이전 버전: [docs/ko/MODELS.md](docs/ko/MODELS.md).

<a id="use-it"></a>

## 사용하기

`POST /v1/systemone`은 `state`(문서)와 `questions`를 받으며, 각 질문은 `noul`(예/아니오), `choice`, `score` 중 하나입니다. 응답에는 모든 질문의 모든 선택지에 대한 확률이 담겨 있습니다. 같은 형태를 서버 없이 여러분의 코드에서도 그대로 쓸 수 있습니다:

```python
from d1a import D1A

m = D1A.load("JohnP1/d1a-e2b-mlx-q8")   # Apple Silicon (MLX, 3.8 GB download); JohnP1/d1a-e2b on NVIDIA or CPU
answers = m.decide("Shoes arrived late and I was charged twice.",
                   {"team": {"type": "choice", "instructions": "Which team should handle this?",
                             "criteria": {"returns": "returns", "shipping": "shipping", "billing": "billing"}},
                    "urgent": {"type": "noul", "instructions": "Is this urgent?"}})
print(answers["team"]["probabilities"], answers["urgent"]["noul"])
```

- Python 및 JS 클라이언트: [docs/ko/USAGE.md](docs/ko/USAGE.md#clients)
- 에이전트(MCP 도구, Claude Code 게이트 훅, 데모): [docs/ko/AGENTS_GUIDE.md](docs/ko/AGENTS_GUIDE.md)
- 사진, 음성, 영상: [docs/ko/USAGE.md](docs/ko/USAGE.md#photos-voice-and-video)

<a id="for-agents"></a>

## 에이전트용

D1A는 에이전트의 작은 의사결정을 빠르고 페일세이프하게 만듭니다. 요청을 누가 맡을지, 얼마나 큰 모델이 필요한지,
도구 호출을 실행해도 되는지(allow, ask, deny), 작업이 끝났는지를 판단합니다.
- **세 가지 연동 방법:** 모든 에이전트용 MCP 도구, Claude Code 게이트 훅, 또는 Python 프리셋.
- **체험해 보기:** `python -m d1a.agents.demo`는 에이전트의 하루 동안 하나의 작업이 거치는 과정을 보여 줍니다.
- **측정 결과:** 홀드아웃 에이전트 항목에서 페일세이프 권고를 적용하면, 질문하거나 거부해야 하는 호출 33건 중 0건을 허용했고,
  끝나지 않은 작업 33건 중 0건을 종료 처리했습니다.
- **한계:** 공개된 프롬프트 인젝션의 60%를 잡아내므로, 보안 경계가 아니라 세컨드 오피니언입니다.

가이드, 데모 출력, 수치: [docs/ko/AGENTS_GUIDE.md](docs/ko/AGENTS_GUIDE.md).

<a id="how-it-works"></a>

## 작동 방식

1. **입력.** 문서는 한 번만 들어가고, 각 질문은 선택지와 `<decide>` 토큰을 갖춘 자체 분기로 뒤따릅니다.
2. **모델.** LoRA 어댑터를 붙인 Gemma 4 백본과 작은 포인터 헤드가 각 선택지의 점수를 매기고, 보정된 소프트맥스가 그 점수를 확률로 바꿉니다.
3. **두 가지 실행 방식.** 패킹 방식(블록 인과 마스크 아래의 단일 시퀀스) 또는 행 방식(문서를 캐시하고 질문마다 짧은 행 하나)이며, 답은 동일합니다.
4. **실행 위치.** `d1a.serving.serve`는 MLX 또는 PyTorch에서 System One API를 제공하며, Gemma 4 자체 인코더로 사진, 음성 메모, 영상을 읽습니다.
5. **결과로부터 학습.** 의사결정은 그것을 내린 모델과 함께 기록되며, 결과가 돌아오면 확률을 다시 맞춥니다.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/arch/model-dark.svg">
  <img src="docs/arch/model-light.svg" alt="LoRA를 붙인 Gemma 4 백본, 포인터 헤드, 선택지에 대한 소프트맥스" width="100%">
</picture>

설명 글과 나머지 다이어그램: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). 정확한 규칙: [docs/SPEC.md](docs/SPEC.md).

<a id="learns-from-outcomes"></a>

## 결과로부터 학습

D1A는 모든 의사결정을 기록합니다. 결과가 돌아오면 확률을 재보정하며(프로덕션에서 사용 중), 홀드아웃 게이트를 통과하는 조건으로 재학습도 할 수 있습니다(아직 게이트를 통과한 재학습은 없습니다). 자세한 내용과 수치: [docs/SELF_LEARNING.md](docs/SELF_LEARNING.md).

<a id="demos"></a>

## 데모

한 번의 순전파로 각각 답하는 13가지 유스케이스는 [jonpol01/d1a-playground](https://github.com/jonpol01/d1a-playground)에 있으며, 그중 12가지(영상 확인 제외)의 녹화본은 [docs/ko/DEMOS.md](docs/ko/DEMOS.md)에 있습니다.

<table>
<tr><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/routing.gif" alt="D1A에서 실행 중인 모델 라우팅 데모" width="100%"><br><b>모델 라우팅</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/tools.gif" alt="D1A에서 실행 중인 도구 호출 게이팅 데모" width="100%"><br><b>도구 호출 게이팅</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/057ed20659fe42d5192345cfa7ec754e09e3394d/docs/gifs/en/photo.gif" alt="D1A에서 실행 중인 사진 확인 데모" width="100%"><br><b>사진 확인</b></td></tr>
</table>

<a id="train-your-own"></a>

## 직접 학습시키기

```bash
uv run python -m d1a.training.recipe run recipes/d1a-e2b.yaml --out runs/d1a-e2b --device cuda
```

L4 한 장이면 충분합니다. 파인튜닝, 보정, 스코어링: [docs/TRAINING.md](docs/TRAINING.md). 레시피: [recipes/README.md](recipes/README.md).

<a id="status"></a>

## 현황

2026-10-22 기준:

- D1A-E4B-it v0.6은 하나의 모델로 매일 사용되고 있습니다. 의사결정(PR 라벨러, 그리고 섀도 모드로 동작하는 Hermes 에이전트의 도구 게이트)과, 로컬 앱과 에이전트를 위한 OpenAI 형식의 채팅 및 임베딩을 프롬프트 캐시와 함께 제공합니다.
- SWE 검증기는 하나의 이슈 안에서 값싼 궤적(trajectory) 휴리스틱보다 나을 것이 없습니다(null result). 결과에 대한 재보정은 효과가 있습니다([docs/SELF_LEARNING.md](docs/SELF_LEARNING.md)).
- 사진, 음성, 영상 데모는 제로샷입니다. 체크포인트는 텍스트만으로 학습되었습니다([docs/ko/DEMOS.md](docs/ko/DEMOS.md)).
- Core ML은 프리뷰이며 Mac에서는 MLX보다 느립니다([docs/ko/APPLE.md](docs/ko/APPLE.md)). v0.6 빌드는 텍스트 전용이고(입력은 최대 512 토큰), D1A-E2B-it QAT의 빌드는 다섯 가지를 모두 지원하며 Mac에서 검증했지만 아직 iPhone에서는 검증하지 않았습니다.
- 한 질문 안의 선택지들은 여전히 서로 영향을 줄 수 있습니다. 질문을 함께 묻든 따로 묻든 확률은 같지만, 선택지 순서가 무관한 것은 아닙니다.
- Kev가 공개하는 벤치마크 수치는 Kev 자체 Qwen 체크포인트에 대한 것이며 D1A에 대한 것이 아닙니다.
- 보고서는 [docs/reports/](docs/reports/)에 있습니다.

<a id="development"></a>

## 개발

```bash
uv sync --extra serve
uv run python -m pytest tests --unit -q   # what CI runs: every test file but the few tests/conftest.py's OUTSIDE_UNIT_TESTS lists
python scripts/check_license.py          # license and provenance rules (see CONTRIBUTING.md)
uv run python scripts/equivalence/metrics.py   # a rewritten module against its last version before the rewrite (api, metrics, benchmark, suite)
```

이 스위트들은 모델 가중치가 필요 없습니다(약 10 MB의 토크나이저는 Hub에서 내려받습니다). `tests/test_model.py`와 `tests/test_mlx.py`는 실제 가중치가 필요합니다. `tests/test_api.py`는 커밋된 작은 체크포인트를 프로세스 안에서 서빙합니다. 대신 실행 중인 서버를 상대로 돌리려면 `D1A_BASE_URL`을 설정합니다(`pytest tests/test_api.py -m server`).

<a id="releases"></a>

## 릴리스

D1A의 코드는 [Semantic Versioning](https://semver.org)으로 버전을 관리하고 [GitHub](https://github.com/jonpol01/d1a/releases)에 릴리스합니다. [CHANGELOG.md](CHANGELOG.md)에는 버전별 변경 내용이 적혀 있으며, 모델 체크포인트는 각자의 Hugging Face 태그를 가집니다. 릴리스를 만드는 방법: [CONTRIBUTING.md](CONTRIBUTING.md#releasing).

<a id="license"></a>

## 라이선스

Apache-2.0: [LICENSE](LICENSE)와 [NOTICE](NOTICE)를 참고하십시오.

- D1A의 코드는 [Kev](https://github.com/jaredpalmer/kev)에서 왔으며, Copyright 2026 Jared Palmer, Apache-2.0입니다. Kev에서 가져와 수정한 파일에는 맨 위에 고지문이 붙어 있으며, [docs/UPSTREAM.md](docs/UPSTREAM.md)에 해당 파일과 모든 변경 사항이 나와 있습니다.
- D1A의 변경 사항은 Copyright 2026 John Soliva, Apache-2.0입니다.
- Gemma 4는 Google이 Apache-2.0으로 공개했으며, Qwen 베이스 모델도 Apache-2.0입니다. 학습 데이터셋은 각자의 라이선스를 가집니다.
- 저자: Kev는 Jared Palmer([@jaredpalmer](https://github.com/jaredpalmer))가 [Devin](https://devin.ai)으로 만들었고, D1A는 John Soliva([@jonpol01](https://github.com/jonpol01))가 만들었습니다.
- Kev에서 이어받은 감사 인사: Kev는 아키텍처 설명 글에 대해 [Archer Hume](https://archerhume.com/posts/jevs-architecture-unmasked)에게, API 설계에 대해 [TypeSafe](https://docs.typesafe.ai/api)에게, 베이스 모델에 대해 [Qwen](https://huggingface.co/Qwen/Qwen3.5-9B-Base)에게, 기여에 대해 [3x3xX3N0N](https://github.com/jaredpalmer/kev/issues/8)과 [Radexito](https://github.com/Radexito)에게 감사를 전합니다. 관련 연구: [Hydragen](https://arxiv.org/abs/2402.05099), [DeFT](https://arxiv.org/abs/2404.00242), [FIRST](https://arxiv.org/abs/2406.15657).
