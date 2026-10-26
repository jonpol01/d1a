<p><a href="../APPLE.md">English</a> · <a href="../ja/APPLE.md">日本語</a> · 한국어</p>

> 이 문서는 영어판 [APPLE.md](../APPLE.md)(커밋 9eb86269 기준)의 번역입니다. 내용이 다를 경우 영어판이 정확한 정보입니다.

<a id="apple-silicon"></a>

# Apple Silicon

Apple 하드웨어에서의 D1A: GPU에서 실행하는 MLX와, Neural Engine에서 실행하는 Core ML 프리뷰입니다.

<a id="apple-silicon-mlx"></a>

## Apple Silicon (MLX)

Apple Silicon에서 `d1a.serving.serve`는 Gemma 4 체크포인트를 MLX로 실행합니다(`d1a/backends/mlx.py`, mlx-lm 0.31.3). 어댑터는 로드할 때 bf16 베이스에 병합되며, 모든 요청은 state 한 번에 질문마다 행 하나를 더한 형태로 실행됩니다. `scripts/export_mlx.py`는 이렇게 병합된 모델을, 베이스나 어댑터 없이 로드되는 폴더로 기록하며 선택적으로 양자화할 수 있습니다:

```bash
uv run --extra mlx python scripts/export_mlx.py --run JohnP1/d1a-e2b --out runs/exports/d1a-e2b-mlx-bf16
uv run --extra mlx python scripts/export_mlx.py --run JohnP1/d1a-e2b --q-bits 4 --q-group-size 64 --out runs/exports/d1a-e2b-mlx-4bit
uv run --extra mlx python scripts/export_mlx.py --run <a Gemma 4 12B-it QAT checkpoint> --q4-0 --out runs/exports/d1a-12b-mlx-q4_0   # Q4_0 grid
uv run --extra mlx python scripts/export_mlx.py --run JohnP1/d1a-e2b-it --ct-base google/gemma-4-E2B-it-qat-w4a16-ct@<rev> --unmerged --out runs/exports/d1a-e2b-it-qat-mlx-q4   # a QAT checkpoint's own int4
uv run --extra serve python -m d1a.serving.serve --run runs/exports/d1a-e2b-mlx-4bit --port 8009
```

`--q4-0`은 mlx-lm의 min/max 스케일 대신 llama.cpp의 Q4_0 그리드(scale = d, bias = −8·d)에 맞춰 32개 단위 그룹의 4비트로 기록합니다. `google/gemma-4-12B-it-qat-q4_0-unquantized`처럼 Q4_0용으로 학습된 베이스를 위한 것으로, 그 체크포인트에서 Q4_0의 가중치 오차는 mlx-lm의 4-bit 그룹 32보다 약 20%, 그룹 64보다 30% 낮습니다(#270).

`--ct-base`는 run의 베이스 대신, 같은 모델의 compressed-tensors QAT 체크포인트(예: `google/gemma-4-E2B-it-qat-w4a16-ct`)를 기반으로 빌드합니다. 이 체크포인트의 int4 Linear(대칭, 32개 단위 그룹, int32 하나에 8개씩 낮은 니블부터 채워 패킹, 각각 q + 8로 저장)는 이미 MLX의 4-bit 레이아웃이므로 정확히 그대로 유지됩니다. 워드는 있는 그대로, scale s와 bias −8·s이며 s·q로 역양자화됩니다. 나머지 텐서도 있는 그대로 유지되고, 임베딩은 평소처럼 양자화하거나, `--q-embedding-bits`로 임베딩 자체의 비트 폭(8이면 체크포인트의 bf16에 더 가깝게 유지됨)을, `--q-embedding-group-size`로 그룹 크기(QAT는 Linear의 그룹을 32로 고정하며 임베딩의 그룹은 고정하지 않습니다. 64로 하면 D1A-E2B-it에서 0.17 GB를 절약합니다)를 지정해 양자화합니다. 어댑터를 병합하면 그리드를 벗어나게 되므로 `--unmerged`가 필요합니다. D1A-E2B-it에서는 양자화된 Linear 276개가 모두 원본과 정확히 일치합니다(d1a#341). 이러한 체크포인트로 학습한 run(`--base_ct`, docs/TRAINING.md)은 `--ct-base` 없이도 그 체크포인트를 기반으로 내보내며, 다른 체크포인트는 거부합니다. `--media`는 QAT 체크포인트 자체의 인코더를 기록합니다. 베이스의 것과 같은 bf16이며, 비전 프로젝션은 int4 텍스트 모델에 맞춰 QAT로 다시 학습된 것입니다.

일치도는 fp32 PyTorch 경로에서 얻은 골든 벡터(`scripts/golden_vectors.py`: 레코드 209개, 질문 274개. decision-v7 개발 레코드, 플레이그라운드 프리셋, 긴 state 레코드 3개)에 대해 측정합니다. M1 Max에서 JohnP1/d1a-e2b의 결과이며, |Δp|는 한 질문에서 어느 선택지든 확률이 가장 크게 바뀐 값입니다:

| | 가중치 | 최대 \|Δp\| | 평균 \|Δp\| | Argmax 뒤집힘 | 정확도(dev, fp32 0.811) | ECE (fp32 0.056) |
|---|---|---|---|---|---|---|
| MLX bf16 | 8.7 GB | 0.043 | 0.003 | 5 (모두 레퍼런스 마진 0.025 미만) | 0.824 | 0.061 |
| MLX 4-bit, 그룹 64, 임베딩 포함 | 2.5 GB | 0.363 | 0.054 | 23 | 0.811 | 0.051 |
| MLX 8-bit, 레이어별 임베딩 4-bit(`--q-bits 8 --q-per-layer-bits 4`) | 3.5 GB | 0.084 | 0.009 | 5 (마진 0.053 미만) | 0.824 | 0.062 |

4-bit 내보내기는 정확도는 유지하지만 개별 확률을 너무 크게 움직여 fp32 모델을 대신할 수 없으므로(답의 8%가 바뀜) 공개하지 않습니다. 오차는 선형 레이어에서 나옵니다. 선형 레이어를 bf16으로 두고 두 임베딩을 4비트로 하면 최댓값이 0.089입니다. 혼합 내보내기(8-bit 선형 레이어와 토큰 임베딩, 4-bit 레이어별 임베딩. 레이어별 임베딩은 E2B 가중치의 절반입니다)는 bf16에 가깝게 유지됩니다. M1 Max에서의 프로세스 메모리 사용량: 4-bit 내보내기는 로드 후 3.1 GB, 혼합 내보내기는 4.2 GB이며 서빙 후에는 약 1 GB 더 늘고, 질문 60개짜리 요청은 최대 5.3 / 6.4 GB에 이릅니다. 질문 6개짜리 요청은 약 0.4 s, 60개짜리는 약 4.5 s가 걸리며 둘 중 어느 쪽이든 같습니다.

Gemma 4의 레이어별 임베딩은 메모리에 올려 두지 않고 요청마다 가중치 파일에서 읽습니다(요청에는 자기 토큰들의 행만 필요합니다). 8-bit E2B 내보내기는 로드 후 4.22 GB 대신 2.96 GB를, E4B는 6.88 GB 대신 5.37 GB를 사용하며, 확률은 비트 단위로 동일하고 더 느려지지도 않습니다(조회에 패스당 약 0.25 ms가 더해짐). `D1A_PLE_FLASH=0`은 이를 메모리에 유지합니다.

MPS에서의 PyTorch(`D1A_BACKEND=torch`, Mac에서의 학습, PyTorch 미디어 서버): torch 2.14 이전에는 MPS가 2^31개를 넘는 요소를 가진 임베딩 테이블에서 2^31 / 테이블 너비를 넘는 모든 행에 대해 0으로 채워진 행을 반환하며(pytorch/pytorch#188756), Gemma 4의 레이어별 테이블이 그만큼 큽니다. E2B는 토큰 id 239,675부터(탭, 일부 CJK 문자, 이모지, 이미지 토큰), E4B는 199,729부터(모든 문장 부호와 숫자 토큰) 잘못된 값이 나옵니다. D1A는 MPS에서 이런 테이블을 인덱싱으로 읽으며, 이 방식은 정확합니다(#353). MLX, CUDA, CPU는 영향을 받은 적이 없습니다. 이 수정 이전의 MPS에서 실행한 PyTorch는 해당 토큰의 레이어별 입력을 0으로 읽었습니다.

<a id="apple-neural-engine-core-ml"></a>

## Apple Neural Engine (Core ML)

D1A는 Core ML을 통해 Apple Neural Engine에서도 실행되며, iPhone/iPad용 프리뷰입니다. Mac에서는 MLX가 더 빠릅니다. 아래의 v0.6 빌드는
텍스트 전용이며 ≤ 512 토큰입니다(#224). D1A-E2B-it QAT의 빌드는 다섯 가지 기능을 모두 지원합니다([아래](#d1a-e2b-it-qat-on-core-ml)):
- 텍스트 모델은 6개 또는 7개 레이어 단위의 청크로 변환되며(E2B는 5개 청크, E4B는 7개), 가중치는 int8입니다(64개 블록 단위).
- 임베딩 테이블은 호스트에 남아 행 조회 방식으로 쓰입니다.
- 각 청크는 자체 프로세스에서 실행됩니다.

두 Core ML 빌드는 명확한 질문에서는 모두 전체 정밀도 모델과 같은 답을 냅니다. Mac에서는 GPU에서 실행하는 MLX가
몇 배 더 빠르므로 서빙 호스트는 MLX를 유지합니다. Core ML은 iPhone과 iPad용이며, 그곳에서는 GPU를 비워 둘 수 있습니다. v0.6
빌드에는 사진, 음성, 영상 인코더가 없습니다. 각 Core ML 저장소의 `reference/run.py`는 내려받은 파일로 요청에 답합니다.

2026-10-11에 같은 실제 요청 28개(질문 42개, 모두 ≤ 512 토큰)로 Apple M1 Max와 Apple M4에서 측정했습니다.

| | E2B · MLX | E2B · Core ML | E4B · MLX | E4B · Core ML |
|---|---|---|---|---|
| 버전 | v0.6 | v0.6 | v0.6 | v0.6 |
| 저장소 | [JohnP1/d1a-e2b-mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-mlx-q8) | [JohnP1/d1a-e2b-coreml](https://huggingface.co/JohnP1/d1a-e2b-coreml) | [JohnP1/d1a-e4b-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-mlx-q8) | [JohnP1/d1a-e4b-coreml](https://huggingface.co/JohnP1/d1a-e4b-coreml) |
| 실행 환경 | Mac GPU | Neural Engine(iPhone, iPad, M4급 Mac) | Mac GPU | Neural Engine |
| 텍스트 | 예 | 예 | 예 | 예 |
| 사진, 음성, 영상(16프레임, 소리 없음) | 예 | 아직 아님(다음 단계) | 예 | 아직 아님 |
| 입력 길이 | 긴 문서 | ≤ 512 토큰 | 긴 문서 | ≤ 512 토큰 |
| 질문 | 예/아니오, 선택, 점수. 각 선택지마다 보정된 확률 제공 | 동일 | 동일 | 동일 |
| 언어 | 질문은 영어, 문서는 영어 또는 일본어 | 동일 | 동일 | 동일 |
| PR 라벨, 최신 PR 487건: 유형 / 심각도 | 87.5% / 77.2%; P0/P1 30건 중 0건 찾음 | 동일 모델 | 90.6% / 79.1%; P0/P1 30건 중 13건 | 동일 모델 |
| 크기 | 3.8 GB + 미디어 1.0 GB | 3.8 GB (청크 2.0 GB + 임베딩 1.7 GB) | 6.6 GB + 미디어 1.0 GB | 6.5 GB (청크 4.2 GB + 임베딩 2.3 GB) |
| 요청 중앙값, M4 (p95) | **0.20 s** (0.26) | 0.96 s (0.97) | 0.63 s (0.89) | 1.80 s (1.91) |
| 요청 중앙값, M1 Max (p95) | **0.17 s** (0.23) | 9.7 s (10.0) ² | 0.81 s (1.33) | 2.93 s (2.95) |
| 로드, M4 / M1 Max | 2 s / 9 s | 40 s / 247 s | 3 s / – | 52 s / 106 s |
| fp32 PyTorch 대비 답 변경 ¹ | 명확한 37개 중 0개(M4: 42개 중 근소한 차이 1개) | 명확한 37개 중 0개(M4: 42개 중 근소한 차이 1개) | 명확한 38개 중 0개 | 명확한 38개 중 0개 |
| 사진 / 음성 / 영상, 중앙값(M1 Max) | 데모 요청 14개에 대해 0.83 s | – | – | – |

¹ 각 빌드는 같은 v0.6 모델의 fp32 PyTorch 재빌드와 비교합니다. "명확함"은 레퍼런스의 최상위
답이 0.05보다 크게 앞선다는 뜻이며, 근소한 차이는 어느 쪽으로든 뒤집힐 수 있습니다.
² M1 Max의 Neural Engine에서는 E2B 청크가 CPU로 폴백합니다. 각 청크는 Neural Engine을 쓰든 쓰지 않든 같은 시간이
걸립니다. M4는 같은 파일을 자신의 Neural Engine에서 실행합니다.

<a id="d1a-e2b-it-qat-on-core-ml"></a>

### Core ML의 D1A-E2B-it QAT

[JohnP1/d1a-e2b-it-qat-coreml](https://huggingface.co/JohnP1/d1a-e2b-it-qat-coreml)은 권장 E2B-it 빌드
([d1a-e2b-it-qat-mlx-q4](https://huggingface.co/JohnP1/d1a-e2b-it-qat-mlx-q4))를 Core ML로 변환한 것으로, 의사결정, 채팅,
사진, 음성, 영상을 지원합니다(#341). 4.6 GB.
- **텍스트 모델:** 35개 레이어가 5개 패키지로 나뉘며, 각 패키지는 두 가지 프리필 크기(512와 1,536 토큰)와 2,048 포지션 KV 캐시 위의 1토큰 디코드를 갖고
  가중치 사본 하나를 공유합니다. Google의 QAT int4 가중치는 정확히 그대로 유지되며(276개 레이어,
  체크포인트 자체의 정수와 스케일), D1A 어댑터는 입력으로 크기를 조절하는 별도 분기입니다(의사결정은 1, 채팅은 0).
- **나머지:** 채팅용 int8 출력 레이어, 사진 및 영상 인코더(fp16이며 픽셀 경로는 fp32), 음성
  인코더(fp32), 호스트 행 조회로 쓰이는 임베딩 테이블(4-bit).
- **GPU에서 실행되며** Neural Engine에서는 실행되지 않고, 아직 iPhone이나 iPad에서 실행해 보지 않았습니다. 저장소의
  Python 레퍼런스 러너 `reference/d1a_coreml_qat.py`로 사용하며, `io.json`에는 포팅을 위한 호스트 쪽 내용이 나열되어 있습니다.
- **Apple M1 Max에서의 검증:**
  - 의사결정: MLX 빌드 대비 답 42개 중 0개 뒤집힘, fp32 대비 확률 최대 변화 0.0017(MLX 빌드
    자체는 fp32와 0.035 차이);
  - 사진, 음성, 영상(플레이그라운드의 요청 14개, 답 28개): MLX 빌드 대비 0개 뒤집힘;
  - 채팅(사진 8장, 답변 토큰 427개, teacher-forced)을 CPU의 fp32 PyTorch와 비교: 토큰 1개가 다름(근소한 차이).
- **M1 Max에서의 시간:** 사진이나 음성 클립이 있는 의사결정은 건당 약 0.4 s, 16프레임 영상은 8~11 s,
  채팅은 토큰당 약 85 ms.
