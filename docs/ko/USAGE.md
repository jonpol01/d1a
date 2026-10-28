<p><a href="../USAGE.md">English</a> · <a href="../ja/USAGE.md">日本語</a> · 한국어</p>

> 이 문서는 영어판 [USAGE.md](../USAGE.md)(커밋 9eb86269 기준)의 번역입니다. 내용이 다를 경우 영어판이 정확한 정보입니다.

<a id="usage"></a>

# 사용법

D1A를 호출하는 방법입니다. System One API, 인프로세스 라이브러리, 에이전트용 MCP 도구, 클라이언트, 그리고 사진, 음성, 영상을 다룹니다.

<a id="the-system-one-api"></a>

## System One API

`POST /v1/systemone`은 `state`(문서), `model` 이름, 그리고 다수의 `questions`를 받으며, 각 질문은 `noul`(예/아니오), `choice`(이름이 붙은 선택지 중 하나), `score`(순서가 있는 척도) 중 하나입니다. 응답에는 질문마다 답이 하나씩 들어 있고 선택지별 확률이 포함되며, 토큰 사용량과 지연 시간도 함께 담깁니다. `GET /v1/models`는 허용되는 모델 이름을 서빙 세부 정보와 함께 나열하며, 부하 정보(처리한 요청 수와 배치 수, 큐, 프리픽스 캐시 적중 수, 최근 배치의 모델 처리 시간(p50, p95, max))도 포함합니다. `GET /metrics`는 스크레이퍼를 위해 같은 내용을 (메모리 정보와 함께) Prometheus 텍스트 형식으로 보고합니다. `/v1`과 마찬가지로 `D1A_API_KEY`가 설정되어 있으면 베어러 키가 필요합니다.

요청에는 유스케이스를 지정할 수도 있습니다(`"use_case": "routing"`, 선택 사항이며 최대 64자, 앞뒤 공백은 무시됩니다. D1A 고유 필드이므로 다른 System One 서버는 무시합니다). 체크포인트에 그 유스케이스의 온도가 있으면 요청의 확률은 그 온도로 계산됩니다. `use_case`가 없거나 체크포인트에 온도가 없는 이름이면 체크포인트 자체의 온도가 적용되어 답은 이전과 정확히 같습니다. 가장 높은 답은 바뀌지 않으며, 바뀌는 것은 그 답에 대한 확신 정도뿐입니다. `GET /v1/models`는 체크포인트의 `use_case_temperatures`를 나열하고(대부분의 체크포인트는 `{}`), 의사결정 로그에는 각 요청의 유스케이스와 그 확률을 계산한 온도가 기록됩니다. 에이전트 프리셋, MCP 도구, `D1A.decide(..., use_case=...)`, 두 클라이언트가 모두 이를 전달합니다.

**조건부 질문(`ask_if`, #36).** 같은 요청 안에서, 앞선 답이 허용할 때만 질문을 하도록 할 수 있습니다.
`"ask_if": {"<question id>": <condition>}`이며, 모든 조건이 충족되어야 합니다:

| 조건이 지정하는 질문 | 조건 |
|---|---|
| `choice` | 선택지 키의 목록: 그 답이 목록 중 하나임 |
| `noul` | `true` 또는 `false`: 그 답(P(yes) ≥ 0.5)이 해당 값임 |
| `score` | `{"min": .., "max": ..}`, 둘 중 어느 쪽도 생략 가능: 그 답의 `score`(0부터 센 기대 수준)가 범위 안에 있음 |

```json
{"state": "Order 4471 was charged twice.",
 "questions": {"intent": {"type": "choice", "instructions": "What does the customer want?", "criteria": {"refund": null, "track": null}},
               "amount": {"type": "score", "instructions": "How much is at stake?", "criteria": ["small", "medium", "large"], "ask_if": {"intent": ["refund"]}},
               "carrier": {"type": "choice", "instructions": "Which carrier?", "criteria": {"ups": null, "dhl": null}, "ask_if": {"intent": ["track"]}}}}
```

- **패스:** 서버는 패스(pass) 단위로 답합니다. `ask_if`가 없는 질문이 먼저이고, 그다음에 각 질문이 자신이 지정한
  질문들 뒤에 옵니다. state는 캐시되므로 이후 패스는 해당 질문 행의 비용만 듭니다.
- **건너뛴 질문:** 조건이 충족되지 않았거나 건너뛴 질문을 지정한 질문은
  `{"type": .., "skipped": true, "ask_if": {..}}` 형태로, 요청의 순서대로 반환됩니다.
- **조건이 읽는 답:** 학습된 플래그와 마찬가지로 모델 자체의 답입니다(결과 보정 이전).
- **하나의 요청:** 사용량과 `latency_ms`는 모든 패스에 걸쳐 합산되며, 의사결정 로그에는 항목이 하나만 남습니다.
- **검증:** 알 수 없는 질문이나 자기 자신을 지정하는 조건, 순환, 질문 유형에 맞지 않는 조건은 422입니다.
- **모델 입력이 되지 않음:** `ask_if`는 D1A 고유의 필드이므로 이 필드가 없는 요청은 이전과 정확히 같게 답해집니다.
  `/v1/systemone/permute`와 `/v1/systemone/separate`는 질문을 단독으로 답하기 때문에 이 필드를 거부합니다. `D1A.decide`도
  같은 질문을 받습니다.

**라우팅 온도(v0.5 시기의 수치).** 라우팅은 첫 번째 유스케이스입니다. v0.5는 T 1.78로 서빙되어 라우팅 답의 확신도가 낮게 나오는데, 홀드아웃 팩토리 라우팅 행에서 라우팅 온도 0.85를 쓰면 ECE가 0.132에서 0.030으로 낮아지고 확신도 0.9 이상인 답의 비율이 0.28에서 0.66으로 올라가며 정확도는 같습니다(#209). v0.6은 라우팅 온도 없이 출시되었고, v0.6.1에서 D1A-E4B는 T 1.18, D1A-E2B는 0.95로 라우팅 온도를 갖게 되었으며(#264), 그 외 모든 요청은 체크포인트 자체의 온도(1.95와 1.45)를 유지합니다.

모델 이름: `GET /v1/models`는 `d1a-latest`만 나열하지만, 서버는 요청이 어떤 모델 이름을 보내든 답하므로, 다른 이름으로 작성된 클라이언트(예를 들어 설정하지 않은 TypeSafe SDK 클라이언트)도 계속 동작합니다. `Authorization: Bearer <key>`를 요구하려면 `D1A_API_KEY`를 설정합니다.

TypeSafe SDK(`uv sync --extra serve`에 포함됨)를 쓰는 예입니다. 영어가 아닌 텍스트도 읽습니다:

```python
from typesafe_sdk import Choice, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009", model="d1a-latest")
r = client.system_one(
    state="注文した靴が2週間遅れて届き、サイズも間違っていました。",
    questions={"team": Choice(instructions="Which team should handle this?",
                              criteria={"returns": None, "shipping": None, "billing": None})},
)
print(r.choices["team"].choice, r.choices["team"].probabilities)
```

<a id="in-your-own-code-no-server"></a>

## 직접 작성한 코드에서(서버 없이)

체크포인트를 한 번 로드하고 같은 프로세스 안에서 질문합니다. 질문과 답은 System One API와 같은 형태이므로, 코드를 둘 사이에서 그대로 옮길 수 있습니다:

```python
from d1a import D1A

m = D1A.load("JohnP1/d1a-e2b-mlx-q8")   # Apple Silicon (MLX, 3.8 GB download); JohnP1/d1a-e2b on NVIDIA or CPU
answers = m.decide("Shoes arrived late and I was charged twice.",
                   {"team": {"type": "choice", "instructions": "Which team should handle this?",
                             "criteria": {"returns": "returns", "shipping": "shipping", "billing": "billing"}},
                    "urgent": {"type": "noul", "instructions": "Is this urgent?"}})
print(answers["team"]["probabilities"], answers["urgent"]["noul"])
```

<a id="from-agents-mcp"></a>

## 에이전트에서 사용하기(MCP)

`d1a.agents.mcp_server`는 에이전트(Hermes, Claude Code, 모든 MCP 클라이언트)에게 에이전트 팩토리의 의사결정을 도구로 제공합니다. `d1a_intake`(새 요청: 어느 워커가 맡을지, 얼마나 큰 모델이 필요한지, 먼저 사용자에게 물을지), `d1a_judge`(카드와 그 최신 보고서: 완료, 차단됨, 사람 필요, 다음 단계), `d1a_tier`(구현에 쓸 소형, 중형, 대형 모델), `d1a_gate`(이 도구 호출을 실행해도 되는지: allow, ask, deny), `d1a_route`(프롬프트에 답할 모델 크기), `d1a_decide`(직접 작성한 질문)입니다. 각 도구는 확률이 포함된 답과 함께, 기본값이 페일세이프인 동작인 `advice`를 반환합니다. 확신이 없는 답은 상위로 올리거나, 사람에게 묻거나, 카드를 열어 둡니다(`d1a.agents.presets.advise`).

```bash
pip install "d1a[mcp] @ git+https://github.com/jonpol01/d1a"
D1A_URL=http://127.0.0.1:8009 python -m d1a.agents.mcp_server        # stdio; asks a running d1a.serving.serve
```

에이전트의 MCP 설정에서 이를 stdio MCP 서버로 등록하고(명령 `python`, 인자 `-m d1a.agents.mcp_server`, 환경 변수 `D1A_URL`), 위의 도구 이름을 허용합니다. `D1A_RUN=<checkpoint>`를 지정하면 서버를 호출하는 대신 MCP 프로세스 안에서 모델을 로드합니다. 질문 세트는 `d1a/agents/presets.py`에 있으며 라우팅 학습 데이터가 묻는 방식 그대로 작성되어 있고, `d1a_decide`를 제외한 모든 도구는 이를 `"use_case": "routing"`과 함께 보냅니다(`presets.USE_CASES`). `advise`는 체크포인트의 온도(`D1A.temperatures` 또는 서버의 `/v1/models` 항목이며, MCP 도구는 호출할 때마다 이를 읽습니다)와 전송된 유스케이스를 받으므로, `fail_up`의 small 임계값(체크포인트의 온도에서 0.7)은 답을 계산한 온도를 따릅니다. 비율을 통해 전달하면 `sigmoid(logit(0.7) × T_checkpoint / T_use_case)`이며, v0.5의 라우팅 T 0.85에서는 0.855입니다. 게이트, 판정(judge), 0.5 임계값은 그대로 유지됩니다.

**Claude Code 게이트 훅.** `d1a.agents.claude_code_gate`는 바로 쓸 수 있는 `PreToolUse` 훅입니다. Claude Code가
도구 호출을 실행하기 전에 D1A의 게이트가 판단합니다.

```json
{"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "python -m d1a.agents.claude_code_gate", "timeout": 30}]}]}}
```

- **D1A의 답:** "deny"는 호출을 거부하고, "ask"는 여러분에게 묻습니다. 각각 D1A의 확률을 사유로 함께 전달합니다.
- **"allow"는 기본적으로 아무것도 더하지 않음:** 훅이 없을 때와 마찬가지로 Claude Code 자체의 권한 규칙이 결정합니다.
  D1A의 "allow"가 확인 프롬프트를 건너뛰게 하려면 `D1A_GATE_ALLOW=1`을 설정합니다.
- **페일세이프:** D1A에 연결할 수 없거나, `D1A_GATE_TIMEOUT`(기본값 10 s)보다 느리거나, 그 밖의 어떤 오류가 나도 훅은 여러분에게 묻습니다.
- **필드 위조 방지:** 명령과 카드는 각각 한 줄로 읽으며 줄바꿈은 이스케이프되므로, 명령이 게이트의 다른 필드를 위조할 수 없습니다.
- **`D1A_URL`을 사용하십시오:** `D1A_RUN`을 쓰면 훅이 호출할 때마다 모델을 로드합니다.
- **읽는 값:** MCP 도구와 마찬가지로 `D1A_URL`/`D1A_RUN`, 그리고 `D1A_GATE_CARD`(에이전트가 맡은 작업,
  기본값은 세션의 디렉터리).
- **다른 도구:** `"Bash|Write|Edit"` 같은 `matcher`를 쓰면 그 도구들도 게이팅하며, 도구의 이름과 입력으로 읽습니다.
- **무엇이고 무엇이 아닌가:** 도구 호출에 대한 세컨드 오피니언이며, 프롬프트 인젝션 방어 수단이 아닙니다. Claude
  Code 자체의 권한을 결코 대체하지 않습니다. D1A-E4B-it v0.6에서 측정한 결과:
  - 공개 프롬프트 인젝션 탐지(devtools-v1)에서 인젝션의 **60%**(75건 중 45건)를 잡아내며, 오탐은 없습니다.
  - 홀드아웃 에이전트 팩토리 도구 호출(분포 내: 학습 데이터와 같은 생성기)에서 게이트의 최상위
    답은 98% 맞고, 페일세이프 권고를 적용하면 질문하거나 거부해야 했던 호출 33건 중 0건을 허용했습니다.

<a id="clients"></a>

## 클라이언트

의존성이 없는 클라이언트가 [`clients/python`](../../clients/python)(PyPI `d1a-client`, import `d1a_client`)과 [`clients/js`](../../clients/js)(npm `d1a-client`)에 있으며, 어떤 System One 서버와도 쓸 수 있습니다:

```python
from d1a_client import Client

client = Client("http://127.0.0.1:8009")
answer = client.decide("Shoes arrived late and I was charged twice.",
                       {"team": {"type": "choice", "instructions": "Which team should handle this?",
                                 "criteria": {"returns": None, "shipping": None, "billing": None}}})
print(answer["answers"]["team"]["probabilities"])
```

<a id="chat-with-d1a"></a>

## D1A와 채팅하기

**인스트럭션 튜닝된** 트렁크 위의 D1A 모델은 채팅 모델이기도 합니다. D1A 어댑터를 끄면 그 트렁크와 정확히
같습니다(은닉 상태가 비트 단위로 동일). `d1a.serving.serve`는 이 점을 이용해, System One 의사결정에 답하는 메모리 속 같은 모델로
OpenAI 형식의 채팅 및 임베딩 요청에도 답합니다. 따라서 OpenAI API용으로 작성된 앱(mem0, n8n,
채팅 UI)이 별도의 채팅 서버 대신 D1A를 쓸 수 있습니다(#274).

- **채팅이 가능한 모델:** 현재는 D1A-E2B-it와 D1A-E4B-it입니다. D1A-12B는 학습이 끝나면 가능합니다. 베이스 트렁크 모델(D1A-E2B, D1A-E4B)은
  불가능하며, 어댑터가 가중치에 병합된 빌드(공개된 `-mlx-q8` / `-mlx-q4` 저장소)도 불가능합니다.
- **PyTorch에서**(NVIDIA, CPU, 또는 MLX가 없는 Mac) LoRA 학습 run 자체를 `D1A_MERGE=0`으로 서빙합니다. 이 설정은 어댑터를
  병합하지 않은 채로 유지하여 끌 수 있게 합니다:
  `D1A_MERGE=0 python -m d1a.serving.serve --run JohnP1/d1a-e2b-it@v0.6 --port 8009`. 이 경우 채팅은 PyTorch에서 생성합니다.
  transformers 자체 생성과 같은 답이 나오지만 MLX 프롬프트 캐시가 없으므로, 긴 대화는 매 턴마다 프롬프트 전체를 읽습니다.
  도구 호출은 MLX에서와 같이 파싱됩니다.
- **달라지는 점:** 채팅은 어댑터를 끈 상태로, 서버의 모델 스레드에서 의사결정 요청 사이에 실행됩니다. 채팅 도중 들어온 의사결정은
  채팅의 토큰 두 개 사이(긴 프롬프트를 읽는 중이라면 그 프롬프트의 두 청크 사이)에서 답해지며, 그동안 어댑터를 켰다가 끝나면 다시 끕니다.
  따라서 답 전체가 아니라 약 토큰 하나만큼만 기다리며, 채팅의 답은 의사결정이 끼어들지 않았을 때와 같습니다.

**1. 공개된 어댑터로 채팅이 가능한 폴더를 빌드합니다**(Apple Silicon. -it 베이스를 내려받으며 E2B는 약 10 GB,
E4B는 약 16 GB입니다. E4B-it에는 `--run JohnP1/d1a-e4b-it@v0.6`을 사용합니다):

```bash
uv run --extra serve python scripts/export_mlx.py --run JohnP1/d1a-e2b-it@v0.6 --q-bits 8 --q-group-size 64 \
    --unmerged --out runs/exports/d1a-e2b-it-chat
```

`--unmerged`는 어댑터를 베이스에 합치지 않고 베이스 옆(`adapter/`)에 두며, 이 폴더에는 베이스의
채팅 템플릿이 들어 있습니다. 28개 요청 벤치에서 이 폴더의 의사결정은 fp32와 일치합니다(뒤집힘 0/42).

**2. 서빙합니다.** 앱에 임베딩이 필요하면(mem0가 그렇습니다) 임베딩도 함께 서빙합니다:

```bash
uv sync --extra serve --extra embed      # llama.cpp, for /v1/embeddings
D1A_EMBED_GGUF=<path to nomic-embed-text-v1.5.Q4_K_M.gguf> \
  uv run --extra serve python -m d1a.serving.serve --run runs/exports/d1a-e2b-it-chat --port 8009
```

GGUF는 `nomic-ai/nomic-embed-text-v1.5-GGUF`의 `nomic-embed-text-v1.5.Q4_K_M.gguf` 파일로, LM Studio가 서빙하는 파일입니다.
LM Studio와 마찬가지로 GPU에서 실행되므로 벡터가 LM Studio가 저장한 벡터와 같고(서빙 호스트에서 코사인 1.00000), 기존 메모리와도 계속 비교할 수 있습니다.

**3. OpenAI 클라이언트가 이 서버를 가리키게 합니다**:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8009/v1", api_key="unused")   # a real key when D1A_API_KEY is set
r = client.chat.completions.create(model="d1a-chat", messages=[{"role": "user", "content": "What is SQL injection?"}])
```

- **엔드포인트:** `POST /v1/chat/completions`(messages, `max_tokens`, `temperature`, `top_p`, `stop`, `tools`와
  `tool_choice`, `response_format`, `stream`), `POST /v1/embeddings`(`input`은 문자열 또는 리스트이며, `encoding_format`은 `float` 또는 `base64`인데
  후자는 n8n 같은 Node OpenAI 클라이언트가 기본으로 보냅니다), 그리고 `GET /v1/models`입니다.
  이 엔드포인트는 의사결정 모델과 함께 `d1a-chat`과 임베더를 나열하며, 유휴 상태의 서버가
  모델을 언로드한 동안(`--idle-unload`)에도 그렇게 합니다.
- **어떤 모델 ID에도 답합니다.** 따라서 다른 모델용으로 설정된 앱(예를 들어 LM Studio ID)은 URL만 바꾸면 됩니다.
- **스트리밍**은 답을 토큰 단위로 보냅니다. `tools`를 쓰면 도구 호출이 파싱된 뒤 하나의 청크로 옵니다.
- **에이전트 루프:** 어시스턴트의 도구 호출을 OpenAI가 주는 그대로(`arguments`는 JSON 문자열) 되돌려 보내고, 이어서
  결과를 담은 `tool` 메시지를 보냅니다. 각 단계는 앞선 프롬프트의 캐시를 재사용합니다. M1
  Max에서 D1A-E4B-it로 측정한 결과: 도구 호출 3번과 요약을 4단계로 처리했고, 콜드 상태에서 4.1 s, 이후 단계당 0.6–1.8 s였습니다. JSON 객체가 아닌 인자는
  422를 받습니다.
- **프롬프트 캐시:** 프롬프트가 앞선 프롬프트를 이어 가는 채팅(에이전트의 다음 단계, 대화의 다음 턴,
  같은 시스템 프롬프트로 시작한 새 세션)은 새 토큰만 프리필합니다. `usage.prompt_tokens_details.cached_tokens`는
  그중 몇 개를 캐시에서 가져왔는지 알려 줍니다. 최대 16개의 프롬프트가 1 GiB 안에 보관되며(`D1A_CHAT_CACHE_MB`, `0`이면 끕니다),
  유휴 언로드 때 모델과 함께 사라집니다. M1 Max에서 D1A-E4B-it로 측정한 결과: 20k 토큰 프롬프트에서 에이전트 한 단계는 33 s 대신 1–1.7 s가 걸리고,
  같은 20k 토큰 시스템 프롬프트로 시작한 새 세션은 2.2 s가 걸립니다.
- **JSON 모드**는 프롬프트에서 모델에게 JSON을 요청할 뿐이며, 제약 디코딩(constrained decoding)은 없습니다.
- **모델의 컨텍스트는 엄격한 한계입니다:** 프롬프트와 `max_tokens`(보내지 않으면 1024)가 이를 넘는 요청
  (D1A-E4B-it의 경우 131,072 토큰, 모델 설정의 `max_position_embeddings`)은 곧바로 400을 받으며, OpenAI의
  `context_length_exceeded`로 반환되고 실행되지 않습니다. 이 오류를 받으면 내용을 압축하는 클라이언트(Hermes가 그렇습니다)는 줄여서 다시 보냅니다.
- **`max_tokens`로 잘린 도구 호출**은 버려집니다. 답에는 그 앞의 텍스트만 남고 도구 호출은 없으며
  `finish_reason`은 `"length"`이므로, 더 큰 `max_tokens`로 다시 시도하십시오.
- **mem0:** OpenAI LLM과 임베더의 base URL을 이 서버로 설정합니다. D1A-E2B-it로 측정한 결과: 사실이 추출되고, 그래프 관계가
  저장되며, 검색으로 찾아집니다. M1 Max에서 추가(add) 한 번에 약 11 s가 걸립니다. 질문에 대한 그래프 검색은 엔티티를 찾지 못합니다. 2B
  모델이 추출 도구를 호출하는 대신 질문에 답해 버리기 때문이며, LM Studio에서도 그랬습니다.

<a id="photos-voice-and-video"></a>

## 사진, 음성, 영상

모델 서버는 이미지, 짧은 음성 클립, 영상에 대한 질문에도 같은 모델로 답하며, 요청 형태는 같고 `media` 필드가 추가됩니다. Apple Silicon에서는 Gemma 4의 비전 및 오디오 인코더가 들어 있는 MLX 빌드를 서빙합니다(`media/`, 약 1 GB이며 그런 첫 요청 때 가져옵니다. `JohnP1/d1a-e4b-mlx-q8@v0.3`에 들어 있고, `scripts/export_mlx.py --media`로 직접 내보낸 결과물에 추가할 수 있습니다):

```bash
uv run --extra serve --extra media python -m d1a.serving.serve --run JohnP1/d1a-e4b-mlx-q8@v0.6.1 --port 8009 --idle-unload 600
```

```bash
curl -s localhost:8009/v1/systemone/media -H 'content-type: application/json' -d '{
  "media": {"type": "image", "data": "'"$(base64 -i parcel.jpg)"'"},
  "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel damaged?"}}
}'
```

`--idle-unload 600`은 요청이 없는 상태로 10분이 지나면 인코더를 포함한 모델을 내리고, 다음 요청 때 다시 로드합니다(M1 Max에서 D1A-E4B는 2 s). `GET /v1/models`는 어느 쪽이든 응답하며 `loaded` 여부를 알려 줍니다. PyTorch 체크포인트(NVIDIA, CPU)에서는 대신 `python -m d1a.serving.media --run JohnP1/d1a-e2b --port 8010`을 실행합니다. 이 서버는 전체 Gemma 4 모델(bf16, E2B 기준 약 10 GB)을 별도 프로세스로 첫 요청 때 로드하고, 유휴 상태로 10분이 지나면 내립니다.


`type`은 `image`(JPEG, PNG, WebP), `audio`(WAV, FLAC 또는 OGG, 최대 30 s이며 샘플 레이트는 무관), `video`(MP4, MOV 또는 WebM, 최대 32 MB: 균등하게 샘플링한 16개 프레임을 각각 타임스탬프가 붙은 이미지로 읽으며 사운드 트랙은 사용하지 않음) 중 하나입니다. 선택 사항인 `state`는 미디어와 함께 텍스트를 추가합니다. 제로샷 테스트 결과는 [DEMOS.md](DEMOS.md#photo-voice-and-video-results)에 있습니다.
