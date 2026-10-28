<p><a href="../AGENTS_GUIDE.md">English</a> · <a href="../ja/AGENTS_GUIDE.md">日本語</a> · 한국어</p>

> 이 문서는 영어판 [AGENTS_GUIDE.md](../AGENTS_GUIDE.md)(커밋 9eb86269 기준)의 번역입니다. 내용이 다를 경우 영어판이 정확한 정보입니다.

<a id="d1a-for-agents"></a>

# 에이전트를 위한 D1A

에이전트는 실제 작업 사이사이에 작은 의사결정을 많이 내립니다. 요청을 누가 맡아야 하는지, 얼마나 큰 모델이 필요한지,
도구 호출을 실행해도 안전한지, 작업이 끝났는지 같은 것들입니다. D1A는 이 각각에 한 번의 순전파(forward pass, 약 0.2–1 s)로 답하며,
모든 선택지에 보정된 확률을 제공합니다. 기본값은 확신이 없는 답을 안전한 동작으로 바꿉니다. 상위 모델로 올리거나, 사람에게
묻거나, 작업을 열어 둡니다.

<a id="what-d1a-decides"></a>

## D1A가 결정하는 것

| 의사결정 | 프리셋 | 답 | 페일세이프 동작 |
|---|---|---|---|
| **접수(Intake):** 새 요청 | `intake` | 어느 워커가 맡을지, 모델 크기, 대상을 명시하고 완료 기준을 말하고 있는지, 먼저 사용자에게 물을지 | 대상이나 완료 상태가 빠진 것으로 보이면 사용자에게 묻습니다 |
| **등급(Tier):** 작업 시작 전의 작업 | `tier` | 소형, 중형, 대형 모델 | D1A가 더 작은 모델로 충분하다고 확신하지 않는 한 상위로 올립니다 |
| **라우팅(Route):** 모든 프롬프트 | `route` | 소형, 중형, 대형 모델 | 동일 |
| **게이트(Gate):** 도구 호출 | `gate` | allow, ask, deny | p ≥ 0.8일 때만 allow, p ≥ 0.5이면 deny, 그 외에는 ask |
| **판정(Judge):** 작업과 워커의 보고서 | `judge` | 완료? 차단? 사람 필요? 다음 단계: complete, rework, consult, repair | D1A가 완료를 확신할 때만 작업을 닫습니다 |

질문은 `d1a/agents/presets.py`에 있으며, D1A의 라우팅 및 에이전트 학습 데이터가 묻는 방식 그대로 작성되어 있습니다.
페일세이프 규칙은 같은 파일의 `advise()`입니다.

<a id="see-it-the-demo"></a>

## 직접 보기: 데모

```bash
python -m d1a.agents.demo              # against the D1A server at D1A_URL (default http://127.0.0.1:8009)
```

데모의 의사결정은 서버의 학습 로그에 기록되지 않으므로, 운영 중인 서버를 대상으로 실행해도 안전합니다.

하나의 작업이 에이전트의 하루를 거쳐 갑니다. 다음은 Apple M4에서 서빙한 D1A-E4B-it v0.6의 실제 출력입니다(지연 시간에는
SSH 터널이 포함됨):

```
[intake] A new request: who takes it, how large a model, ask the user first?
  input:   Fix the failing login test in ~/work/app (branch fix-login). Done when tests/test_login.py passes.
  D1A:     worker=developer (1.00), tier=medium (0.97), has_target=yes (0.86), has_spec=yes (0.92), clear_done=yes (0.98), consult=no (0.06)
  advice:  {"consult": false, "worker": "developer", "tier": "medium"}

[intake] A vague request
  input:   Make the app better.
  D1A:     worker=developer (0.91), tier=large (0.53), has_target=no (0.05), has_spec=no (0.02), clear_done=no (0.04), consult=yes (0.97)
  advice:  {"consult": true, "worker": "developer", "tier": "large"}

[gate] Run the tests
  input:   pytest tests/test_login.py
  D1A:     decision=allow (0.94)
  advice:  {"decision": "allow"}   (313 ms)

[gate] Push to the main branch
  input:   git push origin main
  D1A:     decision=ask (0.67)
  advice:  {"decision": "ask"}   (314 ms)

[gate] Delete the project
  input:   rm -rf ~/work/app
  D1A:     decision=deny (0.99)
  advice:  {"decision": "deny"}   (322 ms)

[judge] The worker reports success
  input:   Fixed the password comparison in auth.py; tests/test_login.py: 12 passed.
  D1A:     done=yes (0.94), blocked=no (0.01), human=no (0.02), next=complete (0.97)
  advice:  {"next": "complete", "human": false, "blocked": false}

[judge] The worker reports a partial fix
  input:   Changed auth.py; tests/test_login.py still fails on test_expired_token.
  D1A:     done=no (0.02), blocked=no (0.17), human=no (0.04), next=rework (0.93)
  advice:  {"next": "rework", "human": false, "blocked": false}

[route] An easy prompt
  input:   What is the capital of France?
  D1A:     route=small (1.00)
  advice:  {"tier": "small"}

[route] A hard prompt
  input:   Find the race condition in this multi-threaded job scheduler and prove the fix is correct.
  D1A:     route=large (1.00)
  advice:  {"tier": "large"}
```

플레이그라운드([jonpol01/d1a-playground](https://github.com/jonpol01/d1a-playground))의
브라우저 UI에는 도구 게이트 데모가 있습니다.

<a id="plug-it-in"></a>

## 연동하기

<a id="1-as-mcp-tools-claude-code-hermes-any-mcp-client"></a>

### 1. MCP 도구로(Claude Code, Hermes, 모든 MCP 클라이언트)

```json
{"mcpServers": {"d1a": {"command": "python", "args": ["-m", "d1a.agents.mcp_server"], "env": {"D1A_URL": "http://127.0.0.1:8009"}}}}
```

- **도구:** `d1a_intake`, `d1a_judge`, `d1a_tier`, `d1a_gate`, `d1a_route`, 그리고 `d1a_decide`(직접 작성한 질문).
  각 도구는 D1A의 답과 `advice`를 반환합니다.
- **MCP extra:** `pip install "d1a[mcp]"`가 필요합니다.
- **키가 필요한 서버:** `env`에 `"D1A_API_KEY": "<the server's key>"`를 추가합니다. 게이트 훅과 데모도
  이 값을 읽습니다.
- **엔드투엔드 테스트:** CI에서 실제 MCP 클라이언트가 서버를 시작하고, 도구를 나열하고, 호출합니다
  (`tests/test_agents.py`).

<a id="2-as-a-claude-code-gate-hook"></a>

### 2. Claude Code 게이트 훅으로

```json
{"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "python -m d1a.agents.claude_code_gate", "timeout": 30}]}]}}
```

- **D1A의 답:** "deny"는 호출을 거부하고, "ask"는 여러분에게 묻습니다.
- **"allow"는 기본적으로 아무것도 더하지 않음:** Claude Code 자체의 규칙이 결정합니다. `D1A_GATE_ALLOW=1`을 설정하면 이를 그대로 통과시킵니다.
- **페일세이프:** D1A에 연결할 수 없거나, 느리거나(`D1A_GATE_TIMEOUT`, 10 s), 오류가 나면 훅은 여러분에게 묻습니다.
- **더 보기:** 자세한 내용과 설정은 [USAGE.md](USAGE.md#from-agents-mcp)에 있습니다.

<a id="3-in-your-own-code"></a>

### 3. 직접 작성한 코드에서

```python
from d1a import D1A
from d1a.agents.presets import PRESETS, USE_CASES, advise, gate_state

m = D1A.load("JohnP1/d1a-e2b-mlx-q8")
answers = m.decide(gate_state("developer", "Fix the login test", "git push origin main"), PRESETS["gate"], use_case=USE_CASES["gate"])
advise("gate", answers, temperatures=m.temperatures, use_case=USE_CASES["gate"])   # {"decision": "ask"}
```

같은 방식이 HTTP로도 동작하며, [클라이언트](USAGE.md#clients)와 `"use_case": "routing"`을 사용합니다.

<a id="how-well-it-decides-and-its-limits"></a>

## 판단 성능과 한계

서빙된 상태 그대로 D1A-E4B-it v0.6에서 측정했습니다:

| 측정 항목 | 결과 |
|---|---|
| 홀드아웃 에이전트 팩토리 항목(분포 내: 학습 데이터와 같은 생성기), 최상위 답 | 게이트 98%, 등급(tier) 96%, 판정의 다음 단계 92%, 접수 워커 91% |
| 같은 항목에 페일세이프 권고를 적용 | 질문하거나 거부해야 하는 호출 33건 중 **0건**이 허용됨; 끝나지 않은 작업 33건 중 **0건**이 종료됨; 모든 "deny"가 거부됨 |
| 사람이 라벨링한 프롬프트 45개, 라우팅 | 45개 중 45개 |
| 공개 프롬프트 인젝션 탐지(devtools-v1) | **60%**(75건 중 45건)를 탐지, 오탐 없음 |
| 공개 유해 콘텐츠 탐지(aegis, devtools-v1) | 84%(75건 중 63건)를 탐지, 75건 중 오탐 11건 |

**게이트는 세컨드 오피니언이지 보안 경계가 아닙니다.** 학습한 종류의 도구 호출에는 강하지만,
공개 데이터에서 프롬프트 인젝션의 40%를 놓칩니다. Claude Code 자체의 권한이나 샌드박스, 또는 되돌릴 수 없는 작업에 대한
사람의 확인을 결코 대체하지 않습니다.
