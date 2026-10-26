<p><a href="../DEMOS.md">English</a> · <a href="../ja/DEMOS.md">日本語</a> · 한국어</p>

> 이 문서는 영어판 [DEMOS.md](../DEMOS.md)(커밋 9eb86269 기준)의 번역입니다. 내용이 다를 경우 영어판이 정확한 정보입니다.

<a id="demos"></a>

# 데모

D1A가 한 번의 순전파(forward pass)로 답하는 13가지 유스케이스(그중 12가지는 플레이그라운드에서 녹화), 그리고 사진, 음성, 영상 데모가 보여 주는 것과 보여 주지 않는 것입니다.

<a id="see-it-running"></a>

## 실행 모습 보기

D1A가 각각 한 번의 순전파로 답하는 13가지 유스케이스이며, 아래의 12가지는 [플레이그라운드](https://github.com/jonpol01/d1a-playground)에서 실시간으로 녹화했습니다. 1–9는 M4 Mac에서 MLX 8-bit로, 10–11은 M1 Max에서 [`d1a.serving.media`](USAGE.md#photos-voice-and-video)를 통해, 12는 서빙 호스트의 D1A-E4B v0.2로 녹화했습니다:

<table>
<tr><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/routing.gif" alt="D1A에서 실행 중인 모델 라우팅 데모" width="100%"><br><b>1. 모델 라우팅</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/guardrails.gif" alt="D1A에서 실행 중인 가드레일 데모" width="100%"><br><b>2. 가드레일</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/tools.gif" alt="D1A에서 실행 중인 도구 호출 게이팅 데모" width="100%"><br><b>3. 도구 호출 게이팅</b></td></tr>
<tr><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/inbox.gif" alt="D1A에서 실행 중인 받은편지함 분류 데모" width="100%"><br><b>4. 받은편지함 분류</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/rerank.gif" alt="D1A에서 실행 중인 재순위화 데모" width="100%"><br><b>5. 재순위화</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/evals.gif" alt="D1A에서 실행 중인 LLM 평가 데모" width="100%"><br><b>6. LLM 평가</b></td></tr>
<tr><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/labeling.gif" alt="D1A에서 실행 중인 대량 라벨링 데모" width="100%"><br><b>7. 대량 라벨링</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/control.gif" alt="D1A에서 실행 중인 실시간 제어 데모" width="100%"><br><b>8. 실시간 제어</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/gate.gif" alt="D1A에서 실행 중인 신뢰도 게이트 데모" width="100%"><br><b>9. 신뢰도 게이트</b></td></tr>
<tr><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/057ed20659fe42d5192345cfa7ec754e09e3394d/docs/gifs/en/photo.gif" alt="D1A에서 실행 중인 사진 확인 데모" width="100%"><br><b>10. 사진 확인</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/057ed20659fe42d5192345cfa7ec754e09e3394d/docs/gifs/en/voice.gif" alt="D1A에서 실행 중인 음성 분류 데모" width="100%"><br><b>11. 음성 분류</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/42b07a301d95ca0accd46e003a57dc3704011391/docs/gifs/en/prlabel.gif" alt="D1A에서 실행 중인 PR 라벨러 데모" width="100%"><br><b>12. PR 라벨러</b></td></tr>
</table>

<a id="playground"></a>

## 플레이그라운드

위의 데모는 [jonpol01/d1a-playground](https://github.com/jonpol01/d1a-playground)에 있으며, `d1a.serving.serve`와 통신하는 Next.js 앱입니다. Kev의 저장소 내 개발자용 플레이그라운드(패킹 답과 개별 답 비교, 선택지 순열, 체스)와 라벨 검토 페이지는 제거되었습니다. [docs/removed-tools.md](../removed-tools.md)에 이들에 대한 설명과 복원 방법이 있습니다. 이들이 사용하던 서버 엔드포인트인 `/v1/systemone/separate`와 `/v1/systemone/permute`는 그대로 남아 있습니다.

<a id="photo-voice-and-video-results"></a>

## 사진, 음성, 영상 결과

체크포인트는 텍스트만으로 학습되었으므로 이는 제로샷입니다. 첫 테스트([#78](https://github.com/jonpol01/d1a/issues/78))에서 D1A-E2B는 배송 사진 6장 중 6장에서 파손을, 6장 중 5장에서 배송물을 둔 장소를, 영어와 일본어 음성 메모 4개 중 4개에서 요청 내용을 읽어 냈습니다. 플레이그라운드의 샘플 10개에서 D1A-E4B v0.3은 파손 6개 중 5개, 장소 6개 중 6개, 요청 4개 중 4개를 맞힙니다. 측정된 결과가 아니라 데모로 받아들이십시오. 영상도 같은 방식으로 읽습니다. 샘플 사진으로 만든 단일 장면 클립에서 D1A-E4B v0.3은 사진일 때와 같은 답을 내며, 사건의 순서에 대한 질문("마지막에는 어디에 있는가?")은 아직 신뢰할 수 없습니다.
