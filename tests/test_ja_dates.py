"""The opt-in Japanese date normaliser (d1a.core.ja_dates, #410): what it rewrites, what it leaves as written, and that a
state without Japanese, or without dates, comes back byte-identical."""
import json

import pytest

from d1a.core.ja_dates import normalise, with_ja_dates

REWRITTEN = [
    ("締切は2026年8月8日です", "締切はAugust 8, 2026です"),
    ("2026/8/8までに", "August 8, 2026までに"),
    ("2026/08/08", "August 8, 2026"),
    ("２０２６年８月８日", "August 8, 2026"),                     # full-width digits
    ("２０２６／８／８", "August 8, 2026"),
    ("令和8年8月8日", "August 8, 2026"),
    ("令和元年5月1日", "May 1, 2019"),                            # 元年 is the era's first year
    ("令和１年５月１日", "May 1, 2019"),
    ("平成31年4月30日", "April 30, 2019"),
    ("平成元年1月8日", "January 8, 1989"),
    ("昭和64年1月7日", "January 7, 1989"),
    ("2026年8月8日（金）に", "August 8, 2026（金）に"),            # the weekday is kept as written
    ("2026年8月8日(金)", "August 8, 2026(金)"),
    ("2026年8月8日〜10日", "August 8, 2026〜August 10, 2026"),     # a range takes the start's stated year
    ("2026年8月30日～9月2日", "August 30, 2026～September 2, 2026"),
    ("2026年8月8日（金）〜10日（日）", "August 8, 2026（金）〜August 10, 2026（日）"),
    ("2026年8月8日〜2026年8月10日", "August 8, 2026〜August 10, 2026"),
    ("会議は2026年8月8日、締切は令和8年9月1日。", "会議はAugust 8, 2026、締切はSeptember 1, 2026。"),
    ("2026年12月30日〜1月2日", "December 30, 2026〜1月2日"),     # the end's year is not stated: only the start is rewritten
    ("2026年8月8日10時まで", "August 8, 2026 10時まで"),         # a time cannot run into the year
    ("２０２６年８月８日１０時", "August 8, 2026 １０時"),
    ("2026年8月8日（金）10時", "August 8, 2026（金）10時"),
    ("2026/8/8 10:00", "August 8, 2026 10:00"),
    ("2026年8月8日〜10日間開催", "August 8, 2026〜10日間開催"),     # a duration, not the end of a range
    ("2026年8月8日〜10日の3日間", "August 8, 2026〜August 10, 2026の3日間"),
]
UNCHANGED = [
    "8月8日",                       # no year
    "8月8日〜10日",
    "2026年2月30日",                # not a real day
    "2026年2月30日〜3月2日",
    "2026/13/1",
    "令和0年5月1日",
    "12026年8月8日",                # digits run on into the date
    "https://example.com/2026/8/8/a",   # a path, not a date
    "型番1234/5/6",                 # a slash "date" outside 1900-2199: a part number
    "v2026/8/8",                    # right after a letter, "=", "." or "-": a version, a query, a file name
    "?d=2026/8/8",
    "x.2026/8/8",
    "2026年 8月 8日",               # spaced, dotted and abbreviated forms are not rewritten
    "2026.8.8",
    "R8.8.8",
    "2026-08-08",                   # ISO: already read well
    "来週の月曜日までに",             # relative terms are out of scope
    "明日まで",
]


@pytest.mark.parametrize("text,want", REWRITTEN)
def test_japanese_dates_are_written_month_day_year(text, want):
    assert normalise(text) == want


@pytest.mark.parametrize("text", UNCHANGED)
def test_what_is_not_a_full_real_date_is_left_as_written(text):
    assert normalise(text) == text


def test_an_english_document_is_byte_identical_even_with_a_slash_date():
    for state in ("Deadline: August 8, 2026. Released 2026-07-22, see /2026/8/8/notes and 2026/8/8.",
                  {"title": "Fix the build", "due": "2026/8/8", "body": ["see #12"]}):
        out = with_ja_dates(state)
        assert out is state and json.dumps(out, ensure_ascii=False) == json.dumps(state, ensure_ascii=False)


def test_a_japanese_document_without_dates_is_byte_identical():
    for state in ("お客様から返金の依頼がありました。来週までに対応してください。", {"件名": "返金", "本文": ["8月中に", "明日まで"]}):
        assert with_ja_dates(state) is state


def test_a_json_state_has_its_string_values_rewritten_never_its_keys():
    state = {"件名": "締切の確認", "2026/8/8": "締切は2026/8/8", "items": [{"日付": "令和8年8月8日（土）"}, 3, None]}
    assert with_ja_dates(state) == {"件名": "締切の確認", "2026/8/8": "締切はAugust 8, 2026",
                                    "items": [{"日付": "August 8, 2026（土）"}, 3, None]}
    assert state["2026/8/8"] == "締切は2026/8/8"   # the input is not modified


def test_rewriting_twice_changes_nothing_more():
    """A client may normalise before sending to a server that has it on too."""
    for text, _ in REWRITTEN:
        once = with_ja_dates(text)
        assert with_ja_dates(once) == once


def test_no_state_is_no_state():
    assert with_ja_dates(None) is None and with_ja_dates("") == ""


Q = {"u": {"type": "noul", "instructions": "urgent?"}}


def test_the_server_rewrites_only_when_asked_and_a_request_overrides_its_setting(monkeypatch):
    from d1a.core.api import SystemOneRequest, with_date_facts
    from d1a.serving import serve
    monkeypatch.setattr(serve, "JA_DATES", False); monkeypatch.setattr(serve, "DATE_FACTS", False)
    req = SystemOneRequest(state="締切は2026年8月8日です", questions=Q)
    assert serve.prepare(req) is req                                               # off: the request itself
    assert serve.prepare(req.model_copy(update={"ja_dates": True})).state == "締切はAugust 8, 2026です"
    monkeypatch.setattr(serve, "JA_DATES", True)
    assert serve.prepare(req).state == "締切はAugust 8, 2026です" and req.state == "締切は2026年8月8日です"
    off = req.model_copy(update={"ja_dates": False})
    assert serve.prepare(off) is off                                               # the request's false wins over the setting
    monkeypatch.setattr(serve, "DATE_FACTS", True)                                 # dates first, then date facts, as before
    both = SystemOneRequest(state="発注は2026年8月1日、納期は2026年8月8日", questions=Q)
    assert serve.prepare(both).state == with_date_facts(with_ja_dates(both.state))


def test_a_request_field_unknown_to_older_servers_is_optional_and_logged_only_when_sent(tmp_path):
    from d1a.core.api import SystemOneRequest
    from d1a.learning.feedback import FeedbackLog
    assert SystemOneRequest(state="x", questions=Q).ja_dates is None
    with pytest.raises(Exception): SystemOneRequest(state="x", questions=Q, ja_dates="yes")
    log = FeedbackLog(tmp_path / "log.jsonl")
    for flag in (None, True, False): log.decision("x", Q, {}, run="r@v", ja_dates=flag)
    events = log.events()
    assert "ja_dates" not in events[0] and events[1]["ja_dates"] is True and events[2]["ja_dates"] is False


def test_a_replay_sends_the_request_as_the_original_was_read(tmp_path, monkeypatch):
    """The log keeps the state as sent, so a replay must carry the request's own ja_dates (and only when it sent one), or the
    new model would read a different state than the one the outcome is about."""
    from d1a.learning import loop
    from d1a.learning.feedback import FeedbackLog
    log, bodies = FeedbackLog(tmp_path / "log.jsonl"), {}

    def http(url, body=None, headers=None, timeout=60):
        if url.endswith("/v1/models"): return {"models": [{"run": "m@new", "batches": {"queued": 0}}]}
        bodies[headers[loop.REPLAY_HEADER]] = body; return {}

    monkeypatch.setattr(loop, "_http", http)
    ids = {}
    for i, flag in enumerate((None, True, False)):
        ids[flag] = log.decision("締切は2026年8月8日", Q, {"u": {"type": "noul", "noul": 0.5}}, run="m@old", ja_dates=flag)
        log.outcome(ids[flag], {"u": True}, {"src": "human", "group": f"o/r#{i + 1}"})
    mix = tmp_path / "mix.jsonl"; mix.write_text(json.dumps({"state": "another", "questions": {}}) + "\n", encoding="utf-8")
    trained = tmp_path / "trained.json"; trained.write_text(json.dumps(loop.build_manifest([mix], "m@new")), encoding="utf-8")
    assert loop.replay(log, "http://s", trained, sleep=lambda s: None)["replayed"] == 3
    assert "ja_dates" not in bodies[ids[None]] and bodies[ids[True]]["ja_dates"] is True and bodies[ids[False]]["ja_dates"] is False
    assert {b["state"] for b in bodies.values()} == {"締切は2026年8月8日"}   # the state as sent; the server rewrites it again


class _Tok:
    def __call__(self, text, add_special_tokens=False):
        from types import SimpleNamespace
        return SimpleNamespace(input_ids=text.split())


def test_in_process_decide_rewrites_only_when_asked():
    """D1A.decide reads no server setting: only its own ja_dates=True rewrites (a stand-in model records the state)."""
    import torch
    from d1a.serving.lib import D1A
    seen = []

    class Model:
        def encode(self, tok, rec, **kw): seen.append(rec["state"]); return {}
        def probs_batch(self, encs, *a): return [[torch.tensor([0.4, 0.6])]], None

    m = D1A(None, None, Model(), "cpu")
    for flag in (None, False, True): m.decide("締切は2026年8月8日", Q, ja_dates=flag)
    assert seen == ["締切は2026年8月8日", "締切は2026年8月8日", "締切はAugust 8, 2026"]


def test_the_media_server_applies_it_as_the_text_server_does(monkeypatch):
    from types import SimpleNamespace
    from d1a.serving import media
    seen = []
    fake = SimpleNamespace(tok=_Tok(), probs=lambda rec, m, u: (seen.append(rec["state"]), ([[0.4, 0.6]], 1))[1])
    req = lambda **kw: media.MediaRequest(state="2026年8月8日の配送", questions=Q, media={"type": "image", "data": ""}, **kw)
    monkeypatch.setattr(media, "JA_DATES", False)
    for r in (req(), req(ja_dates=True)): media.MediaModel.answer(fake, r)
    monkeypatch.setattr(media, "JA_DATES", True)
    for r in (req(), req(ja_dates=False)): media.MediaModel.answer(fake, r)
    media.MediaModel.answer(fake, media.MediaRequest(questions=Q, media={"type": "image", "data": ""}))   # no text beside the media
    assert seen == ["2026年8月8日の配送", "August 8, 2026の配送", "August 8, 2026の配送", "2026年8月8日の配送", ""]
