"""Japanese dates in a state rewritten as "Month D, YYYY" before the decision pass (#410). Opt-in: off unless the server's
D1A_JA_DATES=1 or a request's "ja_dates": true turns it on, so served answers are unchanged otherwise.

D1A reads urgency from dates far better in that form: on Japanese documents whose dates alone were rewritten this way,
urgency accuracy rose from 75.0% to 81.7% (83.0% for the English versions) and AUROC from 0.724 to 0.826 (d1a#23, Part D).

    normalise("締切は2026年8月8日（金）です")   ->  "締切はAugust 8, 2026（金）です"
    with_ja_dates(state)                        ->  the state with every string value normalised, if it holds Japanese

Rewritten:
- 2026年8月8日 and 2026/8/8 (also 2026/08/08 and a full-width ／);
- era dates: 令和8年8月8日, 令和元年5月1日 (元 is year 1), 平成 and 昭和;
- full-width digits inside such a date: ２０２６年８月８日;
- a range that starts with a full date: 2026年8月8日〜10日 and 2026年8月30日〜9月2日 (〜, ～ or ~). The end takes the
  start's stated year. When it cannot (it would fall before the start, over a new year, or is not a real day), only the
  start is rewritten and the rest stays as written. 〜10日間 is a duration, not a range end.
A weekday written after a date, （金） or (金), is kept as written. A digit right after a rewritten date (a time,
2026年8月8日10時) gets a space before it, so it cannot run into the year.

Left as written: a date without a year (8月8日), a date that is not a real day (2026年2月30日, 2026/13/1), digits that run
on into the date (12026年…), a slash date with a year outside 1900-2199 or right after a letter, a slash, "=", "." or "-"
(a model number, a path or URL, a query: 型番1234/5/6, /2026/8/8/, ?d=2026/8/8), spaced forms (2026年 8月 8日), dotted and
abbreviated forms (2026.8.8, R8.8.8), ISO dates (2026-08-08) and relative terms (来週, 明日). A state with no Japanese
script (kana or CJK ideographs) anywhere is returned unchanged, so an English document is never touched.

Measured so far: Japanese documents with 年月日 and era dates (d1a#23). Slash dates, ranges, weekdays and dates with a time
are covered by the unit tests only.

The same rewrite on the client side, before sending: `from d1a.core.ja_dates import with_ja_dates` on the whole state (not
`normalise` on pieces of it, which skips the check for Japanese); a client in another language applies the rules above.
"""
import re
from datetime import date

from d1a.core.api import MONTH_NAMES, render

ERAS = {"令和": 2018, "平成": 1988, "昭和": 1925}   # era year 1 (元年) is base + 1
DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
JAPANESE = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
D = "[0-9０-９]"
WEEKDAY = r"(?:（[月火水木金土日]）|\([月火水木金土日]\))"
KANJI = re.compile(
    rf"(?<![0-9０-９])(?:(?P<era>令和|平成|昭和)(?P<ey>元|{D}{{1,2}})|(?P<y>{D}{{4}}))年(?P<m>{D}{{1,2}})月(?P<d>{D}{{1,2}})日(?P<wd>{WEEKDAY})?"
    rf"(?:(?P<sep>[〜～~])(?:(?P<em>{D}{{1,2}})月)?(?P<ed>{D}{{1,2}})日(?!間)(?P<wd2>{WEEKDAY})?)?")
SLASH = re.compile(rf"(?<![0-9０-９/／A-Za-z=.\-])(?P<y>{D}{{4}})[/／](?P<m>{D}{{1,2}})[/／](?P<d>{D}{{1,2}})(?![0-9０-９/／])(?P<wd>{WEEKDAY})?")
SLASH_YEARS = range(1900, 2200)   # a slash date outside these is more likely a part or model number


def _day(y, m, d):
    try: return date(y, m, d)
    except ValueError: return None


def _written(day):
    return f"{MONTH_NAMES[day.month - 1]} {day.day}, {day.year}"


def _num(s):
    return int(s.translate(DIGITS))


def _spaced(m, out):
    """A space after a rewritten date that ends in its year when a digit follows, so a time (10時) cannot run into it."""
    nxt = m.string[m.end():m.end() + 1]
    return out + " " if nxt and nxt in "0123456789０１２３４５６７８９" and out[-1].isdigit() else out


def _kanji(m):
    if m["era"]:
        n = 1 if m["ey"] == "元" else _num(m["ey"])
        year = ERAS[m["era"]] + n if n >= 1 else None
    else:
        year = _num(m["y"])
    start = _day(year, _num(m["m"]), _num(m["d"])) if year else None
    if start is None: return m[0]
    out = _written(start) + (m["wd"] or "")
    if m["sep"]:
        end = _day(year, _num(m["em"]) if m["em"] else start.month, _num(m["ed"]))
        if end is None or end < start: return out + m[0][m.start("sep") - m.start():]   # the end's year is not stated: as written
        out += m["sep"] + _written(end) + (m["wd2"] or "")
    return _spaced(m, out)


def _slash(m):
    day = _day(_num(m["y"]), _num(m["m"]), _num(m["d"])) if _num(m["y"]) in SLASH_YEARS else None
    return m[0] if day is None else _spaced(m, _written(day) + (m["wd"] or ""))


def normalise(text):
    """`text` with its Japanese dates written as "Month D, YYYY"; the same string when it holds none."""
    if not isinstance(text, str): return text
    return SLASH.sub(_slash, KANJI.sub(_kanji, text))


def _walk(value):
    if isinstance(value, str): return normalise(value)
    if isinstance(value, dict): return {k: _walk(v) for k, v in value.items()}
    if isinstance(value, list): return [_walk(v) for v in value]
    return value


def with_ja_dates(state):
    """The state with every string value (never a key) normalised, when anything in it is written in Japanese; the same
    object otherwise, and whenever nothing changes."""
    if not JAPANESE.search(render(state)): return state
    out = _walk(state)
    return state if out == state else out
