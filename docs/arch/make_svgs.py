"""Writes the README's architecture diagrams as animated SVGs, one light and one dark file each.

    python docs/arch/make_svgs.py

GitHub shows an SVG in a README as an image: CSS animations inside it play, scripts and web fonts do not. The README
picks the light or dark file with <picture>, so the colors follow the reader's GitHub theme. Shapes follow the
playground's architecture page (jonpol01/d1a-playground, src/components/architecture.tsx).
"""
from pathlib import Path

OUT = Path(__file__).parent
C = {"state": "#0284c7", "q": "#7c3aed", "opt": "#d97706", "decide": "#e11d48", "lora": "#059669", "head": "#c026d3", "slate": "#64748b", "amber": "#f59e0b"}
THEMES = {
    "light": {"bg": "#ffffff", "card": "#ffffff", "border": "#d0d7de", "fg": "#1f2328", "muted": "#59636e", "track": "#eaeef2"},
    "dark": {"bg": "#0d1117", "card": "#161b22", "border": "#3d444d", "fg": "#e6edf3", "muted": "#9198a1", "track": "#262c36"},
}

STYLE = """
text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif; fill: {fg}; }
.mono { font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace; }
.muted { fill: {muted}; }
.title { font-weight: 600; }
.sweep { animation: sweep 6s ease-in-out infinite both; }
@keyframes sweep { 0% { opacity: .22; } 7% { opacity: 1; } 80% { opacity: 1; } 100% { opacity: .22; } }
.glow { animation: glow 6s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }
@keyframes glow { 0%, 55% { stroke-width: 1; } 62% { stroke-width: 4; } 72%, 100% { stroke-width: 1; } }
.flow { stroke-dasharray: 5 5; animation: flow 1s linear infinite; }
@keyframes flow { to { stroke-dashoffset: -20; } }
.wave { animation: wave 2.4s ease-in-out infinite; transform-box: fill-box; transform-origin: bottom; }
@keyframes wave { 0%, 100% { transform: scaleY(.55); } 50% { transform: scaleY(1); } }
.grow { animation: grow 6s cubic-bezier(.2,.8,.2,1) infinite; transform-box: fill-box; transform-origin: left; }
@keyframes grow { 0%, 10% { transform: scaleX(0); } 35%, 85% { transform: scaleX(1); } 100% { transform: scaleX(0); } }
.pop { animation: pop 6s ease-out infinite both; transform-box: fill-box; transform-origin: center; }
@keyframes pop { 0% { opacity: 0; transform: scale(.4); } 8% { opacity: 1; transform: scale(1); } 85% { opacity: 1; } 100% { opacity: 0; } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
"""


def svg(theme, w, h, body, label):
    t = THEMES[theme]
    style = STYLE
    for k, v in t.items(): style = style.replace("{" + k + "}", v)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{label}">'
            f"<title>{label}</title><style>{style}</style>"
            f'<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="{t["muted"]}"/></marker></defs>'
            f'<rect width="{w}" height="{h}" rx="14" fill="{t["bg"]}"/>{body}</svg>')


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tok(t, x, y, w, label, color=None, mono=True, cls="", delay=0.0, extra=""):
    fill, op, stroke = (color, .16, color) if color else (t["card"], 1, t["border"])
    style = f' style="animation-delay:{delay:.2f}s"' if cls else ""
    return (f'<g class="{cls}"{style}><rect x="{x}" y="{y}" width="{w}" height="26" rx="6" fill="{t["card"]}" stroke="{stroke}" {extra}/>'
            f'<rect x="{x}" y="{y}" width="{w}" height="26" rx="6" fill="{fill}" fill-opacity="{op}"/>'
            f'<text x="{x + w / 2}" y="{y + 17}" text-anchor="middle" font-size="11" class="{"mono" if mono else ""}">{esc(label)}</text></g>')


def layout(t):
    """1. The input layout: the state once, then one branch per question; the prefill sweeps left to right."""
    row = [(44, "<bos>", None, 1), (58, "<state>", C["state"], 1), (112, "document tokens…", C["state"], 0),
           (34, "<q>", C["q"], 1), (76, "instructions", C["q"], 0), (44, "<opt>", C["opt"], 1), (30, "A", C["opt"], 1), (48, "</opt>", C["opt"], 1),
           (44, "<opt>", C["opt"], 1), (30, "B", C["opt"], 1), (48, "</opt>", C["opt"], 1), (64, "<decide>", C["decide"], 1),
           (34, "<q>", C["q"], 1), (26, "…", None, 1)]
    x, body, xs = 12, [], []
    for i, (w, label, color, mono) in enumerate(row):
        glow = 'class="glow"' if label == "<decide>" else ""
        body.append(tok(t, x, 40, w, label, color, bool(mono), "sweep", i * 0.18, glow)); xs.append(x); x += w + 4
    q1, q2 = xs[3], xs[12]
    body += [f'<text x="12" y="28" font-size="11" class="muted">document (read once)</text>',
             f'<text x="{q1}" y="28" font-size="11" class="muted">question 1</text>',
             f'<text x="{q2}" y="28" font-size="11" class="muted">question 2</text>',
             f'<text x="12" y="92" font-size="11" class="muted">position ids</text>',
             f'<text x="12" y="112" font-size="11" class="mono">0 1 2 … n−1</text>',
             f'<text x="{q1}" y="112" font-size="11" class="mono">n n+1 n+2 …</text>',
             f'<text x="{q2}" y="112" font-size="11" class="mono">n …</text>',
             f'<path d="M {q2 - 6} 106 C {q2 - 30} 132, {q1 + 20} 132, {q1 + 4} 118" fill="none" stroke="{C["q"]}" class="flow"/>',
             f'<text x="{(q1 + q2) / 2}" y="138" text-anchor="middle" font-size="10" class="muted">positions restart for every question, so each sees exactly what it would see alone</text>']
    return x + 12, 150, "".join(body), "Token layout of a D1A request"


def model(t):
    """2. The model: backbone layers pulse, data flows to the pointer head, the probability bars grow."""
    b = [f'<rect x="8" y="20" width="300" height="190" rx="16" fill="{t["card"]}" stroke="{t["border"]}"/>',
         '<text x="24" y="46" font-size="15" class="title">Gemma 4 E2B / E4B</text>',
         '<text x="24" y="64" font-size="11" class="muted">E2B: 35 layers · hidden 1,536</text>']
    for i in range(35):
        g = i % 5 == 4
        b.append(f'<rect x="{24 + i * 7.6}" y="{80 if g else 98}" width="5.6" height="{58 if g else 40}" rx="1.5" fill="{C["q"] if g else t["muted"]}" '
                 f'opacity="{.9 if g else .45}" class="wave" style="animation-delay:{i * 0.06:.2f}s"/>')
    b += ['<text x="24" y="156" font-size="11" class="muted">4 of every 5 layers: sliding window 512 · 1 of 5: global</text>',
          f'<rect x="24" y="168" width="268" height="30" rx="8" fill="{C["lora"]}" fill-opacity=".14" stroke="{C["lora"]}"/>',
          '<text x="158" y="187" text-anchor="middle" font-size="10">LoRA r=16 on q k v o · gate up down</text>',
          f'<path d="M 308 115 L 356 115" stroke="{t["muted"]}" stroke-width="2" marker-end="url(#arr)" class="flow"/>',
          f'<rect x="360" y="40" width="150" height="150" rx="16" fill="{C["head"]}" fill-opacity=".1" stroke="{C["head"]}"/>',
          '<text x="435" y="64" text-anchor="middle" font-size="14" class="title">Pointer head</text>',
          tok(t, 378, 78, 114, "q(h⟨decide⟩)", C["decide"]), tok(t, 378, 110, 114, "k(h⟨/opt⟩ᵢ)", C["opt"]),
          '<text x="435" y="158" text-anchor="middle" font-size="11" class="mono">q·kᵢ / √256</text>',
          '<text x="435" y="176" text-anchor="middle" font-size="10" class="muted">q, k: hidden → 256</text>',
          f'<path d="M 510 115 L 548 115" stroke="{t["muted"]}" stroke-width="2" marker-end="url(#arr)" class="flow"/>',
          f'<rect x="552" y="60" width="160" height="110" rx="16" fill="{t["card"]}" stroke="{t["border"]}"/>',
          '<text x="632" y="84" text-anchor="middle" font-size="12" class="title">softmax(z / T)</text>',
          '<text x="632" y="102" text-anchor="middle" font-size="10" class="muted">T fitted on held-out data</text>']
    for i, p in enumerate((0.62, 0.27, 0.11)):
        b += [f'<text x="568" y="{127 + i * 14}" font-size="10" class="mono muted">{"ABC"[i]}</text>',
              f'<rect x="582" y="{119 + i * 14}" width="110" height="6" rx="3" fill="{t["track"]}"/>',
              f'<rect x="582" y="{119 + i * 14}" width="{110 * p:.1f}" height="6" rx="3" fill="{t["fg"] if i == 0 else t["muted"]}" class="grow" style="animation-delay:{i * 0.12:.2f}s"/>']
    return 720, 230, "".join(b), "Backbone, pointer head and calibrated softmax"


def forms(t):
    """3. Two equivalent execution forms: the block-causal mask fills in; the rows branch off a cached document."""
    seg = ["s", "s", "s", "1", "1", "1", "2", "2", "2"]
    color = {"s": C["state"], "1": C["q"], "2": C["opt"]}
    b = [f'<rect x="8" y="8" width="372" height="184" rx="14" fill="{t["card"]}" stroke="{t["border"]}"/>',
         '<text x="24" y="34" font-size="10" class="muted">attention mask (document | Q1 | Q2)</text>']
    for i, a in enumerate(seg):
        for j, c in enumerate(seg):
            on = j <= i and (c == "s" or a == c)
            x, y = 24 + j * 14, 44 + i * 14
            b.append(f'<rect x="{x}" y="{y}" width="12" height="12" rx="2" fill="{t["track"]}"/>')
            if on: b.append(f'<rect x="{x}" y="{y}" width="12" height="12" rx="2" fill="{color[c]}" fill-opacity=".8" class="pop" style="animation-delay:{(i + j) * 0.08:.2f}s"/>')
    b += ['<text x="168" y="52" font-size="12" class="title">Packed: one pass</text>']
    for k, line in enumerate(["the whole request is one sequence", "a block-causal mask lets each question", "see the document and itself only", "sliding layers get a second mask", "whose distance counts position ids"]):
        b.append(f'<text x="168" y="{72 + k * 16}" font-size="10" class="muted">{line}</text>')
    ox = 392
    b += [f'<rect x="{ox}" y="8" width="372" height="184" rx="14" fill="{t["card"]}" stroke="{t["border"]}"/>',
          f'<rect x="{ox + 16}" y="30" width="96" height="30" rx="8" fill="{C["state"]}" fill-opacity=".18" stroke="{C["state"]}"/>',
          f'<text x="{ox + 64}" y="49" text-anchor="middle" font-size="11">document</text>',
          f'<text x="{ox + 64}" y="76" text-anchor="middle" font-size="10" class="muted">prefix cache</text>',
          f'<text x="{ox + 160}" y="46" font-size="12" class="title">Rows: document once</text>',
          f'<text x="{ox + 160}" y="64" font-size="10" class="muted">then each question as its own causal row</text>']
    for i in range(3):
        y = 96 + i * 28
        b += [f'<path d="M {ox + 112} 45 C {ox + 130} 45, {ox + 130} {y + 11}, {ox + 146} {y + 11}" fill="none" stroke="{C["q"]}" class="flow"/>',
              f'<rect x="{ox + 146}" y="{y}" width="140" height="22" rx="6" fill="{C["q"]}" fill-opacity=".14" stroke="{C["q"]}" class="pop" style="animation-delay:{0.3 + i * 0.25:.2f}s"/>',
              f'<text x="{ox + 216}" y="{y + 15}" text-anchor="middle" font-size="10" class="mono">Q{i + 1}: &lt;q&gt; … &lt;decide&gt;</text>']
    return 772, 200, "".join(b), "Packed and rows execution forms"


def serving(t):
    """4. Where D1A runs: clients, the text server and its backends, the media server for photos and voice."""
    def box(x, y, w, title, sub, color):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="62" rx="14" fill="{t["card"]}" stroke="{color}"/>'
                f'<rect x="{x}" y="{y}" width="{w}" height="62" rx="14" fill="{color}" fill-opacity=".1"/>'
                f'<text x="{x + w / 2}" y="{y + 26}" text-anchor="middle" font-size="12" class="title">{esc(title)}</text>'
                f'<text x="{x + w / 2}" y="{y + 44}" text-anchor="middle" font-size="10" class="muted">{esc(sub)}</text>')

    def arrow(x1, x2, y, label):
        return (f'<path d="M {x1} {y} L {x2} {y}" stroke="{t["muted"]}" stroke-width="1.5" marker-end="url(#arr)" class="flow"/>'
                f'<text x="{(x1 + x2) / 2}" y="{y - 6}" text-anchor="middle" font-size="9" class="mono muted">{esc(label)}</text>')
    b = [box(8, 20, 170, "Your code / agents", "TypeSafe SDK · curl · MCP", C["slate"]),
         arrow(178, 308, 51, "/v1/systemone"),
         box(310, 20, 190, "d1a.serve", "System One API · prefix cache", C["q"]),
         arrow(500, 538, 51, ""),
         box(540, 20, 254, "MLX (Apple Silicon) · PyTorch", "8-bit on a Mac · bf16 on CUDA / MPS", C["lora"]),
         box(8, 120, 170, "Photo · voice note", "JPEG / PNG · WAV, ≤ 30 s", C["amber"]),
         arrow(178, 308, 151, "/v1/systemone/media"),
         box(310, 120, 190, "d1a.media", "same answers, media in the state", C["decide"]),
         arrow(500, 538, 151, ""),
         box(540, 120, 254, "Gemma 4 + vision & audio encoders", "adapter merged · no speech-to-text", C["head"])]
    return 802, 200, "".join(b), "Where D1A runs: the text server and the media server"


for name, fn in (("layout", layout), ("model", model), ("forms", forms), ("serving", serving)):
    for theme in THEMES:
        w, h, body, label = fn(THEMES[theme])
        (OUT / f"{name}-{theme}.svg").write_text(svg(theme, w, h, body, label), encoding="utf-8")
print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
