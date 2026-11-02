"""Build hosted.html — the Hosted (paid API) tab (docs/hosted.md). Called from publish.py's
build_top_level(); reads only hosted data, never the local board's:

    hosted/<slug>/default/p<N>_t<take>.{wav,json}   clips + sidecars (the sweep)
    hosted/<slug>/cloning/p<N>_t<take>.{wav,json}   cloned-voice clips (the cloning lens)
    hosted/speed/samples.jsonl                      multi-day speed sampler (speed_sample.py)
    scoring/hosted_scores.csv                       UTMOS / WER / SIM / Health (score_hosted.py, Linux)

Layout follows the Scores/Speed pages: one short reading guide, then tabs (Board / Listen /
Cloning / Speed detail). Only the board is a <table>: the shared sort script sorts every table
on a page by one column index, so the other sections are CSS grids.

Clips are copied to <worktree>/hosted/<slug>/<lens>/p<N>_t<take>.wav (all takes: no cherry-picking).
"""

import csv
import json
import re
import shutil
import statistics as st
from collections import defaultdict
from html import escape

from scoring.prompts import PROMPTS

from .core import OUT
from .providers import ADAPTERS
from .speed_sample import BOTH, SKIP, transport

REPO = OUT.parent
SCORES = REPO / "scoring" / "hosted_scores.csv"
SAMPLES = OUT / "speed" / "samples.jsonl"

# slug -> (display name, $/1M chars as shown, price note). Source + read dates: docs/hosted.md.
CATALOG = {
    "eleven_v4": ("ElevenLabs Eleven v4", "$80", ""),
    "eleven_v4_turbo": ("ElevenLabs Eleven v4 Turbo", "$40", ""),
    "rime_coda": ("Rime Coda", "$50", ""),
    "giggy_tts": ("Giggy TTS", "$10", "streaming tier"),
    "cartesia_sonic36": ("Cartesia Sonic 3.6", "$38–65", "by plan"),
    "gemini38_flash_tts": ("Gemini 3.8 Flash TTS", "≈$15", "token-priced; doubles 2027-01-01"),
    "inworld_tts2": ("Inworld TTS-2", "$25", "on-demand"),
    "speechify_simba35": ("Speechify SIMBA 3.5", "$12–20", "by plan; launch half price"),
    "fish_drama3": ("Fish Audio Drama 3", "$10", "per 1M bytes; $50 after 2026-11-12"),
    "voxtral3_tts": ("Mistral Voxtral TTS 3", "$16", ""),
    "xai_tts": ("xAI TTS", "$15", ""),
    "assemblyai_tts1": ("AssemblyAI TTS-1", "$15", ""),
    "soniox_tts2": ("Soniox TTS v2", "≈$14.1", "$0.70 per hour of speech"),
    "stepaudio3_tts": ("StepFun Step-Audio 3 TTS", "$36", "≈$18 for English"),
    "smallest_lightning31pro": ("Smallest.ai Lightning v3.1 Pro", "$19.5", ""),
    "deepgram_flux_tts": ("Deepgram Flux TTS", "$45", ""),
    "mai_voice21": ("Microsoft MAI-Voice-2.1", "$22", ""),
    "vui_luna": ("VUI Labs Luna", "$15", ""),
    "openai_gpt4o_mini_tts": ("OpenAI gpt-4o-mini-tts", "≈$17", "token-priced"),
    "openai_realtime21": ("OpenAI gpt-realtime-2.1", "token", "$64 per 1M audio-out tokens"),
    "murf_falcon21": ("Murf Falcon 2.1", "$10", ""),
    "heygen_voice1": ("HeyGen Voice 1", "$30", "$15 via OpenRouter"),
    "minimax_speech28hd": ("MiniMax speech-2.8-hd", "$100", "via OpenRouter"),
    "bland_speech3": ("Bland Speech v3", "$40", ""),
    "gradium_tts": ("Gradium", "$35.9–57.8", "by plan"),
    "async_flash15": ("Async Flash v1.5", "≈$9.3", "$0.50 per hour of audio"),
    "palabra_tts": ("Palabra TTS", "$12", ""),
    "resemble_ultra": ("Resemble Ultra", "≈$33", "$0.0005 per second of audio (Flex)"),
}

# Per-vendor transport note for the A/B grid.
TRANSPORT_NOTE = {
    "cartesia_sonic36": "Socket clearly faster to first audio; board uses it.",
    "assemblyai_tts1": "Both fast; socket a little faster; board uses it.",
    "deepgram_flux_tts": "Socket sends ~0.6 s of silence before speech, so first speech is "
                         "slower than HTTP; board uses HTTP.",
    "xai_tts": "Close; HTTP kept.",
    "inworld_tts2": "Close; HTTP kept.",
    "soniox_tts2": "Socket somewhat faster; both pace audio near playback speed.",
    "smallest_lightning31pro": "Close on first audio; socket's 'complete' message lags ~4 s.",
    "murf_falcon21": "Socket somewhat faster.",
    "stepaudio3_tts": "Socket session drops or stalls often; HTTP kept.",
}

# Vendors without API cloning on our route, and why (docs read 2026-11-01).
NO_CLONE = {
    "rime_coda": "enterprise-trained voices only (30–60 min of audio, days)",
    "giggy_tts": "cloning in the web app only, no API",
    "assemblyai_tts1": "cloning in the dashboard only, no API",
    "deepgram_flux_tts": "no voice cloning",
    "murf_falcon21": "API cloning is Enterprise-plan only",
    "openai_gpt4o_mini_tts": "custom voices are limited to approved customers",
    "openai_realtime21": "custom voices are limited to approved customers",
    "mai_voice21": "needs Microsoft Limited Access approval",
    "minimax_speech28hd": "no cloning through OpenRouter (our route)",
}

STYLE = """<style>
.hgrid{display:grid;gap:.4rem .9rem;align-items:center;margin:.6rem 0 1.2rem;font-size:.92em}
.hgrid>div.h{color:var(--muted);font-size:.85em;border-bottom:1px solid var(--border);padding-bottom:.3rem}
.hgrid.ab{grid-template-columns:minmax(9rem,1.2fr) repeat(3,auto) minmax(10rem,2fr)}
.hgrid.det{grid-template-columns:minmax(10rem,1.5fr) repeat(6,auto)}
.hgrid.clone{grid-template-columns:minmax(10rem,1.5fr) repeat(7,auto)}
.hgrid.listen{grid-template-columns:minmax(9rem,1fr) repeat(3,minmax(0,1.3fr));margin:.4rem 0 .9rem}
.hgrid.listen audio{width:100%;height:32px}
.hgrid .wide{grid-column:span 3}
.hgrid .num{text-align:right;font-variant-numeric:tabular-nums}
.hgrid .lbl{display:none;color:var(--muted)}
.scores-table-scroll{max-width:100%;overflow-x:auto}
.prompt-jumper{margin:.3rem 0 .9rem;color:var(--muted);font-size:.9em}
details.panel{background:var(--panel);border:1px solid var(--border);border-radius:10px;margin-bottom:.8rem;padding:0 1.1rem}
details.panel>summary{cursor:pointer;padding:.75rem 0;font-weight:600;list-style:none}
details.panel>summary::-webkit-details-marker{display:none}
details.panel>summary::before{content:"\\25B8";color:var(--accent);margin-right:.5rem;display:inline-block;transition:transform .12s}
details.panel[open]>summary::before{transform:rotate(90deg)}
details.panel .summ-text{color:var(--muted);font-style:italic;font-weight:400}
.ref-row{margin:.4rem 0 1rem;display:flex;align-items:center;gap:.6rem;flex-wrap:wrap}
.ref-row audio{width:280px;height:32px}
.ref-row .ref-label{color:var(--accent);font-weight:600}
@media(max-width:760px){
 .hgrid .lbl{display:inline}
 .hgrid.ab,.hgrid.det,.hgrid.clone{grid-template-columns:1fr 1fr}
 .hgrid>div.h{display:none}
 .hgrid>div.name{grid-column:1/-1;margin-top:.6rem;font-weight:600}
 .hgrid.ab>div.note{grid-column:1/-1}
 .hgrid.listen{grid-template-columns:1fr}
 .hgrid .wide{grid-column:auto}
}
</style>"""

TAB_SCRIPT = """<script>
(function(){
  var tabs = document.querySelectorAll('#mode-tabs .lens-tab');
  var subs = document.querySelectorAll('.subsection[data-mode]');
  function show(m){
    subs.forEach(function(s){ s.style.display = s.dataset.mode === m ? '' : 'none'; });
    tabs.forEach(function(t){ t.classList.toggle('active', t.dataset.mode === m); });
    try { history.replaceState(null, '', '#' + m); } catch (e) {}
  }
  tabs.forEach(function(t){
    t.addEventListener('click', function(e){ e.preventDefault(); show(t.dataset.mode); });
  });
  var h = location.hash.slice(1);
  show(['board', 'listen', 'cloning', 'speed'].indexOf(h) >= 0 ? h : 'board');
})();
</script>"""


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _med(xs):
    return st.median(xs) if xs else None


def _mean(xs):
    return st.mean(xs) if xs else None


def _p90(xs):
    if len(xs) < 2:
        return None
    return st.quantiles(xs, n=10)[-1]


def _ms(v):
    return "—" if v is None else f"{v:,.0f}"


def _fmt(v, spec):
    return "—" if v is None else format(v, spec)


def _ds(v):
    return f' data-sort="{v:.4f}"' if v is not None else ' data-sort=""'


def _price_sort(price):
    """First number in a price label ("$12–20" -> 12, "≈$9.3" -> 9.3); None for "token"."""
    m = re.search(r"\d+(?:\.\d+)?", price)
    return float(m.group()) if m else None


def _lbl(t):  # inline label, shown only on phones where grid headers are hidden
    return f'<span class="lbl">{t} </span>'


def _sidecars(slug):
    d = OUT / slug / "default"
    out = []
    for p in sorted(d.glob("p*_t*.json")) if d.is_dir() else []:
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except ValueError:
            pass
    return out


def _samples():
    rows = []
    if SAMPLES.exists():
        for line in SAMPLES.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows  # call 0 rows are failed warmups: counted as errors in _speed


def _scores():
    by = defaultdict(list)
    if SCORES.exists():
        with open(SCORES, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                by[r["model"]].append(r)
    return by


def _speed(samples, slug, tr):
    rs = [r for r in samples if r["slug"] == slug and r["transport"] == tr]
    err = sum(1 for r in rs if r["error"])
    # Warm ranking = calls on a reused connection. A call that had to reconnect (cold) is neither
    # an error nor a warm sample; neither is a success with no speech onset to time.
    ok = [r["ttfa_speech_ms"] for r in rs if not r["error"] and not r.get("cold")
          and r.get("call", 0) > 0 and r["ttfa_speech_ms"] is not None]
    days = {r["ts"][:10] for r in rs}
    return dict(med=_med(ok), p90=_p90(ok), n=len(ok), err=err, days=len(days))


def _row_data(samples, scores):
    """Per model: name, speed aggregate, scores, price — shared by the board and the detail grid."""
    out = []
    for slug, cls in ADAPTERS.items():
        if slug in SKIP:
            continue
        name, price, pnote = CATALOG.get(slug, (slug, "?", ""))
        tr = transport(cls)
        sp = _speed(samples, slug, tr)
        cars = [c for c in _sidecars(slug) if c.get("status") == "ok"]
        if not sp["n"]:  # no sampler data yet: warm sweep calls, marked
            xs = [c["ttfa_speech_ms"] for c in cars
                  if c.get("connection") == "warm" and c.get("ttfa_speech_ms") is not None]
            sp = dict(med=_med(xs), p90=_p90(xs), n=len(xs), err=sp["err"], days=sp["days"],
                      sweep=True)  # keep the sampler's failures visible
        p3 = [c["total_ms"] / 1000 for c in cars if c["prompt_id"] == 3 and c.get("total_ms")]
        rs = [r for r in scores.get(slug, []) if r["mode"] == "default"]
        en = [r for r in rs if r["prompt_id"] in ("1", "2", "3", "4")]
        xx = [r for r in rs if r["prompt_id"] in ("5", "6")]
        srs = {c.get("sample_rate") for c in cars if c.get("sample_rate")}
        out.append(dict(
            slug=slug, cls=cls, name=name, price=price, pnote=pnote, tr=tr, sp=sp, rs=rs,
            p3=_med(p3), sr=max(srs) if srs else None,
            um=_med([v for v in (_f(r["utmos"]) for r in en) if v is not None]),
            we=_mean([v for v in (_f(r["wer"]) for r in en) if v is not None]),
            wx=_mean([v for v in (_f(r["wer"]) for r in xx) if v is not None])))
    return out


def _main_table(rows):
    # Plain-text headers: the shared sort script rewrites th.textContent (drops markup).
    # Same cell hierarchy as the Scores table: one plain line per row, metrics as td.num,
    # descriptive text muted, direction arrows in the headers.
    cols = ["Model", "First speech ms ↓", "UTMOS ↑", "WER en ↓", "WER fr/es ↓",
            "$ / 1M chars", "Langs"]
    head = "<thead><tr>" + "".join(
        f'<th{" class=num" if 0 < i < 6 else ""}>{escape(c)}</th>'
        for i, c in enumerate(cols)) + "</tr></thead>"
    body = []
    for d in rows:
        sp, cls = d["sp"], d["cls"]
        flags = sorted({f for r in d["rs"] for f in r["health"].split(";") if f})
        warn = (f' <span class="health-flag" title="clip defect: {escape(", ".join(flags))} '
                f'— listen before trusting">⚠</span>') if flags else ""
        mark = (' <span class="muted" title="from the sweep; multi-day sampling pending">†</span>'
                if sp.get("sweep") else "")
        tip = f'p90 {_ms(sp["p90"])} ms · {sp["n"]} calls · {sp["err"]} errors · {d["tr"]}'
        via = " · via OpenRouter" if cls.route == "openrouter" else ""
        note = "*" if d["pnote"] else ""
        ptitle = f' title="{escape(d["pnote"])}"' if d["pnote"] else ""
        body.append(
            f'<tr><td title="model id: {escape(cls.model)}{via}">{escape(d["name"])}{warn}</td>'
            f'<td class="num"{_ds(sp["med"])} title="{escape(tip)}">{_ms(sp["med"])}{mark}</td>'
            f'<td class="num"{_ds(d["um"])}>{_fmt(d["um"], ".3f")}</td>'
            f'<td class="num"{_ds(d["we"])}>{_fmt(d["we"], ".3f")}</td>'
            f'<td class="num"{_ds(d["wx"])}>{_fmt(d["wx"], ".3f")}</td>'
            f'<td class="num"{_ds(_price_sort(d["price"]))}{ptitle}>{escape(d["price"])}{note}'
            f'</td><td class="muted">{escape(" ".join(sorted(cls.langs)))}</td></tr>')
    return (f'<div class="scores-table-scroll"><table class="hosted-board">{head}<tbody>'
            f'{"".join(body)}</tbody></table></div>')


def _detail_grid(rows):
    head = ["Model", "Transport", "Median ms", "p90 ms", "Calls · errors",
            "Delivery p3", "Sample rate"]
    cells = [f'<div class="h">{h}</div>' for h in head]
    key = lambda d: d["sp"]["med"] if d["sp"]["med"] is not None else float("inf")
    for d in sorted(rows, key=key):
        sp = d["sp"]
        mark = " †" if sp.get("sweep") else ""
        sr = "—" if not d["sr"] else f'{d["sr"] / 1000:g} kHz'
        cells += [
            f'<div class="name">{escape(d["name"])}</div>',
            f'<div>{_lbl("transport")}{escape(d["tr"])}</div>',
            f'<div class="num">{_lbl("median")}{_ms(sp["med"])}{mark}</div>',
            f'<div class="num">{_lbl("p90")}{_ms(sp["p90"])}</div>',
            f'<div class="num">{_lbl("calls · errors")}{sp["n"]} · {sp["err"]}</div>',
            f'<div class="num">{_lbl("delivery")}{_fmt(d["p3"], ".1f")} s</div>',
            f'<div class="num">{_lbl("SR")}{sr}</div>']
    return '<div class="hgrid det">' + "".join(cells) + "</div>"


def _ab_table(samples):
    cells = [f'<div class="h">{h}</div>' for h in
             ("Vendor", "HTTP ms", "WebSocket ms", "Board uses", "Note")]
    for slug in BOTH:
        a, b = _speed(samples, slug, "http"), _speed(samples, slug, "websocket")
        cells += [f'<div class="name">{escape(CATALOG[slug][0])}</div>',
                  f'<div class="num">{_lbl("HTTP")}{_ms(a["med"])} '
                  f'<span class="muted">({a["n"]}/{a["err"]})</span></div>',
                  f'<div class="num">{_lbl("WS")}{_ms(b["med"])} '
                  f'<span class="muted">({b["n"]}/{b["err"]})</span></div>',
                  f'<div>{_lbl("board uses")}{escape(transport(ADAPTERS[slug]))}</div>',
                  f'<div class="muted note">{escape(TRANSPORT_NOTE.get(slug, ""))}</div>']
    return '<div class="hgrid ab">' + "".join(cells) + "</div>"


def _clone_slugs():
    from .core import CLONES
    reg = json.loads(CLONES.read_text(encoding="utf-8")) if CLONES.exists() else {}
    return {s: v for s, v in reg.items() if (OUT / s / "cloning").is_dir()}


def _clone_table(scores):
    from .providers.clone import MIN_REC
    reg = _clone_slugs()
    rows = []
    for slug, info in reg.items():
        cls = ADAPTERS[slug]
        rs = [r for r in scores.get(slug, []) if r["mode"] == "cloning"]
        en = [r for r in rs if r["prompt_id"] in ("1", "2", "3", "4")]
        mean = lambda xs, k: _mean([x for x in (_f(r[k]) for r in xs) if x is not None])
        sim, um, we = mean(rs, "sim"), mean(en, "utmos"), mean(en, "wer")
        mn, rec = MIN_REC.get(slug, ("—", "—"))
        sec = lambda v: f"{v} s" if v != "—" else "—"
        rows.append((sim if sim is not None else -1, [
            escape(CATALOG[slug][0]), f'{info["seconds"]:g} s', sec(mn), sec(rec),
            "—" if sim is None else f"<b>{sim:.3f}</b>", _fmt(um, ".2f"), _fmt(we, ".3f"),
            escape(" ".join(sorted(cls.clone_langs or cls.langs)))]))
    rows.sort(key=lambda r: -r[0])
    head = ["Model", "Sample sent", "Vendor min", "Recommended", "SIM ↑", "UTMOS", "WER en",
            "Clone langs"]
    cells = [f'<div class="h">{h}</div>' for h in head]
    for _, r in rows:
        cells.append(f'<div class="name">{r[0]}</div>')
        cells += [f'<div class="{"num" if i < 7 else ""}">{_lbl(head[i])}{v}</div>'
                  for i, v in enumerate(r) if i]
    grid = '<div class="hgrid clone">' + "".join(cells) + "</div>"
    missing = [s for s in ADAPTERS if s not in reg and s not in SKIP]
    why = "".join(f"<li>{escape(CATALOG[s][0])}: {escape(NO_CLONE.get(s, 'pending'))}</li>"
                  for s in missing)
    if not why:
        return grid
    return (grid + f'<details class="panel"><summary>Not in the cloning lens <span '
            f'class="summ-text">({len(missing)}: no API cloning on our route)</span></summary>'
            f'<ul class="meta">{why}</ul></details>')


def _listen(worktree, lens="default"):
    dest = worktree / "hosted"
    out = []
    slugs = [s for s in ADAPTERS if s not in SKIP and
             (lens == "default" or s in _clone_slugs())]
    for pid, lang, text in PROMPTS:
        rows = []
        for slug in slugs:
            cls = ADAPTERS[slug]
            name = escape(CATALOG.get(slug, (slug,))[0])
            langs = (cls.clone_langs or cls.langs) if lens == "cloning" else cls.langs
            if lang not in langs:
                rows.append(f'<div>{name}</div><div class="muted wide">not offered in '
                            f'{escape(lang)}</div>')
                continue
            cells = []
            for take in (1, 2, 3):
                src = OUT / slug / lens / f"p{pid}_t{take}.wav"
                if not src.exists():
                    cells.append('<div class="muted">missing</div>')
                    continue
                d = dest / slug / lens
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, d / src.name)
                cells.append(f'<div><audio controls preload="none" '
                             f'src="hosted/{slug}/{lens}/{src.name}"></audio></div>')
            rows.append(f'<div>{name}</div>{"".join(cells)}')
        short = escape(text[:90]) + ("…" if len(text) > 90 else "")
        out.append(f'<details class="panel" id="{lens[0]}p{pid}"{" open" if pid == 1 else ""}>'
                   f'<summary>Prompt {pid} <span class="summ-text">[{escape(lang)}] "{short}"'
                   f'</span></summary><div class="hgrid listen"><div class="h">Model</div>'
                   f'<div class="h">Take 1</div><div class="h">Take 2</div>'
                   f'<div class="h">Take 3</div>{"".join(rows)}</div></details>')
    jump = " · ".join(f'<a href="#{lens[0]}p{pid}">P{pid}</a>' for pid, _, _ in PROMPTS)
    return f'<nav class="prompt-jumper">Jump to: {jump}</nav>' + "".join(out)


def _clone_intro(worktree):
    from .core import REF_DIR
    ref = REF_DIR / "tym_hosted_full.wav"
    player = ""
    if ref.exists():
        (worktree / "hosted").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ref, worktree / "hosted" / "_reference.wav")
        player = ('<div class="ref-row"><span class="ref-label">Reference voice</span>'
                  '<audio controls preload="none" src="hosted/_reference.wav"></audio>'
                  '<span class="muted">full recording, consented by the speaker</span></div>')
    return ('<div class="meta">Each product clones the same consented voice through its own '
            'API. <strong>Sample lengths differ on purpose:</strong> each gets the longest cut it '
            'accepts, because some need more audio for a good clone. SIM = speaker similarity '
            'to the reference (WavLM, same as the local board), higher is closer.</div>' + player)


def build(worktree, head, foot):
    """Write <worktree>/hosted.html. `head` = page head + controls + logo (publish.py owns the
    shared style); `foot` = shared scripts + closing tags."""
    samples, scores = _samples(), _scores()
    days = sorted({r["ts"][:10] for r in samples})
    span = f"{days[0]} → {days[-1]}" if days else "sampling pending"
    rows = _row_data(samples, scores)
    tabs = [("board", "Board"), ("listen", "Listen"),
            ("cloning", f"Cloning ({len(_clone_slugs())})"), ("speed", "Speed detail")]
    tab_links = "".join(f'<a class="lens-tab{" active" if k == "board" else ""}" '
                        f'data-mode="{k}" href="#{k}">{v}</a>' for k, v in tabs)
    page = [
        head, STYLE, '<h1>Hosted APIs</h1>',
        f'<div class="reading-guide">{len(rows)} paid TTS APIs, each with its documented default '
        'voice, on the same prompts as the local board. <strong>Not ranked against the local '
        'models.</strong> <strong>First speech</strong> = ms until audible speech on a warm '
        f'connection, measured from Eastern US ({escape(span)}); hover it for p90 and call '
        'counts. <strong>UTMOS</strong> (higher is better) and <strong>WER</strong> (lower is '
        'better) use the same scorers as the Scores tab. Prices are list $/1M characters; '
        '* = hover for a note.</div>',
        f'<div class="mode-select"><div class="lens-tabs" id="mode-tabs">{tab_links}</div></div>',
        '<div class="subsection" data-mode="board"><h3 class="sub-head">Default voice '
        '<span class="muted">· speed, naturalness, intelligibility, price</span></h3>',
        _main_table(rows), '</div>',
        '<div class="subsection" data-mode="listen" style="display:none">',
        '<div class="meta">Every take of every prompt, nothing cherry-picked.</div>',
        _listen(worktree), '</div>',
        '<div class="subsection cloning" data-mode="cloning" style="display:none">',
        _clone_intro(worktree), _clone_table(scores),
        '<h3 class="sub-head">Listen: cloned voice</h3>', _listen(worktree, "cloning"), '</div>',
        '<div class="subsection" data-mode="speed" style="display:none">',
        '<div class="meta">Warm calls on one fixed English sentence, fastest first. '
        '<em>Delivery p3</em> = the full prompt-3 clip including vendor pacing (some stream at '
        'about playback speed), so don\'t rank by it. † = sweep calls only.</div>',
        _detail_grid(rows),
        '<h3 class="sub-head">HTTP vs WebSocket</h3>',
        '<div class="meta">Vendors offering both, sampled in alternating calls: median ms to '
        'first speech (calls/errors).</div>', _ab_table(samples), '</div>',
        '<div class="scores-foot">Each clip\'s request, model id, voice id and timings are in '
        'its JSON sidecar.</div>',
        '<script>window.__defaultSort = {colIdx: 1, dir: 1};</script>', TAB_SCRIPT, foot]
    (worktree / "hosted.html").write_text("\n".join(page), encoding="utf-8")
