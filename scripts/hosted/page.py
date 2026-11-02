"""Build hosted.html — the Hosted (paid API) tab (docs/hosted.md). Called from publish.py's
build_top_level(); reads only hosted data, never the local board's:

    hosted/<slug>/default/p<N>_t<take>.{wav,json}   clips + sidecars (the sweep)
    hosted/<slug>/cloning/p<N>_t<take>.{wav,json}   cloned-voice clips (the cloning lens)
    hosted/speed/samples.jsonl                      multi-day speed sampler (speed_sample.py)
    scoring/hosted_scores.csv                       UTMOS / WER / SIM / Health (score_hosted.py, Linux)

Layout follows the Scores page: one short reading guide, then tabs (Default voice / Cloning /
Speed detail). Each lens tab is the Scores pattern (bar chart, sortable table) followed by its
clips. The shared sort script sorts every table on a page by one column index, so both tables
keep their lead metric in column 1 (UTMOS / SIM); the speed sections are CSS grids.

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

from .catalog import CATALOG, score_aggregates
from .core import OUT
from .providers import ADAPTERS
from .speed_sample import BOTH, SKIP, transport

REPO = OUT.parent
SCORES = REPO / "scoring" / "hosted_scores.csv"
SAMPLES = OUT / "speed" / "samples.jsonl"


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
 .hgrid.ab,.hgrid.det{grid-template-columns:1fr 1fr}
 .hgrid>div.h{display:none}
 .hgrid>div.name{grid-column:1/-1;margin-top:.6rem;font-weight:600}
 .hgrid.ab>div.note{grid-column:1/-1}
 .hgrid.listen{grid-template-columns:1fr}
 .hgrid .wide{grid-column:auto}
}
</style>"""

TAB_SCRIPT = """<script>
(function(){
  // Only one clip plays at a time, same as the Listen page. 'play' doesn't bubble, so capture it.
  document.addEventListener('play', function(e){
    document.querySelectorAll('audio').forEach(function(a){ if(a !== e.target) a.pause(); });
  }, true);
  var tabs = document.querySelectorAll('#mode-tabs .lens-tab');
  var subs = document.querySelectorAll('.subsection[data-mode]');
  function show(m){
    document.querySelectorAll('audio').forEach(function(a){ a.pause(); });
    subs.forEach(function(s){ s.style.display = s.dataset.mode === m ? '' : 'none'; });
    tabs.forEach(function(t){ t.classList.toggle('active', t.dataset.mode === m); });
    try { history.replaceState(null, '', '#' + m); } catch (e) {}
  }
  tabs.forEach(function(t){
    t.addEventListener('click', function(e){ e.preventDefault(); show(t.dataset.mode); });
  });
  var h = location.hash.slice(1);
  show(['cloning', 'speed'].indexOf(h) >= 0 ? h : 'default');  // old #board/#listen too
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
    """{"default"|"cloning": {slug: aggregate}}: means over every clip, as on the Scores page."""
    return {m: score_aggregates(SCORES, m) for m in ("default", "cloning")}


def _health_td(flags):
    """Same Health cell as the Scores page: a muted check, or a flag naming the prompts."""
    if not flags:
        return '<td class="num muted" data-sort="0">✓</td>'
    by = defaultdict(set)
    for pid, f in flags:
        by[f].add(pid)
    label = "; ".join(f'{f} ({",".join(sorted(p))})' for f, p in sorted(by.items()))
    return (f'<td class="num" data-sort="{len(flags)}"><span class="health-flag" title="mechanical '
            f'defect in a clip (long internal silence / clipping / dead audio) — listen before '
            f'trusting">⚠ {escape(label)}</span></td>')


def _metric_td(key, v, spec=".3f"):
    if v is None:
        return f'<td class="num muted" data-metric="{key}" data-sort="">—</td>'
    return f'<td class="num" data-metric="{key}" data-sort="{v:.4f}">{format(v, spec)}</td>'


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
        srs = {c.get("sample_rate") for c in cars if c.get("sample_rate")}
        sc = scores["default"].get(slug, {})
        out.append(dict(slug=slug, cls=cls, name=name, price=price, pnote=pnote, tr=tr, sp=sp,
                        p3=_med(p3), sr=max(srs) if srs else None, um=sc.get("utmos"),
                        we=sc.get("wer"), flags=sc.get("flags", [])))
    return out


def _main_table(rows):
    # Same columns, cells and definitions as the Scores page's table (metric first, Health
    # triage), plus what only the hosted board has: speed and price. Plain-text headers: the
    # shared sort script rewrites th.textContent.
    cols = ["Model", "UTMOS ↑", "WER ↓", "Health", "First speech ms ↓", "$ / 1M chars", "Langs"]
    head = "<thead><tr>" + "".join(
        f'<th{" class=num" if 0 < i < 6 else ""}>{escape(c)}</th>'
        for i, c in enumerate(cols)) + "</tr></thead>"
    body = []
    for d in rows:
        sp, cls = d["sp"], d["cls"]
        mark = (' <span class="muted" title="from the sweep; multi-day sampling pending">†</span>'
                if sp.get("sweep") else "")
        tip = f'p90 {_ms(sp["p90"])} ms · {sp["n"]} calls · {sp["err"]} errors · {d["tr"]}'
        via = " · via OpenRouter" if cls.route == "openrouter" else ""
        note = "*" if d["pnote"] else ""
        ptitle = f' title="{escape(d["pnote"])}"' if d["pnote"] else ""
        ms = sp["med"]
        body.append(
            f'<tr><td title="model id: {escape(cls.model)}{via}">{escape(d["name"])}</td>'
            + _metric_td("utmos", d["um"]) + _metric_td("wer", d["we"]) + _health_td(d["flags"])
            + f'<td class="num" data-metric="ttfa"{_ds(ms)} title="{escape(tip)}">{_ms(ms)}{mark}</td>'
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
    """Same layout as the default table: metrics first (SIM, UTMOS, WER, Health), then the
    sample each vendor was sent against its documented minimum."""
    from .providers.clone import MIN_REC
    reg = _clone_slugs()
    cols = ["Model", "SIM ↑", "UTMOS ↑", "WER ↓", "Health", "Sample sent", "Vendor min",
            "Recommended", "Clone langs"]
    head = "<thead><tr>" + "".join(
        f'<th{" class=num" if 0 < i < 8 else ""}>{escape(c)}</th>'
        for i, c in enumerate(cols)) + "</tr></thead>"
    sec = lambda v: f"{v} s" if v != "—" else "—"
    body = []
    for slug, info in reg.items():
        cls = ADAPTERS[slug]
        sc = scores["cloning"].get(slug, {})
        mn, rec = MIN_REC.get(slug, ("—", "—"))
        body.append(
            f'<tr><td>{escape(CATALOG[slug][0])}</td>' + _metric_td("sim", sc.get("sim"))
            + _metric_td("utmos", sc.get("utmos")) + _metric_td("wer", sc.get("wer"))
            + _health_td(sc.get("flags", []))
            + f'<td class="num" data-sort="{info["seconds"]}">{info["seconds"]:g} s</td>'
            f'<td class="num muted">{sec(mn)}</td><td class="num muted">{sec(rec)}</td>'
            f'<td class="muted">{escape(" ".join(sorted(cls.clone_langs or cls.langs)))}</td></tr>')
    table = (f'<div class="scores-table-scroll"><table class="hosted-board">{head}<tbody>'
             f'{"".join(body)}</tbody></table></div>')
    missing = [s for s in ADAPTERS if s not in reg and s not in SKIP]
    why = "".join(f"<li>{escape(CATALOG[s][0])}: {escape(NO_CLONE.get(s, 'pending'))}</li>"
                  for s in missing)
    if not why:
        return table
    return (table + f'<details class="panel"><summary>Not in the cloning lens <span '
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
    from publish import score_chart  # publish.py imports this module lazily: no cycle at load
    tabs = [("default", "Default voice"), ("cloning", f"Cloning ({len(_clone_slugs())})"),
            ("speed", "Speed detail")]
    tab_links = "".join(f'<a class="lens-tab{" active" if k == "default" else ""}" '
                        f'data-mode="{k}" href="#{k}">{v}</a>' for k, v in tabs)
    page = [
        head, STYLE, '<h1>Hosted APIs</h1>',
        f'<div class="reading-guide">{len(rows)} paid TTS APIs, each with its documented default '
        'voice, on the same prompts as the local board. Kept off the local board by default; '
        'the Scores page can show them side by side (default voice only). <strong>First speech</strong> = ms until audible speech on a warm '
        f'connection, measured from Eastern US ({escape(span)}); hover it for p90 and call '
        'counts. <strong>UTMOS</strong>, <strong>WER</strong>, <strong>SIM</strong> and '
        '<strong>Health</strong> mean the same as on the <a href="scores.html">Scores</a> page: the '
        'mean over every clip (3 takes of each prompt). Prices are list $/1M characters; '
        '* = hover for a note.</div>',
        f'<div class="mode-select"><div class="lens-tabs" id="mode-tabs">{tab_links}</div></div>',
        '<div class="subsection" data-mode="default"><h3 class="sub-head">Default voice '
        '<span class="muted">· naturalness, intelligibility, speed, price</span></h3>',
        score_chart([("utmos", "UTMOS ↑"), ("wer", "WER ↓", {"dir": "asc"}),
                     ("ttfa", "First speech ↓", {"dir": "asc", "decimals": 0, "unit": " ms"})]),
        _main_table(rows),
        '<h3 class="sub-head">Listen: default voice</h3>',
        '<div class="meta">Every take of every prompt, nothing cherry-picked.</div>',
        _listen(worktree), '</div>',
        '<div class="subsection cloning" data-mode="cloning" style="display:none">',
        '<h3 class="sub-head">Cloning <span class="muted">· fidelity, naturalness, '
        'intelligibility</span></h3>', _clone_intro(worktree),
        score_chart([("sim", "SIM ↑"), ("utmos", "UTMOS ↑")]), _clone_table(scores),
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
        # colIdx 1 = the lead metric in both tables (UTMOS / SIM), as on Scores.
        '<script>window.__defaultSort = {colIdx: 1, dir: -1};</script>', TAB_SCRIPT, foot]
    (worktree / "hosted.html").write_text("\n".join(page), encoding="utf-8")
