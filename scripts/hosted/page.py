"""Build hosted.html — the Hosted (paid API) tab (docs/hosted.md). Called from publish.py's
build_top_level(); reads only hosted data, never the local board's:

    hosted/<slug>/default/p<N>_t<take>.{wav,json}   clips + sidecars (the sweep)
    hosted/speed/samples.jsonl                      multi-day speed sampler (speed_sample.py)
    scoring/hosted_scores.csv                       UTMOS / WER / Health (score_hosted.py, Linux)

Clips are copied to <worktree>/hosted/<slug>/p<N>_t<take>.wav (all takes: no cherry-picking).
"""

import csv
import json
import shutil
import statistics as st
from collections import defaultdict
from html import escape
from pathlib import Path

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
}

# Per-vendor transport note for the A/B table.
TRANSPORT_NOTE = {
    "cartesia_sonic36": "Socket clearly faster to first audio; board uses it.",
    "assemblyai_tts1": "Both fast; socket a little faster; board uses it.",
    "deepgram_flux_tts": "Socket sends ~0.6 s of silence before speech, so first *speech* is "
                         "slower than HTTP; board uses HTTP.",
    "xai_tts": "Close; HTTP kept.",
    "inworld_tts2": "Close; HTTP kept.",
    "soniox_tts2": "Socket somewhat faster in sampling; both pace audio near playback speed.",
    "smallest_lightning31pro": "Close on first audio; socket's 'complete' message lags ~4 s "
                               "(total measured to last audio byte).",
    "murf_falcon21": "Socket somewhat faster in sampling.",
    "stepaudio3_tts": "Socket session drops or stalls often; HTTP kept.",
}


STYLE = (
    "<style>.hgrid{display:grid;gap:.35rem .8rem;align-items:center;margin:.6rem 0 1.2rem;"
    "font-size:.92em}.hgrid.ab{grid-template-columns:minmax(9rem,1.3fr) repeat(5,auto) 2fr}"
    ".hgrid.listen{grid-template-columns:minmax(8rem,1fr) repeat(3,minmax(0,1.4fr))}"
    ".hgrid.listen audio{width:100%;height:32px}.hgrid .wide{grid-column:span 3}"
    "@media(max-width:760px){.hgrid.ab{grid-template-columns:1fr 1fr}"
    ".hgrid.ab>div:nth-child(7n){grid-column:span 2}"
    ".hgrid.listen{grid-template-columns:1fr}.hgrid .wide{grid-column:auto}}</style>")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _med(xs):
    return st.median(xs) if xs else None


def _p90(xs):
    if len(xs) < 2:
        return None
    return st.quantiles(xs, n=10)[-1]


def _ms(v):
    return "—" if v is None else f"{v:,.0f}"


def _ds(v):
    return f' data-sort="{v:.4f}"' if v is not None else ' data-sort=""'


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
    return [r for r in rows if r.get("call", 0) > 0]  # call 0 = failed warmup, counted below


def _scores():
    by = defaultdict(list)
    if SCORES.exists():
        with open(SCORES, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                by[r["model"]].append(r)
    return by


def _speed(samples, slug, tr):
    rs = [r for r in samples if r["slug"] == slug and r["transport"] == tr]
    ok = [r["ttfa_speech_ms"] for r in rs if not r["error"] and r["ttfa_speech_ms"] is not None]
    days = {r["ts"][:10] for r in rs}
    return dict(med=_med(ok), p90=_p90(ok), n=len(ok), err=len(rs) - len(ok), days=len(days))


def _health_td(rs):
    flags = sorted({(r["prompt_id"], f) for r in rs for f in r["health"].split(";") if f})
    if not flags:
        return '<td class="num muted" data-sort="0">✓</td>'
    label = "; ".join(f"{f} (p{p})" for p, f in flags)
    return (f'<td class="num" data-sort="{len(flags)}"><span class="health-flag" '
            f'title="mechanical defect in a clip — listen before trusting">⚠ {escape(label)}'
            f'</span></td>')


def _main_table(samples, scores):
    # Plain-text headers: the shared sort script rewrites th.textContent (drops markup).
    cols = ["Model", "Transport", "First speech ms (median)", "p90", "Calls", "Errors",
            "Delivery p3 s (incl. pacing)", "UTMOS en", "WER en", "WER fr/es", "Health",
            "Langs", "SR", "$/1M chars"]
    head = "<thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead>"
    body = []
    for slug, cls in ADAPTERS.items():
        if slug in SKIP:
            continue
        name, price, pnote = CATALOG.get(slug, (slug, "?", ""))
        ad_tr = transport(cls)
        sp = _speed(samples, slug, ad_tr)
        cars = [c for c in _sidecars(slug) if c.get("status") == "ok"]
        if not sp["n"]:  # no sampler data yet: warm sweep calls, marked
            xs = [c["ttfa_speech_ms"] for c in cars
                  if c.get("connection") == "warm" and c.get("ttfa_speech_ms") is not None]
            sp = dict(med=_med(xs), p90=_p90(xs), n=len(xs), err=0, days=0, sweep=True)
        p3 = [c["total_ms"] / 1000 for c in cars if c["prompt_id"] == 3 and c.get("total_ms")]
        rs = scores.get(slug, [])
        en = [r for r in rs if r["prompt_id"] in ("1", "2", "3", "4")]
        xx = [r for r in rs if r["prompt_id"] in ("5", "6")]
        um = _med([v for v in (_f(r["utmos"]) for r in en) if v is not None])
        we = [v for v in (_f(r["wer"]) for r in en) if v is not None]
        wx = [v for v in (_f(r["wer"]) for r in xx) if v is not None]
        we, wx = (st.mean(we) if we else None), (st.mean(wx) if wx else None)
        srs = {c.get("sample_rate") for c in cars if c.get("sample_rate")}
        sr = max(srs) if srs else None
        route = " · via OpenRouter" if cls.route == "openrouter" else ""
        model_id = escape(cls.model)
        mark = ' <span class="muted" title="sweep calls; multi-day sampling pending">†</span>' \
            if sp.get("sweep") else ""
        body.append(
            f'<tr><td><strong>{escape(name)}</strong><br><span class="muted"><code>{model_id}'
            f'</code>{route}</span></td><td>{escape(ad_tr)}</td>'
            f'<td class="num"{_ds(sp["med"])}>{_ms(sp["med"])}{mark}</td>'
            f'<td class="num"{_ds(sp["p90"])}>{_ms(sp["p90"])}</td>'
            f'<td class="num" data-sort="{sp["n"]}">{sp["n"]}</td>'
            f'<td class="num" data-sort="{sp["err"]}">{sp["err"] or "—"}</td>'
            f'<td class="num"{_ds(_med(p3))}>'
            f'{"—" if not p3 else f"{_med(p3):.1f}"}</td>'
            f'<td class="num"{_ds(um)}>{"—" if um is None else f"{um:.2f}"}</td>'
            f'<td class="num"{_ds(we)}>{"—" if we is None else f"{we:.3f}"}</td>'
            f'<td class="num"{_ds(wx)}>{"—" if wx is None else f"{wx:.3f}"}</td>'
            f'{_health_td(rs)}'
            f'<td style="white-space:nowrap">{escape(" ".join(sorted(cls.langs)))}</td>'
            f'<td class="num"{_ds(sr)}>{"—" if not sr else f"{sr / 1000:g}k"}</td>'
            f'<td title="{escape(pnote)}">{escape(price)}'
            f'{"" if not pnote else f"<br><span class=muted>{escape(pnote)}</span>"}</td></tr>')
    return (f'<div style="overflow-x:auto"><table class="sortable">{head}<tbody>'
            f'{"".join(body)}</tbody></table></div>')


def _ab_table(samples):
    # Grids, not <table>s: the shared sort script re-sorts every table on the page together.
    cells = ["<b>Vendor</b>", "<b>HTTP ms</b>", "<b>calls · err</b>", "<b>WebSocket ms</b>",
             "<b>calls · err</b>", "<b>Board uses</b>", "<b>Note</b>"]
    for slug in BOTH:
        a, b = _speed(samples, slug, "http"), _speed(samples, slug, "websocket")
        cells += [escape(CATALOG[slug][0]), _ms(a["med"]), f'{a["n"]} · {a["err"]}',
                  _ms(b["med"]), f'{b["n"]} · {b["err"]}', escape(transport(ADAPTERS[slug])),
                  f'<span class="muted">{escape(TRANSPORT_NOTE.get(slug, ""))}</span>']
    return '<div class="hgrid ab">' + "".join(f"<div>{c}</div>" for c in cells) + "</div>"


def _listen(worktree):
    dest = worktree / "hosted"
    out = []
    for pid, lang, text in PROMPTS:
        rows = []
        for slug, cls in ADAPTERS.items():
            if slug in SKIP:
                continue
            name = CATALOG.get(slug, (slug,))[0]
            if lang not in cls.langs:
                rows.append(f'<div>{escape(name)}</div><div class="muted wide">'
                            f'not offered in {escape(lang)}</div>')
                continue
            cells = []
            for take in (1, 2, 3):
                src = OUT / slug / "default" / f"p{pid}_t{take}.wav"
                if not src.exists():
                    cells.append('<div class="muted">missing</div>')
                    continue
                d = dest / slug
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, d / src.name)
                cells.append(f'<div><audio controls preload="none" '
                             f'src="hosted/{slug}/{src.name}"></audio></div>')
            rows.append(f'<div>{escape(name)}</div>{"".join(cells)}')
        out.append(f'<details class="panel" id="p{pid}"><summary>Prompt {pid} · {escape(lang)} '
                   f'<span class="muted">— {escape(text[:90])}{"…" if len(text) > 90 else ""}'
                   f'</span></summary><div class="hgrid listen"><div><b>Model</b></div><div><b>Take 1</b>'
                   f'</div><div><b>Take 2</b></div><div><b>Take 3</b></div>{"".join(rows)}'
                   f'</div></details>')
    return "".join(out)


def build(worktree, head, foot):
    """Write <worktree>/hosted.html. `head` = page head + controls + logo (publish.py owns the
    shared style); `foot` = shared scripts + closing tags."""
    samples, scores = _samples(), _scores()
    days = sorted({r["ts"][:10] for r in samples})
    span = f"{days[0]} → {days[-1]}" if days else "sampling pending"
    n_models = sum(1 for s in ADAPTERS if s not in SKIP)
    page = [head, STYLE, '<h1>Hosted APIs</h1>',
            '<div class="meta">Paid TTS APIs as a buyer gets them: each vendor\'s documented '
            'default voice, the same 6 prompts as the local board, 3 takes each, all shown. '
            f'{n_models} models. <strong>Never ranked against the local models</strong> — '
            'different question (products vs. models on known hardware).</div>',
            '<h2>Board</h2>',
            f'<div class="meta">Speed <strong>measured from Eastern US</strong> ({escape(span)}), '
            'warm connection, one fixed English sentence. <em>First speech</em> = first sample '
            'above −40 dBFS, so leading silence doesn\'t count; that is the speed ranking. '
            '<em>Delivery</em> includes vendor pacing (some stream at about playback speed) — '
            'don\'t rank by it. Prices are list $/1M characters; check the vendor page before '
            'buying.</div>',
            _main_table(samples, scores),
            '<h2>HTTP vs WebSocket</h2>',
            '<div class="meta">Vendors offering both, sampled on both in alternating calls. '
            'Median time to first speech.</div>',
            _ab_table(samples),
            '<h2>Listen</h2>', _listen(worktree),
            '<div class="scores-foot">UTMOS <code>utmos22_strong</code>, WER Whisper-large-v3, '
            'same scorers as the local Scores tab. Every clip\'s request, model id sent, voice id '
            'and timings are in its JSON sidecar in the repo.</div>',
            '<script>window.__defaultSort = {colIdx: 2, dir: 1};</script>', foot]
    (worktree / "hosted.html").write_text("\n".join(page), encoding="utf-8")
