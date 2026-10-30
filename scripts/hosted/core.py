"""Shared harness for hosted-API collection (see docs/hosted.md).

An adapter describes one request and the audio it returns; this module sends it, times it and
saves the clip plus two sidecars:

- public  hosted/<model>/<lens>/p<N>_t<take>.json: provider, model id sent, voice, non-default
  params, text/audio hashes, format, timings. No request body, no headers, no reference audio.
- private hosted/_private/<model>/<lens>/p<N>_t<take>.json: full request body (minus any audio)
  and response headers, for debugging. Never published.

Timing contract (same for every provider):
- One httpx.Client per provider run. The first call is `cold` (pays DNS/TCP/TLS), later calls
  reuse the connection and are `warm`. Each sidecar says which.
- All times are ms from the moment the request is handed to the client.
- ttfa_ms: first chunk that carries audio bytes (iter_raw, no read buffer). Only meaningful when
  the adapter is a streaming endpoint; non-streaming adapters record ttfa_ms = null.
- Phases (tcp incl. DNS, tls, request sent, headers received) come from httpcore's trace hook
  and go in the private sidecar.
- Measured from Eastern US (Tym's home connection). No other location claims.
"""

import hashlib
import io
import json
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import httpx

from . import keys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "hosted"
PRIVATE = OUT / "_private"


class Adapter:
    """Subclass per provider/model. Fill the class attributes and request()."""

    slug = ""          # row id, e.g. "eleven_v4"
    provider = ""      # key field in 1Password, e.g. "elevenlabs"
    vendor = ""        # display, e.g. "ElevenLabs"
    model = ""         # model id sent to the API
    route = "direct"   # or "openrouter"
    langs = {"en"}     # languages the vendor documents for this model
    streaming = True   # True = progressive audio, ttfa is meaningful
    fmt = "pcm16"      # "pcm16" (headerless, needs sr) | "wav" | "mp3"
    sr = 24000
    params = {}        # non-default parameters we send, recorded in the public sidecar

    def __init__(self, key):
        self.key = key

    def voice(self, lang):
        """Voice id for the default lens. Rule (docs/hosted.md): the voice in the vendor's own
        TTS quickstart; if it can't speak `lang`, the first voice the vendor's catalog lists
        for that language."""
        raise NotImplementedError

    def request(self, text, lang, voice):
        """Return dict(method, url, headers, json|content, params)."""
        raise NotImplementedError

    def audio_chunks(self, raw_iter):
        """Turn the raw response body into audio bytes. Default: passthrough."""
        yield from raw_iter

    def echo(self, headers, body_meta):
        """Vendor-reported identity/timing worth keeping in the public sidecar."""
        keep = ("request-id", "x-request-id", "server-timing", "x-model", "openai-model",
                "x-processing-ms", "history-item-id", "character-cost", "x-character-count")
        return {k: v for k, v in headers.items() if k.lower() in keep}


def _trace(marks, t0):
    def cb(event, info):
        if event.endswith(".complete") or event.endswith(".started"):
            marks.setdefault(event, round((time.perf_counter() - t0) * 1000, 1))
    return cb


def to_wav(data, fmt, sr):
    """Return (wav_bytes, sr, duration_s). pcm16 gets a header; wav/mp3 are decoded to check."""
    if fmt == "pcm16":
        if len(data) % 2:
            data = data[:-1]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(data)
        return buf.getvalue(), sr, len(data) / 2 / sr
    import soundfile as sf
    audio, rate = sf.read(io.BytesIO(data), dtype="int16")
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue(), rate, len(audio) / rate


def synth(client, ad, text, lang):
    voice = ad.voice(lang)
    req = ad.request(text, lang, voice)
    marks = {}
    t0 = time.perf_counter()
    ttfa = None
    chunks = []
    with client.stream(req["method"], req["url"], headers=req.get("headers"),
                       params=req.get("params"), json=req.get("json"),
                       content=req.get("content"),
                       extensions={"trace": _trace(marks, t0)}) as r:
        t_headers = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            body = r.read()[:600]
            raise RuntimeError(f"{ad.slug}: HTTP {r.status_code}: {body!r}")
        for chunk in ad.audio_chunks(r.iter_raw()):
            if chunk:
                if ttfa is None:
                    ttfa = (time.perf_counter() - t0) * 1000
                chunks.append(chunk)
        total = (time.perf_counter() - t0) * 1000
        headers = dict(r.headers)
        http_version = r.http_version
    data = b"".join(chunks)
    if not data:
        raise RuntimeError(f"{ad.slug}: empty audio")
    return dict(voice=voice, req=req, data=data, marks=marks, headers=headers,
                http_version=http_version, ttfb_ms=round(t_headers, 1),
                ttfa_ms=round(ttfa, 1) if ad.streaming and ttfa else None,
                total_ms=round(total, 1))


def run_cell(client, ad, lens, pid, lang, text, take, cold):
    name = f"p{pid}_t{take}"
    pub_dir = OUT / ad.slug / lens
    priv_dir = PRIVATE / ad.slug / lens
    pub_dir.mkdir(parents=True, exist_ok=True)
    priv_dir.mkdir(parents=True, exist_ok=True)
    base = {"model_row": ad.slug, "vendor": ad.vendor, "model_id_sent": ad.model,
            "route": ad.route, "lens": lens, "prompt_id": pid, "lang": lang, "take": take,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "client_location": "Eastern US"}
    if lang not in ad.langs:
        base["status"] = "not_supported"
        (pub_dir / f"{name}.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
        print(f"{ad.slug:>18} {lens:<8} p{pid} t{take}  not supported ({lang})")
        return base
    try:
        res = synth(client, ad, text, lang)
    except Exception as e:  # published as a defect/error, never dropped
        base.update(status="error", error=str(e)[:400])
        (pub_dir / f"{name}.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
        print(f"{ad.slug:>18} {lens:<8} p{pid} t{take}  ERROR {str(e)[:160]}")
        return base
    wav, sr, dur = to_wav(res["data"], ad.fmt, ad.sr)
    (pub_dir / f"{name}.wav").write_bytes(wav)
    total_s = res["total_ms"] / 1000
    base.update(
        status="ok", voice_id=res["voice"], params=ad.params, streaming=ad.streaming,
        format_received=ad.fmt, sample_rate=sr, duration_s=round(dur, 3),
        audio_sha256=hashlib.sha256(wav).hexdigest()[:16],
        connection="cold" if cold else "warm", http_version=res["http_version"],
        ttfa_ms=res["ttfa_ms"], ttfb_ms=res["ttfb_ms"], total_ms=res["total_ms"],
        rtfx=round(dur / total_s, 2) if total_s else None,
        vendor_echo=ad.echo(res["headers"], None),
        key_fingerprint=keys.fingerprint(ad.key))
    (pub_dir / f"{name}.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    req = {k: v for k, v in res["req"].items() if k not in ("headers", "content")}
    priv = dict(base, request=req, response_headers=res["headers"], phases_ms=res["marks"])
    (priv_dir / f"{name}.json").write_text(json.dumps(priv, indent=2), encoding="utf-8")
    ttfa = f"{res['ttfa_ms']:.0f}" if res["ttfa_ms"] else "-"
    print(f"{ad.slug:>18} {lens:<8} p{pid} t{take}  {dur:6.2f}s {sr}Hz  "
          f"ttfa {ttfa:>5}ms total {res['total_ms']:6.0f}ms  {base['connection']}")
    return base


def new_client():
    return httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0), http2=False)
