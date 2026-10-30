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

    def on_headers(self, headers):
        """Hook for vendors that report the sample rate in a response header."""

    def model_sent(self, lang):
        """Model id actually sent for this language (Deepgram encodes voice+lang in it)."""
        return self.model

    def echo(self, headers, body_meta):
        """Vendor-reported identity/timing worth keeping in the public sidecar."""
        keep = ("request-id", "x-request-id", "server-timing", "x-model", "openai-model",
                "x-processing-ms", "history-item-id", "character-cost", "x-character-count")
        return {k: v for k, v in headers.items() if k.lower() in keep}


def json_lines(raw_iter):
    """Yield parsed JSON objects from an SSE (`data: {...}`) or NDJSON byte stream."""
    buf = b""
    for chunk in raw_iter:
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if line.startswith(b"data:"):
                line = line[5:].strip()
            if not line or line == b"[DONE]" or line.startswith((b"event:", b"id:", b":")):
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue
    if buf.strip():
        line = buf.strip()
        if line.startswith(b"data:"):
            line = line[5:].strip()
        try:
            yield json.loads(line)
        except ValueError:
            pass


def _trace(marks, t0):
    def cb(event, info):
        if event.endswith(".complete") or event.endswith(".started"):
            marks.setdefault(event, round((time.perf_counter() - t0) * 1000, 1))
    return cb


def to_wav(data, fmt, sr):
    """Return (wav_bytes, sr, duration_s). pcm16 gets a header; wav/mp3 are decoded to check."""
    if fmt == "pcm_f32":
        import numpy as np
        f = np.frombuffer(data[:len(data) // 4 * 4], dtype="<f4")
        data = (np.clip(f, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        fmt = "pcm16"
    if fmt == "pcm16" and data[:4] == b"RIFF":  # vendor wrapped it in WAV after all
        fmt = "wav"
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


SPEECH_THRESHOLD = 0.01  # |sample| as a fraction of full scale (-40 dBFS)


def _first_speech_ms(data, fmt, arrivals):
    """Arrival time of the chunk holding the first sample above SPEECH_THRESHOLD, so a vendor
    that streams leading silence early doesn't get credit for it. None for mp3 (not decodable
    per byte offset)."""
    import numpy as np
    head = 0
    if fmt == "pcm16" and data[:4] == b"RIFF":  # e.g. Deepgram wraps linear16 in WAV
        i = data.find(b"data", 12)
        head = i + 8 if i > 0 else 44
    if fmt == "pcm16":
        body = data[head:]
        a, width = np.frombuffer(body[:len(body) // 2 * 2], dtype="<i2") / 32768.0, 2
    elif fmt == "pcm_f32":
        a, width = np.frombuffer(data[:len(data) // 4 * 4], dtype="<f4"), 4
    else:
        return None
    hits = np.flatnonzero(np.abs(a) > SPEECH_THRESHOLD)
    if not hits.size:
        return None
    off = head + int(hits[0]) * width
    for end, ms in arrivals:
        if off < end:
            return round(ms, 1)
    return None


def synth(client, ad, text, lang):
    if hasattr(ad, "synth_ws"):  # WebSocket-only vendors time their own session
        return ad.synth_ws(text, lang)
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
        ctype = r.headers.get("content-type", "")
        if type(ad).audio_chunks is Adapter.audio_chunks and ("json" in ctype or "text/" in ctype):
            # A raw-audio adapter got a JSON/text body with HTTP 200: an error envelope, not audio.
            raise RuntimeError(f"{ad.slug}: expected audio, got {ctype}: {r.read()[:300]!r}")
        ad.on_headers(r.headers)
        # iter_bytes = as-received chunks with Content-Encoding (gzip on Gemini's SSE) decoded;
        # iter_raw would hand back compressed bytes. Neither applies a fixed read size.
        arrivals = []  # (byte offset where the chunk ends, arrival ms)
        nbytes = 0
        for chunk in ad.audio_chunks(r.iter_bytes()):
            if chunk:
                now = (time.perf_counter() - t0) * 1000
                if ttfa is None:
                    ttfa = now
                chunks.append(chunk)
                nbytes += len(chunk)
                arrivals.append((nbytes, now))
        total = (time.perf_counter() - t0) * 1000
        headers = dict(r.headers)
        http_version = r.http_version
    data = b"".join(chunks)
    if not data:
        raise RuntimeError(f"{ad.slug}: empty audio")
    # Fresh connection iff httpcore opened a TCP connection for this request.
    cold = any(k.startswith("connection.connect_tcp") for k in marks)
    ttfa_speech = _first_speech_ms(data, ad.fmt, arrivals) if ad.streaming else None
    return dict(voice=voice, req=req, data=data, marks=marks, headers=headers, cold=cold,
                ttfa_speech_ms=ttfa_speech,
                http_version=http_version, ttfb_ms=round(t_headers, 1),
                ttfa_ms=round(ttfa, 1) if ad.streaming and ttfa else None,
                total_ms=round(total, 1))


def warmup(client, ad, text="Hello."):
    """Untimed, unsaved call so the sweep's timed calls all run on a warm connection."""
    try:
        synth(client, ad, text, "en")
    except Exception as e:
        print(f"{ad.slug:>18} warmup failed: {str(e)[:160]}")


def run_cell(client, ad, lens, pid, lang, text, take):
    name = f"p{pid}_t{take}"
    pub_dir = OUT / ad.slug / lens
    priv_dir = PRIVATE / ad.slug / lens
    pub_dir.mkdir(parents=True, exist_ok=True)
    priv_dir.mkdir(parents=True, exist_ok=True)
    for stale in (pub_dir / f"{name}.wav", priv_dir / f"{name}.json"):
        stale.unlink(missing_ok=True)
    base = {"model_row": ad.slug, "vendor": ad.vendor, "model_id_sent": ad.model_sent(lang),
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
        wav, sr, dur = to_wav(res["data"], ad.fmt, ad.sr)
        if dur < 0.05:
            raise RuntimeError(f"{ad.slug}: {dur:.3f}s of audio")
    except Exception as e:  # published as a defect/error, never dropped
        base.update(status="error", error=str(e)[:400])
        (pub_dir / f"{name}.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
        print(f"{ad.slug:>18} {lens:<8} p{pid} t{take}  ERROR {str(e)[:160]}")
        return base
    (pub_dir / f"{name}.wav").write_bytes(wav)
    total_s = res["total_ms"] / 1000
    base.update(
        status="ok", voice_id=res["voice"], params=ad.params, streaming=ad.streaming,
        format_received=ad.fmt, sample_rate=sr, duration_s=round(dur, 3),
        audio_sha256=hashlib.sha256(wav).hexdigest()[:16],
        connection="cold" if res["cold"] else "warm", http_version=res["http_version"],
        ttfa_ms=res["ttfa_ms"], ttfa_speech_ms=res["ttfa_speech_ms"],
        ttfb_ms=res["ttfb_ms"], total_ms=res["total_ms"],
        rtfx=round(dur / total_s, 2) if total_s else None,
        vendor_echo=ad.echo(res["headers"], None),
        key_fingerprint=keys.fingerprint(ad.key))
    (pub_dir / f"{name}.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    req = {k: v for k, v in res["req"].items() if k not in ("headers", "content")}
    if req.get("params"):
        req["params"] = {k: v for k, v in req["params"].items() if "key" not in k.lower()}
    priv = dict(base, request=req, response_headers=res["headers"], phases_ms=res["marks"])
    (priv_dir / f"{name}.json").write_text(json.dumps(priv, indent=2), encoding="utf-8")
    ttfa = f"{res['ttfa_ms']:.0f}" if res["ttfa_ms"] else "-"
    sp = f"{res['ttfa_speech_ms']:.0f}" if res["ttfa_speech_ms"] else "-"
    print(f"{ad.slug:>18} {lens:<8} p{pid} t{take}  {dur:6.2f}s {sr}Hz  "
          f"ttfa {ttfa:>5}ms speech {sp:>5}ms total {res['total_ms']:6.0f}ms  "
          f"{base['connection']}")
    return base


def new_client():
    # identity: no compressed bodies, so the first chunk is never held back by a decoder.
    return httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0), http2=False,
                        headers={"Accept-Encoding": "identity"})
