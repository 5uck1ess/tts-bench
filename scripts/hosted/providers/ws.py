"""WebSocket transports for vendors where a warm socket beat warm HTTP streaming on time to first
audio. Measured 2026-10-30, same sentence, 10 alternating calls each (median first-audio ms,
HTTP -> WS): Cartesia 251 -> 133, AssemblyAI 57 -> 35; the 3-take sweeps agree on first *speech*
(Cartesia 143 -> 128, AssemblyAI 65 -> 47). Deepgram's socket sends ~0.6 s of silence before
speech (first speech ~800 ms vs ~200 ms over HTTP) and xAI tied in the sweep, so both stay on
HTTP, as do Inworld, Soniox, Murf and Smallest (ties or HTTP wins). The Deepgram/xAI socket
classes are kept for re-measuring. Same model, voice and output format as the HTTP adapters they
extend; only the transport differs."""

import base64
import json
import time
import uuid
from urllib.parse import urlencode

from websockets.sync.client import connect

from ..core import _first_speech_ms
from .batch_a import CartesiaSonic36
from .batch_b import AssemblyTTS1, DeepgramFluxTTS, XaiTTS


class WsMixin:
    """One persistent socket per language (some vendors fix voice/language at connect time).
    Timing starts when the text is sent; connect time is recorded separately on cold calls."""
    route = "websocket"
    _socks = None

    def ws_open(self, lang):  # -> connected socket, ready for a request
        raise NotImplementedError

    def ws_send(self, ws, text, lang, voice):
        raise NotImplementedError

    def ws_recv(self, ws):  # yields audio bytes until the end-of-utterance message
        raise NotImplementedError

    def _j(self, ws, timeout=60):
        m = ws.recv(timeout=timeout)
        return m if isinstance(m, bytes) else json.loads(m)

    def synth_ws(self, text, lang):
        if self._socks is None:
            self._socks = {}
        cold, marks = lang not in self._socks, {}
        if cold:
            t = time.perf_counter()
            self._socks[lang] = self.ws_open(lang)
            marks["ws_connect_ms"] = round((time.perf_counter() - t) * 1000, 1)
        ws, voice = self._socks[lang], self.voice(lang)
        t0 = time.perf_counter()
        self.ws_send(ws, text, lang, voice)
        chunks, arrivals, n, ttfa = [], [], 0, None
        try:
            for b in self.ws_recv(ws):
                if b:
                    now = (time.perf_counter() - t0) * 1000
                    ttfa = ttfa or now
                    chunks.append(b)
                    n += len(b)
                    arrivals.append((n, now))
        except Exception:
            self._socks.pop(lang, None)  # don't reuse a socket left mid-utterance
            raise
        total = (time.perf_counter() - t0) * 1000
        data = b"".join(chunks)
        if not data:
            raise RuntimeError(f"{self.slug}: empty audio")
        return dict(voice=voice, req={"transport": "websocket", "model": self.model_sent(lang)},
                    data=data, marks=marks, headers={}, cold=cold, http_version="websocket",
                    ttfa_speech_ms=_first_speech_ms(data, self.fmt, arrivals), ttfb_ms=None,
                    ttfa_ms=round(ttfa, 1) if ttfa else None, total_ms=round(total, 1))


class CartesiaSonic36WS(WsMixin, CartesiaSonic36):
    def ws_open(self, lang):
        return connect("wss://api.cartesia.ai/tts/websocket?" + urlencode(
            {"api_key": self.key, "cartesia_version": "2026-08-14"}), max_size=None)

    def ws_send(self, ws, text, lang, voice):
        ws.send(json.dumps({"model_id": self.model, "transcript": text, "language": lang,
                            "voice": {"mode": "id", "id": voice},
                            "output_format": {"container": "raw", "encoding": "pcm_s16le",
                                              "sample_rate": self.sr},
                            "context_id": uuid.uuid4().hex, "continue": False}))

    def ws_recv(self, ws):
        while True:
            e = self._j(ws)
            if e.get("type") == "error":
                raise RuntimeError(f"{self.slug}: {e}")
            if e.get("type") == "chunk" and e.get("data"):
                yield base64.b64decode(e["data"])
            if e.get("done"):
                return


class DeepgramFluxTTSWS(WsMixin, DeepgramFluxTTS):
    def ws_open(self, lang):
        return connect("wss://api.deepgram.com/v2/speak?" + urlencode(
            {"model": self.model_sent(lang), "encoding": "linear16", "sample_rate": self.sr}),
            additional_headers={"Authorization": f"Token {self.key}"}, max_size=None)

    def ws_send(self, ws, text, lang, voice):
        ws.send(json.dumps({"type": "Speak", "text": text}))
        ws.send(json.dumps({"type": "Flush"}))

    def ws_recv(self, ws):
        while True:
            e = self._j(ws)
            if isinstance(e, bytes):  # audio is raw binary frames
                yield e
            elif e.get("type") == "SpeechMetadata":  # sent after Flushed; ends the utterance
                return
            elif e.get("type") == "Error":
                raise RuntimeError(f"{self.slug}: {e}")


class XaiTTSWS(WsMixin, XaiTTS):
    def ws_open(self, lang):
        return connect("wss://api.x.ai/v1/tts?" + urlencode(
            {"language": self.LANG[lang], "voice": self.voice(lang), "codec": "pcm",
             "sample_rate": self.sr}),
            additional_headers={"Authorization": f"Bearer {self.key}"}, max_size=None)

    def ws_send(self, ws, text, lang, voice):
        ws.send(json.dumps({"type": "text.delta", "delta": text}))
        ws.send(json.dumps({"type": "text.done"}))

    def ws_recv(self, ws):
        while True:
            e = self._j(ws)
            if e.get("type") == "audio.delta":
                yield base64.b64decode(e["delta"])
            elif e.get("type") == "audio.done":
                return
            elif e.get("type") == "error":
                raise RuntimeError(f"{self.slug}: {e}")


class AssemblyTTS1WS(WsMixin, AssemblyTTS1):
    def ws_open(self, lang):
        ws = connect("wss://tts.us.assemblyai.com/v1/ws?" + urlencode(
            {"voice_id": self.voice(lang), "language": self.LANG[lang], "sample_rate": self.sr,
             "encoding": "pcm_s16le"}),
            additional_headers={"Authorization": self.key}, max_size=None)
        while self._j(ws).get("type") != "Begin":
            pass
        return ws

    def ws_send(self, ws, text, lang, voice):
        ws.send(json.dumps({"type": "Generate", "text": text}))
        ws.send(json.dumps({"type": "Flush"}))

    def ws_recv(self, ws):
        while True:
            e = self._j(ws)
            if e.get("type") == "Audio":
                yield base64.b64decode(e["audio"])
            elif e.get("type") == "FlushDone":
                return
            elif e.get("type") in ("Error", "error"):
                raise RuntimeError(f"{self.slug}: {e}")
