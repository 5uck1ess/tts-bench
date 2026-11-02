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

from .. import core

from ..core import _first_speech_ms
from .batch_a import CartesiaSonic36, InworldTTS2
from .batch_b import (AssemblyTTS1, DeepgramFluxTTS, SmallestLightning31Pro, SonioxTTS2,
                      StepAudio3TTS, XaiTTS)
from .batch_c import MurfFalcon21


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
        chunks, arrivals, n, ttfa = [], [], 0, None
        try:
            self.ws_send(ws, text, lang, voice)
            for b in self.ws_recv(ws):
                core.guard(n, t0, self.slug)  # time ceiling also covers audio-less messages
                if b:
                    now = (time.perf_counter() - t0) * 1000
                    ttfa = ttfa or now
                    chunks.append(b)
                    n += len(b)
                    core.guard(n, t0, self.slug)
                    arrivals.append((n, now))
        except Exception:
            # closed or left mid-utterance: drop it so the next call reconnects
            dead = self._socks.pop(lang, None)
            try:
                dead and dead.close()
            except Exception:  # noqa: BLE001
                pass
            raise
        # Total = last audio byte, not the end-of-utterance message: Smallest sends "complete"
        # ~4 s after its final chunk, which isn't delivery time.
        total = arrivals[-1][1] if arrivals else (time.perf_counter() - t0) * 1000
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
            {"api_key": self.key, "cartesia_version": "2026-08-14"}), max_size=core.MAX_RAW_BYTES)

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
            additional_headers={"Authorization": f"Token {self.key}"}, max_size=core.MAX_RAW_BYTES)

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
            additional_headers={"Authorization": f"Bearer {self.key}"}, max_size=core.MAX_RAW_BYTES)

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
            additional_headers={"Authorization": self.key}, max_size=core.MAX_RAW_BYTES)
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


# --- Socket twins used only by the speed sampler (scripts/hosted/speed_sample.py). The sweep
# keeps these vendors on HTTP; the sampler measures both transports over several days.

class InworldTTS2WS(WsMixin, InworldTTS2):
    def ws_open(self, lang):
        return connect("wss://api.inworld.ai/tts/v1/voice:streamBidirectional",
                       additional_headers={"Authorization": f"Basic {self.key}"}, max_size=core.MAX_RAW_BYTES)

    def ws_send(self, ws, text, lang, voice):
        cid = uuid.uuid4().hex[:12]
        create = {"voiceId": voice, "modelId": self.model,
                  "audioConfig": {"audioEncoding": "PCM", "sampleRateHertz": self.sr}}
        if lang != "en":
            create["language"] = lang
        ws.send(json.dumps({"create": create, "contextId": cid}))
        ws.send(json.dumps({"send_text": {"text": text, "flush_context": {}}, "contextId": cid}))
        ws.send(json.dumps({"close_context": {}, "contextId": cid}))

    def ws_recv(self, ws):
        while True:
            e = self._j(ws)
            r = e.get("result") or {}
            if e.get("error") or (r.get("status") or {}).get("code"):
                raise RuntimeError(f"{self.slug}: {e}")
            a = (r.get("audioChunk") or {}).get("audioContent")
            if a:
                yield base64.b64decode(a)
            if "contextClosed" in r:
                return


class SonioxTTS2WS(WsMixin, SonioxTTS2):
    def ws_open(self, lang):
        return connect("wss://tts-rt.soniox.com/tts-websocket",
                       additional_headers={"Authorization": f"Bearer {self.key}"}, max_size=core.MAX_RAW_BYTES)

    def ws_send(self, ws, text, lang, voice):
        sid = uuid.uuid4().hex[:12]
        ws.send(json.dumps({"stream_id": sid, "model": self.model, "language": lang,
                            "voice": voice, "audio_format": "pcm_s16le",
                            "sample_rate": self.sr}))
        ws.send(json.dumps({"stream_id": sid, "text": text, "text_end": True}))

    def ws_recv(self, ws):
        while True:
            e = self._j(ws)
            if e.get("error_code") or e.get("error"):
                raise RuntimeError(f"{self.slug}: {e}")
            if e.get("audio"):
                yield base64.b64decode(e["audio"])
            if e.get("terminated"):  # follows the audio_end chunk
                return


class SmallestLightning31ProWS(WsMixin, SmallestLightning31Pro):
    def ws_open(self, lang):
        return connect("wss://api.smallest.ai/waves/v1/tts/live",
                       additional_headers={"Authorization": f"Bearer {self.key}"}, max_size=core.MAX_RAW_BYTES)

    def ws_send(self, ws, text, lang, voice):
        ws.send(json.dumps({"text": text, "voice_id": voice, "model": self.model,
                            "sample_rate": self.sr, "language": lang}))

    def ws_recv(self, ws):
        while True:
            e = self._j(ws)
            if e.get("status") == "error":
                raise RuntimeError(f"{self.slug}: {e}")
            a = (e.get("data") or {}).get("audio")
            if a:
                yield base64.b64decode(a)
            if e.get("status") in ("complete", "done", "completed"):
                return


class MurfFalcon21WS(WsMixin, MurfFalcon21):
    def ws_open(self, lang):
        return connect("wss://global.api.murf.ai/v1/speech/stream-input?" + urlencode(
            {"api-key": self.key, "model": self.model, "sample_rate": self.sr,
             "channel_type": "MONO", "format": "PCM"}), max_size=core.MAX_RAW_BYTES)

    def ws_send(self, ws, text, lang, voice):
        cid = uuid.uuid4().hex[:12]
        ws.send(json.dumps({"voice_config": {"voiceId": voice, "locale": self.VOICES[lang][1]},
                            "context_id": cid}))
        ws.send(json.dumps({"text": text, "end": True, "context_id": cid}))

    def ws_recv(self, ws):
        while True:
            e = self._j(ws)
            if e.get("error"):
                raise RuntimeError(f"{self.slug}: {e}")
            if e.get("audio"):
                yield base64.b64decode(e["audio"])
            if e.get("final"):
                return


class StepAudio3TTSWS(WsMixin, StepAudio3TTS):
    """Session is created at connect (tts.create); the server drops idle sockets, so a dropped
    socket surfaces as an error row and the next call reconnects (WsMixin pops it)."""
    def ws_open(self, lang):
        ws = connect(f"wss://api.stepfun.ai/v1/realtime/audio?model={self.model}",
                     additional_headers={"Authorization": f"Bearer {self.key}"}, max_size=core.MAX_RAW_BYTES)
        self._sid = self._j(ws)["data"]["session_id"]
        ws.send(json.dumps({"type": "tts.create", "data": {
            "session_id": self._sid, "voice_id": self.voice(lang), "response_format": "pcm",
            "sample_rate": self.sr, "mode": "sentence"}}))
        while self._j(ws).get("type") != "tts.response.created":
            pass
        return ws

    def ws_send(self, ws, text, lang, voice):
        ws.send(json.dumps({"type": "tts.text.delta",
                            "data": {"session_id": self._sid, "text": text}}))
        ws.send(json.dumps({"type": "tts.text.flush", "data": {"session_id": self._sid}}))

    def ws_recv(self, ws):
        while True:
            e = self._j(ws)
            t = e.get("type", "")
            if t == "tts.response.audio.delta":
                yield base64.b64decode((e.get("data") or {})["audio"])
            elif t == "tts.response.sentence.end":
                return
            elif "error" in t:
                raise RuntimeError(f"{self.slug}: {e}")
