"""OpenAI gpt-realtime-2.1 over the Realtime WebSocket. A speech-to-speech LLM, not a TTS model:
it is told to read the prompt verbatim, and WER shows where it doesn't. Benched because OpenAI's
TTS guide names the Realtime API as the migration path off gpt-4o-mini-tts."""

import base64
import json
import time

from websockets.sync.client import connect

from .. import core

from ..core import Adapter, _first_speech_ms

INSTRUCTIONS = ("You are a text-to-speech engine. Read the user's text aloud exactly as written, "
                "word for word, in the language it is written in. Do not add, remove, translate, "
                "answer, or comment on anything.")


class OpenAIRealtime21(Adapter):
    slug = "openai_realtime21"
    provider = "openai"
    vendor = "OpenAI"
    model = "gpt-realtime-2.1"
    route = "realtime-ws"
    langs = {"en", "fr", "es"}
    sr = 24000
    params = {"instructions": "read-verbatim", "conversation": "none"}
    _ws = None

    def voice(self, lang):
        return "marin"  # Realtime docs' example voice; also the TTS guide's best-quality pick

    def _session(self):
        """One socket per run (warm after the warmup call); each prompt is an out-of-band
        response with its own input, so earlier prompts never sit in the model's context."""
        if self._ws is not None:
            return self._ws, False
        t0 = time.perf_counter()
        ws = connect(f"wss://api.openai.com/v1/realtime?model={self.model}",
                     additional_headers={"Authorization": f"Bearer {self.key}"},
                     max_size=core.MAX_RAW_BYTES, open_timeout=15)
        ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime", "model": self.model, "output_modalities": ["audio"],
            "instructions": INSTRUCTIONS,
            "audio": {"output": {"format": {"type": "audio/pcm", "rate": self.sr},
                                 "voice": self.voice("en")}}}}))
        while True:
            ev = json.loads(ws.recv(timeout=30))
            if ev["type"] == "error":
                raise RuntimeError(f"{self.slug}: {ev.get('error')}")
            if ev["type"] == "session.updated":
                break
        self._connect_ms = round((time.perf_counter() - t0) * 1000, 1)
        self._ws = ws
        return ws, True

    def synth_ws(self, text, lang):
        try:
            return self._synth_once(text, lang)
        except Exception:
            # a closed or mid-response session must not be reused: next call reconnects
            ws, self._ws = self._ws, None
            try:
                ws and ws.close()
            except Exception:  # noqa: BLE001
                pass
            raise

    def _synth_once(self, text, lang):
        ws, cold = self._session()
        t0 = time.perf_counter()
        ws.send(json.dumps({"type": "response.create", "response": {
            "conversation": "none", "output_modalities": ["audio"],
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": text}]}]}}))
        chunks, arrivals, nbytes, ttfa, transcript = [], [], 0, None, ""
        while True:
            ev = json.loads(ws.recv(timeout=120))
            core.guard(nbytes, t0, self.slug)  # time ceiling also covers audio-less events
            t = ev["type"]
            if t == "response.output_audio.delta":
                now = (time.perf_counter() - t0) * 1000
                ttfa = ttfa or now
                b = base64.b64decode(ev["delta"])
                chunks.append(b)
                nbytes += len(b)
                core.guard(nbytes, t0, self.slug)
                arrivals.append((nbytes, now))
            elif t == "response.output_audio_transcript.done":
                transcript = ev.get("transcript", "")
            elif t == "error":
                raise RuntimeError(f"{self.slug}: {ev.get('error')}")
            elif t == "response.done":
                status = ev["response"].get("status")
                if status != "completed":
                    raise RuntimeError(f"{self.slug}: response {status}: "
                                       f"{ev['response'].get('status_details')}")
                break
        total = (time.perf_counter() - t0) * 1000
        data = b"".join(chunks)
        if not data:
            raise RuntimeError(f"{self.slug}: empty audio")
        self.last_transcript = transcript
        return dict(voice=self.voice(lang), req={"url": "wss://api.openai.com/v1/realtime",
                                                 "model": self.model},
                    data=data, marks={"ws_connect_ms": self._connect_ms} if cold else {},
                    headers={}, cold=cold, http_version="websocket",
                    ttfa_speech_ms=_first_speech_ms(data, self.fmt, arrivals),
                    ttfb_ms=None, ttfa_ms=round(ttfa, 1) if ttfa else None,
                    total_ms=round(total, 1))

    def echo(self, headers, body_meta):
        return {"model_transcript": getattr(self, "last_transcript", None)}
