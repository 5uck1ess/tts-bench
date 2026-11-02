"""Palabra realtime TTS over WebSocket (platform.palabra.ai/docs/text-to-speech/realtime-tts, read
2026-11-02). WebSocket is the only TTS transport Palabra offers, so there is no HTTP twin.

- US endpoint (stream.us.palabra.ai): the bench measures from Eastern US.
- One session per language: `init` fixes language + voice for the session, so WsMixin's
  per-language socket cache fits. Each prompt is sent as one `text` message with is_eos=true and
  reset_context=true, so a take never inherits intonation from the previous prompt.
- `default_low` = the docs' example voice and the language's default voice (rule: quickstart voice).
- 48 kHz PCM, the top of the documented 8000–48000 range.
- End of a take = the audio_chunk with data.last_chunk=true (it carries no audio).
- Price: $0.012 / 1,000 characters (palabra.ai/pricing, read 2026-11-02).
"""

import base64
import json
from urllib.parse import urlencode

from websockets.sync.client import connect

from .. import core
from ..core import Adapter
from .ws import WsMixin

URL = "wss://stream.us.palabra.ai/tts-api/v1/text-to-speech/stream"


class PalabraBase(Adapter):
    slug = "palabra_tts"
    provider = "palabra"
    vendor = "Palabra"
    model = "auto"  # the API's own model selector: "best model for the language"
    langs = {"en", "fr", "es"}
    sr = 48000

    def voice(self, lang):
        return "default_low"


class PalabraTTS(WsMixin, PalabraBase):
    def ws_open(self, lang):
        ws = connect(URL + "?" + urlencode({"token": self.key}), max_size=core.MAX_RAW_BYTES,
                     open_timeout=15)
        ws.send(json.dumps({"type": "init", "language": lang, "model": self.model,
                            "voice_options": {"voice_id": self.voice(lang), "speed": 1.0},
                            "output": {"format": "pcm", "sample_rate": self.sr}}))
        return ws

    def ws_send(self, ws, text, lang, voice):
        # voice_id rides on every message too, so a socket opened for one lens can't read the
        # other lens's prompts in the wrong voice.
        ws.send(json.dumps({"type": "text", "text": text, "is_eos": True,
                            "reset_context": True, "voice_options": {"voice_id": voice}}))

    def ws_recv(self, ws):
        while True:
            e = self._j(ws)
            if isinstance(e, bytes):
                continue
            kind, data = e.get("message_type"), e.get("data") or {}
            if kind == "audio_chunk":
                if data.get("audio"):
                    yield base64.b64decode(data["audio"])
                if data.get("last_chunk"):
                    return
            elif kind == "error":
                raise RuntimeError(f"{self.slug}: {data}")
