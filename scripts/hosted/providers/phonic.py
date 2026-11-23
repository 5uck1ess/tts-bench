"""Phonic Merritt over HTTP (docs.phonic.ai/api-reference/voices/preview, read 2026-11-23).

- Phonic sells speech-to-speech agents (Phonic-1, Phonic-1.1 released 2026-11-23). Those take
  audio in and speak the agent's own reply, so they can't read the bench prompts and have no row.
  Text-to-speech is documented as the voice "Preview" endpoint: POST https://api.phonic.ai/v1/tts,
  one model, `merritt`. That endpoint is this row.
- Synchronous: the response is JSON {"audio": <base64>}, the whole clip at once, so
  streaming = False (no first-speech time).
- Default settings, per the docs: voice `sabrina`, speed 1.2, `pcm_44100` (headerless PCM16 mono,
  confirmed by probe 2026-11-23). Voice and format are sent explicitly to pin them; speed is left
  at the vendor default.
- Voices carry no language tag (the base-voice table lists accents only), and the endpoint takes
  `languages` separately, so the default voice reads fr/es too, with `languages: [<lang>]` (one
  language "selects it directly"; an empty list means English).
- Price: none for TTS on its own. Phonic bills speech-to-speech conversation minutes ($0.07/min
  Starter, phonic.ai/pricing 2026-11-23); the preview endpoint isn't priced separately.
- Cloning: no self-serve API. Custom voices are provisioned for a workspace by Phonic.
"""

import base64
import json

from ..core import Adapter


class PhonicMerritt(Adapter):
    slug = "phonic_merritt"
    provider = "phonic"
    vendor = "Phonic"
    model = "merritt"
    langs = {"en", "fr", "es"}
    streaming = False
    fmt = "pcm16"
    sr = 44100
    VOICE = "sabrina"  # the endpoint's documented default voice_id

    def voice(self, lang):
        return self.VOICE

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://api.phonic.ai/v1/tts",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={"text": text, "model": self.model, "voice_id": voice,
                          "output_format": f"pcm_{self.sr}", "languages": [lang]})

    def audio_chunks(self, raw_iter):
        body = json.loads(b"".join(raw_iter))
        if not body.get("audio"):
            raise RuntimeError(f"{self.slug}: no audio in response: {str(body)[:300]}")
        yield base64.b64decode(body["audio"])
