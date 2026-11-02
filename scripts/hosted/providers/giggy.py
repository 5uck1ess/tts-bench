
from .. import core
from ..core import Adapter

BASE = "https://api.giggy.ai/v1"
LANG_NAME = {"en": "english", "fr": "french", "es": "spanish"}


class GiggyTTS(Adapter):
    """GiggySpeech. Streaming mode (paid credits) is the only progressive one: PCM16 24 kHz.
    Batch is free but returns finished MP3 only, so it can't give a TTFA."""

    slug = "giggy_tts"
    provider = "giggy"
    vendor = "Giggy"
    model = "giggyspeech"
    # /v1/voices (2026-10-29) lists 10 voices: 8 English, 1 Oromo, 1 Urdu. language_code must
    # match the voice's language, so fr/es are not supported.
    langs = {"en"}
    fmt = "pcm16"
    sr = 24000
    params = {"mode": "streaming"}
    _voices = None

    def _catalog(self):
        if self._voices is None:
            with core.new_client() as c:  # bounded transport, like every other call
                r = c.get(f"{BASE}/voices", headers={"xi-api-key": self.key})
            r.raise_for_status()
            self._voices = r.json()["voices"]
        return self._voices

    def voice(self, lang):
        # No quickstart voice id is published; rule fallback = first catalog voice whose
        # labels.language is exactly this language.
        for v in self._catalog():
            if (v.get("labels") or {}).get("language", "").lower() == LANG_NAME[lang]:
                return v["voice_id"]
        raise RuntimeError(f"no Giggy voice for {lang}")

    def request(self, text, lang, voice):
        return dict(method="POST", url=f"{BASE}/text-to-speech",
                    headers={"xi-api-key": self.key},
                    json={"text": text, "voice_id": voice, "model_id": self.model,
                          "mode": "streaming", "output_format": "pcm_24000"})
