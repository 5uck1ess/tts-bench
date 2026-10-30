import httpx

from ..core import Adapter

BASE = "https://api.giggy.ai/v1"


class GiggyTTS(Adapter):
    """GiggySpeech. Streaming mode (paid credits) is the only progressive one: PCM16 24 kHz.
    Batch is free but returns finished MP3 only, so it can't give a TTFA."""

    slug = "giggy_tts"
    provider = "giggy"
    vendor = "Giggy"
    model = "giggyspeech"
    langs = {"en", "fr", "es"}  # narrowed at runtime to languages /v1/voices actually lists
    fmt = "pcm16"
    sr = 24000
    params = {"mode": "streaming"}
    _voices = None

    def _catalog(self):
        if self._voices is None:
            r = httpx.get(f"{BASE}/voices", headers={"xi-api-key": self.key}, timeout=30)
            r.raise_for_status()
            data = r.json()
            self._voices = data.get("voices", data) if isinstance(data, dict) else data
        return self._voices

    def voice(self, lang):
        # Giggy publishes no quickstart voice id; rule fallback: first catalog voice per language.
        for v in self._catalog():
            vl = (v.get("language") or v.get("labels", {}).get("language") or "").lower()
            if vl.startswith(lang):
                return v.get("voice_id") or v.get("id")
        raise RuntimeError(f"no Giggy voice for {lang}")

    def request(self, text, lang, voice):
        return dict(method="POST", url=f"{BASE}/text-to-speech",
                    headers={"xi-api-key": self.key},
                    json={"text": text, "voice_id": voice, "mode": "streaming",
                          "output_format": "pcm_24000"})
