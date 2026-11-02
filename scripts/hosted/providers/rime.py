from ..core import Adapter

# docs.rime.ai: quickstart voice is "celeste" (English only). Coda voices are per-language, so
# fr/es use the first voice the Coda catalog lists for that language (docs/voices-coda.md).
VOICES = {"en": "celeste", "fr": "aurelie", "es": "alba"}


class RimeCoda(Adapter):
    slug = "rime_coda"
    provider = "rime"
    vendor = "Rime"
    model = "coda"
    langs = {"en", "fr", "es"}
    fmt = "pcm16"
    sr = 24000  # Coda default samplingRate

    def voice(self, lang):
        return VOICES[lang]

    def request(self, text, lang, voice):
        # Model is chosen by host; Accept audio/L16 = headerless PCM16 at samplingRate.
        return dict(method="POST", url="https://coda.api.rime.ai/",
                    headers={"Authorization": f"Bearer {self.key}", "Accept": "audio/L16"},
                    json={"text": text, "speaker": voice, "lang": lang,
                          "samplingRate": self.sr})
