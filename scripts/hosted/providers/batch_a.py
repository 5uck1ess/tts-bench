"""Cartesia, Gemini, Inworld, Speechify, Fish, Mistral. Specs read from each vendor's docs 2026-10-29."""

import base64

from ..core import Adapter, json_lines

FR_ES = {"en", "fr", "es"}


class CartesiaSonic36(Adapter):
    slug = "cartesia_sonic36"
    provider = "cartesia"
    vendor = "Cartesia"
    model = "sonic-3.6"
    langs = FR_ES
    sr = 24000

    def voice(self, lang):
        return "f786b574-daa5-4673-aa0c-cbe3e8534c02"  # quickstart voice; speaks all model langs

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://api.cartesia.ai/tts/bytes",
                    headers={"Authorization": f"Bearer {self.key}",
                             "Cartesia-Version": "2026-08-14"},
                    json={"model_id": self.model, "transcript": text, "voice": voice,
                          "language": lang,
                          "output_format": {"container": "raw", "encoding": "pcm_s16le",
                                            "sample_rate": self.sr}})


class Gemini38FlashTTS(Adapter):
    slug = "gemini38_flash_tts"
    provider = "gemini"
    vendor = "Google"
    model = "gemini-3.8-flash-tts"
    langs = FR_ES
    sr = 24000
    params = {"stream": True, "mime_type": "audio/l16"}

    def voice(self, lang):
        return "Kore"  # quickstart voice; language is detected from the text

    def request(self, text, lang, voice):
        return dict(method="POST",
                    url="https://generativelanguage.googleapis.com/v1beta/interactions",
                    headers={"x-goog-api-key": self.key},
                    json={"model": self.model,
                          "input": [{"type": "user_input",
                                     "content": [{"type": "text", "text": text}]}],
                          "response_format": {"type": "audio", "mime_type": "audio/l16",
                                              "sample_rate": self.sr},
                          "generation_config": {"speech_config": [{"voice": voice}]},
                          "stream": True})

    def audio_chunks(self, raw_iter):
        for ev in json_lines(raw_iter):
            d = ev.get("delta") or {}
            # Live events carry {"mime_type": "audio/l16", "data": b64}, no "type" key.
            if str(d.get("mime_type", "")).startswith("audio") and d.get("data"):
                yield base64.b64decode(d["data"])


class InworldTTS2(Adapter):
    slug = "inworld_tts2"
    provider = "inworld"
    vendor = "Inworld"
    model = "inworld-tts-2"
    langs = FR_ES
    sr = 48000

    def voice(self, lang):
        return "Ashley"  # quickstart voice; other languages are "best-effort" via `language`

    def request(self, text, lang, voice):
        body = {"text": text, "voiceId": voice, "modelId": self.model,
                "audioConfig": {"audioEncoding": "PCM", "sampleRateHertz": self.sr}}
        if lang != "en":
            body["language"] = lang
        return dict(method="POST", url="https://api.inworld.ai/tts/v1/voice:stream",
                    headers={"Authorization": f"Basic {self.key}"}, json=body)

    def audio_chunks(self, raw_iter):
        for ev in json_lines(raw_iter):
            a = (ev.get("result") or {}).get("audioContent")
            if a:
                yield base64.b64decode(a)


class SpeechifySimba35(Adapter):
    slug = "speechify_simba35"
    provider = "speechify"
    vendor = "Speechify"
    model = "simba-3.5"
    langs = FR_ES
    sr = 24000
    LOCALE = {"en": "en-US", "fr": "fr-FR", "es": "es-ES"}

    def voice(self, lang):
        return "geffen_32"  # quickstart voice; on simba-3.5 every voice speaks all 48 langs

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://api.speechify.ai/v1/audio/stream",
                    headers={"Authorization": f"Bearer {self.key}", "Accept": "audio/pcm"},
                    json={"input": text, "voice_id": voice, "model": self.model,
                          "language": self.LOCALE[lang], "output_format": f"pcm_{self.sr}"})


class FishDrama3(Adapter):
    slug = "fish_drama3"
    provider = "fish"
    vendor = "Fish Audio"
    model = "drama-3"
    langs = FR_ES
    sr = 44100
    params = {"latency": "balanced"}

    def voice(self, lang):
        # Quickstart sends no reference_id (random speaker per call); rule fallback = first voice
        # on the Drama 3 docs page, "Adrian".
        return "bf322df2096a46f18c579d0baa36f41d"

    def request(self, text, lang, voice):
        # A missing/unknown `model` header silently falls back to s2.1-pro.
        return dict(method="POST", url="https://api.fish.audio/v1/tts",
                    headers={"Authorization": f"Bearer {self.key}", "model": self.model},
                    json={"text": text, "reference_id": voice, "format": "pcm",
                          "sample_rate": self.sr, "latency": "balanced"})


class Voxtral3TTS(Adapter):
    slug = "voxtral3_tts"
    provider = "mistral"
    vendor = "Mistral"
    model = "voxtral-mini-tts-3-0"
    langs = FR_ES
    fmt = "pcm_f32"  # "pcm" is float32 LE (checked on a live clip 2026-10-29), not s16
    sr = 24000
    params = {"stream": True, "response_format": "pcm"}
    # Quickstart voice is English; voices are per-language, so fr/es use the first Voxtral 3
    # voice the catalog lists for each.
    VOICES = {"en": "gb_jane_neutral", "fr": "voice_592", "es": "voice_483"}

    def voice(self, lang):
        return self.VOICES[lang]

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://api.mistral.ai/v1/audio/speech",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={"model": self.model, "input": text, "voice_id": voice,
                          "response_format": "pcm", "stream": True})

    def audio_chunks(self, raw_iter):
        for ev in json_lines(raw_iter):
            if ev.get("audio_data"):
                yield base64.b64decode(ev["audio_data"])
