"""xAI, AssemblyAI, Soniox, StepFun, Smallest.ai, Deepgram. Specs read from each vendor's docs
2026-10-29 (HTTP streaming paths; WebSocket variants are a later speed pass)."""

import base64

from ..core import Adapter, json_lines

FR_ES = {"en", "fr", "es"}


class XaiTTS(Adapter):
    slug = "xai_tts"
    provider = "xai"
    vendor = "xAI"
    model = "xai-tts"  # the TTS endpoint takes no model field
    langs = FR_ES
    sr = 24000
    LANG = {"en": "en", "fr": "fr", "es": "es-ES"}  # plain "es" isn't an accepted code

    def voice(self, lang):
        return "eve"  # quickstart voice; every voice speaks every language

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://api.x.ai/v1/tts",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={"text": text, "voice_id": voice, "language": self.LANG[lang],
                          "output_format": {"codec": "pcm", "sample_rate": self.sr}})


class AssemblyTTS1(Adapter):
    slug = "assemblyai_tts1"
    provider = "assemblyai"
    vendor = "AssemblyAI"
    model = "assemblyai-tts-1"  # default model; echoed in x-tts-model
    langs = FR_ES
    sr = 24000
    LANG = {"en": "english", "fr": "french", "es": "spanish"}

    def voice(self, lang):
        return "49c443a4-8a1d-5f7e-b70b-98bdbd9823c3"  # quickstart "Audrey"; speaks all 6 langs

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://tts.us.assemblyai.com/v1/speech",
                    headers={"Authorization": self.key},
                    json={"text": text, "voice_id": voice, "language": self.LANG[lang]})

    def echo(self, headers, body_meta):
        e = super().echo(headers, body_meta)
        if "x-tts-model" in headers:
            e["x-tts-model"] = headers["x-tts-model"]
        return e


class SonioxTTS2(Adapter):
    slug = "soniox_tts2"
    provider = "soniox"
    vendor = "Soniox"
    model = "tts-rt-v2"
    langs = FR_ES
    sr = 24000

    def voice(self, lang):
        return "Adrian"  # example default voice; any voice speaks every language

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://tts-rt.soniox.com/tts",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={"model": self.model, "language": lang, "voice": voice,
                          "audio_format": "pcm_s16le", "sample_rate": self.sr, "text": text})


class StepAudio3TTS(Adapter):
    slug = "stepaudio3_tts"
    provider = "stepfun"
    vendor = "StepFun"
    model = "stepaudio-3-tts"
    langs = FR_ES  # fr/es are "preview" per StepFun docs
    sr = 24000
    params = {"stream_format": "sse"}

    def voice(self, lang):
        return "lively-girl"  # quickstart voice (documented on 2.5; probe confirms on 3)

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://api.stepfun.ai/v1/audio/speech",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={"model": self.model, "input": text, "voice": voice, "language": lang,
                          "response_format": "pcm", "sample_rate": self.sr,
                          "stream_format": "sse"})

    def audio_chunks(self, raw_iter):
        for ev in json_lines(raw_iter):
            if ev.get("type") == "speech.audio.delta" and ev.get("audio"):
                yield base64.b64decode(ev["audio"])


class SmallestLightning31Pro(Adapter):
    slug = "smallest_lightning31pro"
    provider = "smallest"
    vendor = "Smallest.ai"
    model = "lightning_v3.1_pro"
    langs = FR_ES
    sr = 24000
    # Quickstart voice "meher" is Indian English; Pro voices are per-language.
    VOICES = {"en": "meher", "fr": "manon", "es": "martina"}

    def voice(self, lang):
        return self.VOICES[lang]

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://api.smallest.ai/waves/v1/tts/live",
                    headers={"Authorization": f"Bearer {self.key}",
                             "Accept": "text/event-stream"},
                    json={"text": text, "voice_id": voice, "model": self.model,
                          "sample_rate": self.sr, "language": lang})

    def audio_chunks(self, raw_iter):
        for ev in json_lines(raw_iter):
            a = ev.get("audio") or (ev.get("data") or {}).get("audio") if isinstance(ev, dict) else None
            if a:
                yield base64.b64decode(a)


class DeepgramFluxTTS(Adapter):
    slug = "deepgram_flux_tts"
    provider = "deepgram"
    vendor = "Deepgram"
    model = "flux-tts"
    langs = FR_ES
    sr = 24000
    # One voice per language; the model string is flux-{voice}-{lang}. Quickstart = haley (en).
    VOICES = {"en": "haley", "fr": "arnaud", "es": "aina"}

    def voice(self, lang):
        return self.VOICES[lang]

    def model_sent(self, lang):
        return f"flux-{self.VOICES[lang]}-{lang}"

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://api.deepgram.com/v2/speak",
                    params={"model": self.model_sent(lang), "encoding": "linear16",
                            "sample_rate": self.sr},
                    headers={"Authorization": f"Token {self.key}"},
                    json={"text": text})
