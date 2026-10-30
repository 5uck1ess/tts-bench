"""Bland, Gradium, Async, Speechmatics. Specs read from each vendor's docs 2026-10-29. All four
docs point at plain HTTP for a complete, known-up-front text."""

import base64

from ..core import Adapter, json_lines


class BlandSpeech3(Adapter):
    slug = "bland_speech3"
    provider = "bland"
    vendor = "Bland"
    model = "btts-3"  # no model param: the voice picks it; x-model response header confirms
    # /v1/models lists BTTS_V3 as en-only; fr/es voices are BTTS_V2 (a different model).
    langs = {"en"}
    sr = 24000

    def voice(self, lang):
        return "29158307-9893-4149-8a75-bc9ce313d64e"  # quickstart "Karen" (General American)

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://api.bland.ai/v2/tts",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={"text": text, "voice": voice,
                          "audio": {"encoding": "pcm_s16le", "sample_rate": self.sr,
                                    "container": "raw"}})

    def echo(self, headers, body_meta):
        return {k: headers[k] for k in ("x-model", "x-latency") if k in headers}


class GradiumTTS(Adapter):
    slug = "gradium_tts"
    provider = "gradium"
    vendor = "Gradium"
    model = "default"  # docs: the latest production TTS; no versioned public id
    langs = {"en", "fr", "es"}
    sr = 24000
    params = {"only_audio": False}
    # Quickstart voice Emma is en-us; language comes from the voice, so fr/es use the first
    # voice the docs' catalog table lists for each.
    VOICES = {"en": "YTpq7expH9539ERJ", "fr": "b35yykvVppLXyw_l", "es": "B36pbz5_UoWn4BDl"}

    def voice(self, lang):
        return self.VOICES[lang]

    def request(self, text, lang, voice):
        # only_audio=false is the documented streaming path (NDJSON with base64 audio).
        return dict(method="POST", url="https://api.gradium.ai/api/post/speech/tts",
                    headers={"x-api-key": self.key},
                    json={"text": text, "voice_id": voice, "model_name": self.model,
                          "output_format": f"pcm_{self.sr}", "only_audio": False})

    def audio_chunks(self, raw_iter):
        for ev in json_lines(raw_iter):
            if ev.get("type") == "audio" and ev.get("audio"):
                yield base64.b64decode(ev["audio"])
            elif ev.get("type") == "error":  # mid-stream errors arrive inside an HTTP 200
                raise RuntimeError(f"{self.slug}: {ev}")


class AsyncFlash15(Adapter):
    slug = "async_flash15"
    provider = "async"
    vendor = "Async"
    model = "async_flash_v1.5"  # latest Flash; Pro v1.0 is English-only
    langs = {"en", "fr", "es"}
    sr = 24000

    def voice(self, lang):
        return "e0f39dc4-f691-4e78-bba5-5c636692cc04"  # quickstart voice; any voice, any lang

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://api.async.com/text_to_speech/streaming",
                    headers={"x-api-key": self.key, "version": "v1"},
                    json={"model_id": self.model, "transcript": text,
                          "voice": {"mode": "id", "id": voice}, "language": lang,
                          "output_format": {"container": "raw", "encoding": "pcm_s16le",
                                            "sample_rate": self.sr}})


class SpeechmaticsTTS(Adapter):
    slug = "speechmatics_tts"
    provider = "speechmatics"
    vendor = "Speechmatics"
    model = "speechmatics-tts"  # no model id; voice goes in the path
    langs = {"en"}  # English only
    sr = 16000      # the only rate offered

    def voice(self, lang):
        return "sarah"  # quickstart voice (UK female)

    def request(self, text, lang, voice):
        return dict(method="POST",
                    url=f"https://preview.tts.speechmatics.com/generate/{voice}",
                    params={"output_format": "pcm_16000"},
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={"text": text})
