"""Microsoft MAI-Voice, VUI Luna, OpenAI, Murf, HeyGen, MiniMax via OpenRouter. Specs read from
each vendor's docs 2026-10-29."""

import base64
import io
import wave
from xml.sax.saxutils import escape

from ..core import Adapter, json_lines

FR_ES = {"en", "fr", "es"}


class MaiVoice21(Adapter):
    slug = "mai_voice21"
    provider = "azure"
    vendor = "Microsoft"
    model = "MAI-Voice-2.1"
    langs = FR_ES
    sr = 24000
    REGION = "eastus2"  # the region Tym's Azure Speech key belongs to
    # Doc example voice is Harper; it exists in each of our locales.
    LOCALES = {"en": "en-US", "fr": "fr-FR", "es": "es-MX"}

    def voice(self, lang):
        return f"{self.LOCALES[lang]}-Harper:{self.model}"

    def request(self, text, lang, voice):
        ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
                f'xml:lang="{self.LOCALES[lang]}"><voice name="{voice}">{escape(text)}'
                f'</voice></speak>')
        return dict(method="POST",
                    url=f"https://{self.REGION}.tts.speech.microsoft.com/cognitiveservices/v1",
                    headers={"Ocp-Apim-Subscription-Key": self.key,
                             "Content-Type": "application/ssml+xml",
                             "X-Microsoft-OutputFormat": "raw-24khz-16bit-mono-pcm",
                             "User-Agent": "tts-bench"},
                    content=ssml.encode("utf-8"))


class VuiLuna(Adapter):
    slug = "vui_luna"
    provider = "vui"
    vendor = "VUI Labs"
    model = "luna-tts"
    langs = {"en"}  # trained on zh/en/ja/ko only
    sr = 24000      # overwritten from X-Audio-Sample-Rate

    def voice(self, lang):
        return "fiona"  # quickstart has a placeholder; first system voice

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://api.vuilabs.ai/v1/text-to-speech/stream",
                    headers={"X-API-Key": self.key},
                    json={"model_id": self.model, "text": text, "voice_id": voice,
                          "language": lang, "audio_format": "pcm"})

    def on_headers(self, headers):
        if headers.get("x-audio-sample-rate"):
            self.sr = int(headers["x-audio-sample-rate"])


class OpenAIMiniTTS(Adapter):
    slug = "openai_gpt4o_mini_tts"
    provider = "openai"
    vendor = "OpenAI"
    # Latest snapshot, pinned. Deprecated 2026-10-01, shuts down 2027-01-06; no newer TTS model.
    model = "gpt-4o-mini-tts-2025-12-15"
    langs = FR_ES
    sr = 24000

    def voice(self, lang):
        return "marin"  # TTS guide: "For best quality, we recommend using marin or cedar"

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://api.openai.com/v1/audio/speech",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={"model": self.model, "input": text, "voice": voice,
                          "response_format": "pcm"})


class MurfFalcon21(Adapter):
    slug = "murf_falcon21"
    provider = "murf"
    vendor = "Murf"
    model = "falcon-2.1"  # old "falcon-2" id now serves 2.1
    langs = FR_ES
    sr = 24000
    # Quickstart voice Gordon is en-US only; first catalog voice per locale otherwise.
    VOICES = {"en": ("Gordon", "en-US"), "fr": ("Adélie", "fr-FR"), "es": ("Carmen", "es-ES")}

    def voice(self, lang):
        return self.VOICES[lang][0]

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://global.api.murf.ai/v1/speech/stream",
                    headers={"api-key": self.key},
                    json={"text": text, "voiceId": voice, "model": self.model,
                          "locale": self.VOICES[lang][1], "format": "PCM",
                          "sampleRate": self.sr})


class HeyGenVoice1(Adapter):
    slug = "heygen_voice1"
    provider = "heygen"
    vendor = "HeyGen"
    model = "heygen-voice-1"
    langs = FR_ES
    sr = 44100  # replaced by the first part's WAV header

    def voice(self, lang):
        return "83548de556df48ba8c09c42a57c51d85"  # quickstart "Nadine"; speaks any `language`

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://api.heygen.com/v3/models/audio/tts/stream",
                    headers={"X-Api-Key": self.key, "Accept": "text/event-stream"},
                    json={"model": self.model, "voice_id": voice, "text": text,
                          "language": lang})

    def audio_chunks(self, raw_iter):
        # Each SSE part is a complete base64 WAV; strip headers and concatenate the PCM.
        for ev in json_lines(raw_iter):
            if ev.get("type") == "audio" and ev.get("audio"):
                with wave.open(io.BytesIO(base64.b64decode(ev["audio"]))) as w:
                    self.sr = w.getframerate()
                    yield w.readframes(w.getnframes())


class MiniMaxSpeech28HD(Adapter):
    slug = "minimax_speech28hd"
    provider = "openrouter"
    vendor = "MiniMax"
    model = "minimax/speech-2.8-hd"
    route = "openrouter"
    langs = {"en"}  # OpenRouter lists only English_* voices for this model
    fmt = "mp3"     # MiniMax on OpenRouter is mp3-only (pcm returns 400)

    def voice(self, lang):
        return "English_expressive_narrator"

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://openrouter.ai/api/v1/audio/speech",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={"model": self.model, "input": text, "voice": voice,
                          "response_format": "mp3"})
