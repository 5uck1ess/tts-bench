"""Voicefinder Voyager 1 over HTTP (voicefinder.io/app/developers/docs, read 2026-11-18).

- One model, `voyager-1`, 55 languages; any voice speaks any of them (native where the voice has a
  recording in that language, otherwise with its main language's accent: x-language-warning).
- Endpoint: POST https://api.voicefinder.io/v1/text-to-speech/{voice_id}. Synchronous: the body IS
  the audio. /stream exists but "does not reduce latency: the audio is generated and marked in full
  before the first byte is sent", so this row is the plain endpoint and streaming = False.
- WAV, the default format, is delivered at 48 kHz (x-output-format reports wav_48000).
- `language` is always sent: auto-detection on short Latin text falls back to Spanish and skips
  the pronunciation dictionary.
- Every file carries an inaudible watermark + C2PA credentials (x-watermarked / x-c2pa-signed):
  PCM16 mono 48 kHz with a trailing `C2PA` RIFF chunk. Terms 7.3 forbid removing AI marks or
  metadata, so keep_bytes saves the WAV exactly as delivered (core's re-encode would drop the chunk).
- Rate-limit strikes escalate to 15 min / 1 h / 24 h lockouts, and the free tier allows one
  concurrent generation, so requests are paced and serial.
- Voice: the quickstart voice, Agatha (f6abdab0, English (British)), for en. It has only an English
  recording, so fr/es would get an English accent (x-language-warning, seen 2026-11-18); per the
  rule they use the first voice GET /v1/voices?language= lists: Antoine (fr), Abel (es).
- Price: API balance $0.06 / 1,000 characters (launch price until 2026-12-18, then $0.08).
- Cloning: POST /v1/voices/add (multipart, one WAV/MP3 <= 4 MB). Up to 14 s the recording is used
  whole; longer, Voicefinder picks a ~12 s clip itself. The 10 s cut (9.85 s) is the longest cut
  used whole (the 15 s cut is 14.09 s), sent with its transcript. See clone.py.
"""

from ..core import Adapter


class VoicefinderVoyager1(Adapter):
    slug = "voicefinder_voyager1"
    provider = "voicefinder"
    vendor = "Voicefinder"
    model = "voyager-1"
    langs = {"en", "fr", "es"}
    streaming = False
    fmt = "wav"
    sr = 48000
    params = {"output_format": "wav"}
    keep_bytes = True
    min_interval = 6.5  # free tier: 10 generations/min per key; stay clear of strikes

    VOICES = {"en": "f6abdab0-2524-43d1-92e2-b39486f39fbe",  # quickstart voice "Agatha"
              "fr": "5c3ad2c8-4772-4162-a04e-dc159d4e330c",  # "Antoine", French (European)
              "es": "b66ed819-adca-4107-9bd4-bc142a4882c0"}  # "Abel", Spanish (Spain)

    def voice(self, lang):
        return self.VOICES[lang]

    def request(self, text, lang, voice):
        return dict(method="POST",
                    url=f"https://api.voicefinder.io/v1/text-to-speech/{voice}",
                    headers={"x-api-key": self.key},
                    json={"text": text, "language": lang, "model_id": self.model,
                          "output_format": "wav"})

    def echo(self, headers, body_meta):
        keep = ("x-request-id", "x-character-count", "x-audio-duration-s", "x-output-format",
                "x-model-id", "x-model-id-warning", "x-language", "x-language-warning",
                "x-watermarked", "x-c2pa-signed")
        return {k: v for k, v in headers.items() if k.lower() in keep}
