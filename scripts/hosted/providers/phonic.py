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
- Response validation (anything else is an error, never a saved clip): the body must be exactly
  the documented object, one "audio" field holding one canonical base64 string of headerless
  PCM16. It is matched with one linear regex instead of json.loads, so extra or duplicate fields
  are rejected outright and a body of many fields can't spend parse time past the call deadline;
  the body is also capped at MAX_BODY. Decoded bytes that start with a known container signature
  (WAV/RF64/BW64/RIFX/AIFF/Ogg/FLAC/ID3-tagged MP3) are rejected too: core.to_wav would otherwise read a WAV
  at its embedded rate, or save any container's header as 44.1 kHz samples.
"""

import base64
import binascii
import re

from ..core import Adapter

# The longest bench clip is ~18 s = ~1.6 MB of 44.1 kHz PCM16, ~2.1 MB as base64. 16 MiB covers
# ~95 s of audio and keeps the regex + base64 work far below a second.
MAX_BODY = 16 * 1024 * 1024
_BODY = re.compile(rb'\A\s*\{\s*"audio"\s*:\s*"([A-Za-z0-9+/=\\]*)"\s*\}\s*\Z')
_CONTAINERS = (b"RIFF", b"RIFX", b"RF64", b"BW64", b"FORM", b"OggS", b"fLaC", b"ID3")


def _looks_like_container(pcm):
    # Fixed file signatures only. A bare MPEG frame sync (0xFF 0xEx) is NOT checked: as little-endian
    # PCM16 it is just a small negative first sample, common in real clips.
    return pcm.startswith(_CONTAINERS)  # prefix match: "ID3" is 3 bytes, the rest 4


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
        parts, size = [], 0
        for chunk in raw_iter:
            size += len(chunk)
            if size > MAX_BODY:
                raise RuntimeError(f"{self.slug}: response over {MAX_BODY} bytes")
            parts.append(chunk)
        body = b"".join(parts)
        m = _BODY.match(body)
        if not m or not m.group(1):
            raise RuntimeError(f"{self.slug}: response is not exactly {{\"audio\": <base64>}}: "
                               f"{body[:200]!r}")
        b64 = m.group(1).replace(b"\\/", b"/")  # JSON may escape "/" as "\/"
        try:
            pcm = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError) as e:
            raise RuntimeError(f"{self.slug}: audio is not valid base64: {e}") from None
        # The docs promise ONE base64 string. Lenient decoders stop at the first padding and drop
        # the rest silently, so require that the bytes re-encode to exactly what was sent.
        if base64.b64encode(pcm) != b64:
            raise RuntimeError(f"{self.slug}: audio is not one canonical base64 string")
        if _looks_like_container(pcm):
            raise RuntimeError(f"{self.slug}: got an audio container ({pcm[:4]!r}), "
                               "requested headerless PCM16")
        if len(pcm) % 2:
            raise RuntimeError(f"{self.slug}: odd byte count {len(pcm)} for PCM16")
        yield pcm
