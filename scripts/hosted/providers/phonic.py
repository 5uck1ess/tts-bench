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
- Response validation (anything else is an error, never a saved clip): the body must parse as
  JSON (a real parser, so escapes and whitespace follow RFC 8259) into exactly one object with
  one "audio" key (duplicates rejected) holding one canonical base64 string of headerless
  PCM16. The body is buffered in one bytearray and capped at MAX_BODY, which bounds both memory
  and the parse + decode work done after the last byte arrives (core.guard checks the call
  deadline per chunk, so a body finishing just under it is still reported as over).
- Decoded audio that is really a container file (WAV/RF64/BW64/RIFX, Wave64, AIFF, Ogg, FLAC,
  ID3-tagged MP3) is rejected: core.to_wav would read a WAV at its embedded rate, or save the
  header as 44.1 kHz samples. Each signature is matched on its STRUCTURE (form type, version and
  size fields), not a 3-4 byte prefix, because headerless PCM can start with any 4 bytes
  (samples 17481, 51 are b"ID3\\x00"); a false positive on the 8-16 constrained bytes checked
  here is far below one in a billion clips.
"""

import base64
import binascii
import json

from ..core import Adapter

# The longest bench clip is ~18 s = ~1.6 MB of 44.1 kHz PCM16, ~2.1 MB as base64. 4 MiB covers
# ~47 s of audio; parsing + decoding a body that size takes tens of milliseconds.
MAX_BODY = 4 * 1024 * 1024

_W64_RIFF = bytes.fromhex("72696666 2e91cf11 a5d628db 04c10000".replace(" ", ""))


def _is_container(b):
    """True when b starts with a recognised audio container header (structure-checked)."""
    if b[:4] in (b"RIFF", b"RIFX", b"RF64", b"BW64") and b[8:12] == b"WAVE":
        return True
    if b[:16] == _W64_RIFF:                                   # Sony Wave64 'riff' GUID
        return True
    if b[:4] == b"FORM" and b[8:12] in (b"AIFF", b"AIFC"):
        return True
    if b[:4] == b"OggS" and b[4] == 0 and b[5] in (0, 2):     # version 0, BOS page
        return True
    if b[:4] == b"fLaC" and b[4] & 0x7F == 0 and b[5:8] == b"\x00\x00\x22":  # STREAMINFO, len 34
        return True
    if (b[:3] == b"ID3" and len(b) >= 10 and b[3] in (2, 3, 4) and b[4] != 0xFF
            and all(x < 0x80 for x in b[6:10])):               # ID3v2: version, syncsafe size
        return True
    return False


def _single_key(pairs):
    keys = [k for k, _ in pairs]
    if keys != ["audio"]:
        raise ValueError(f"keys {keys[:5]}")
    return dict(pairs)


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
        body = bytearray()  # one growing buffer: tiny HTTP chunks don't each cost an object
        for chunk in raw_iter:
            if len(body) + len(chunk) > MAX_BODY:
                raise RuntimeError(f"{self.slug}: response over {MAX_BODY} bytes")
            body += chunk
        try:
            obj = json.loads(body, object_pairs_hook=_single_key)
        except (ValueError, UnicodeDecodeError) as e:  # JSONDecodeError is a ValueError
            raise RuntimeError(f"{self.slug}: response is not exactly {{\"audio\": <base64>}} "
                               f"({e}): {bytes(body[:200])!r}") from None
        b64 = obj.get("audio") if isinstance(obj, dict) else None
        if not isinstance(b64, str) or not b64:
            raise RuntimeError(f"{self.slug}: response is not exactly {{\"audio\": <base64>}}: "
                               f"{bytes(body[:200])!r}")
        try:
            pcm = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError) as e:
            raise RuntimeError(f"{self.slug}: audio is not valid base64: {e}") from None
        # The docs promise ONE base64 string. Lenient decoders stop at the first padding and drop
        # the rest silently, so require that the bytes re-encode to exactly what was sent.
        if base64.b64encode(pcm).decode("ascii") != b64:
            raise RuntimeError(f"{self.slug}: audio is not one canonical base64 string")
        if _is_container(pcm):
            raise RuntimeError(f"{self.slug}: got an audio container ({pcm[:4]!r}), "
                               "requested headerless PCM16")
        if len(pcm) % 2:
            raise RuntimeError(f"{self.slug}: odd byte count {len(pcm)} for PCM16")
        yield pcm
