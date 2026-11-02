"""Cloning lens: how each vendor makes a voice from the reference recording (docs/hosted.md).

Read from each vendor's API docs 2026-11-01. Every vendor gets the LONGEST cut it accepts
(Tym 2026-11-01: some products need more audio for a good clone, so one shared length is unfair);
MIN_REC records each vendor's documented minimum / recommended seconds for the page.

No API cloning on our route (listed on the page, not run): Rime (enterprise training only),
Giggy + AssemblyAI (dashboard only), Deepgram (none), Murf (Enterprise plan), OpenAI (sales-gated),
Microsoft MAI (Limited Access approval + its own consent statement), MiniMax via OpenRouter (no
cloning on that route), Speechify (needs a live consent recording: scripts/hosted/speechify_clone).
"""

import base64
import time

from .batch_a import CartesiaSonic36, FishDrama3, InworldTTS2, Voxtral3TTS
from .batch_b import SmallestLightning31Pro, SonioxTTS2, StepAudio3TTS, XaiTTS
from .batch_c import HeyGenVoice1, VuiLuna
from .batch_d import AsyncFlash15, BlandSpeech3, GradiumTTS
from .elevenlabs import ElevenV4, ElevenV4Turbo

NAME = "tts-bench reference voice"
WAV = "reference.wav"

# slug -> (documented min s, recommended s or range, as text). "—" = not stated.
MIN_REC = {
    "eleven_v4": ("10", "60–120"), "eleven_v4_turbo": ("10", "60–120"),
    "cartesia_sonic36": ("10", "≤60"), "inworld_tts2": ("3", "10–30"),
    "fish_drama3": ("—", "10–30"), "voxtral3_tts": ("a few", "—"), "xai_tts": ("—", "≥90"),
    "soniox_tts2": ("—", "≤120"), "stepaudio3_tts": ("5", "5–10"),
    "smallest_lightning31pro": ("5", "5–15"), "vui_luna": ("3", "3–60"),
    "heygen_voice1": ("—", "60–180"), "bland_speech3": ("—", "≈10"), "gradium_tts": ("10", "≥10"),
    "async_flash15": ("3", "—"), "gemini38_flash_tts": ("10", "10–30"),
    "speechify_simba35": ("10", "10–30"), "assemblyai_tts1": ("10", "10–30"),
}


def _check(r, slug):
    if r.status_code >= 300:
        raise RuntimeError(f"{slug} clone: HTTP {r.status_code}: {r.text[:400]}")
    return r.json()


def _poll(client, slug, url, headers, ready, every=2.0, limit=180):
    t = time.monotonic()
    while time.monotonic() - t < limit:
        j = _check(client.get(url, headers=headers), slug)
        state = ready(j)
        if state is True:
            return j
        if state is False:
            raise RuntimeError(f"{slug} clone failed: {str(j)[:300]}")
        time.sleep(every)
    raise RuntimeError(f"{slug} clone not ready after {limit}s")


def _setup(cls, cut, fn, langs=None):
    cls.clone_kind, cls.clone_cut, cls.create_clone = "voice_id", cut, fn
    if langs:
        cls.clone_langs = langs


# ElevenLabs IVC: total >=10 s, 1-2 min recommended -> full (113 s).
def _eleven(self, client, ref):
    r = client.post("https://api.elevenlabs.io/v1/voices/add",
                    headers={"xi-api-key": self.key},
                    data={"name": NAME, "remove_background_noise": "false"},
                    files={"files": (WAV, ref["wav"], "audio/wav")})
    return _check(r, self.slug)["voice_id"]


_setup(ElevenV4, "full", _eleven)
_setup(ElevenV4Turbo, "full", _eleven)


# Cartesia: 10-60 s used on Sonic 3.6 -> 58 s cut. One clone (English); fr/es sent as `language`.
def _cartesia(self, client, ref):
    r = client.post("https://api.cartesia.ai/voices/clone",
                    headers={"Authorization": f"Bearer {self.key}",
                             "Cartesia-Version": "2026-08-14"},
                    data={"name": NAME, "language": "en"},
                    files={"clip": (WAV, ref["wav"], "audio/wav")})
    return _check(r, self.slug)["id"]


_setup(CartesiaSonic36, "60", _cartesia)


# Inworld: samples over 30 s are trimmed -> 23 s cut; transcript optional (TTS-2 ignores it).
def _inworld(self, client, ref):
    r = client.post("https://api.inworld.ai/voices/v1/voices:clone",
                    headers={"Authorization": f"Basic {self.key}"},
                    json={"displayName": NAME, "languageCode": "en-US",
                          "voiceSamples": [{"audioData": base64.b64encode(ref["wav"]).decode(),
                                            "transcription": ref["text"]}],
                          "audioProcessingConfig": {"removeBackgroundNoise": False}})
    j = _check(r, self.slug)
    return (j.get("voice") or j).get("voiceId") or j.get("voiceId")


_setup(InworldTTS2, "30", _inworld)


# Fish: create a fast model; 10-30 s recommended -> 23 s cut, with transcript.
def _fish(self, client, ref):
    r = client.post("https://api.fish.audio/model", headers={"Authorization": f"Bearer {self.key}"},
                    data={"type": "tts", "title": NAME, "train_mode": "fast",
                          "visibility": "private", "texts": ref["text"]},
                    files={"voices": (WAV, ref["wav"], "audio/wav")})
    return _check(r, self.slug)["_id"]


_setup(FishDrama3, "30", _fish)


# Mistral: "a few seconds to a few minutes", but the sample must be <=10 MB -> 58 s cut.
def _mistral(self, client, ref):
    r = client.post("https://api.mistral.ai/v1/audio/voices",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={"name": NAME, "sample_audio": base64.b64encode(ref["wav"]).decode(),
                          "sample_filename": WAV, "languages": ["en", "fr", "es"]})
    return _check(r, self.slug)["id"]


_setup(Voxtral3TTS, "60", _mistral)


# xAI: <=120 s, >=90 recommended -> full. Docs: API endpoint needs an Enterprise plan.
def _xai(self, client, ref):
    r = client.post("https://api.x.ai/v1/custom-voices",
                    headers={"Authorization": f"Bearer {self.key}"},
                    data={"name": NAME, "language": "en"},
                    files={"file": (WAV, ref["wav"], "audio/wav")})
    j = _check(r, self.slug)
    return j.get("voice_id") or j.get("id")


_setup(XaiTTS, "full", _xai)


# Soniox: <=2 min -> full; async, poll until ready.
def _soniox(self, client, ref):
    h = {"Authorization": f"Bearer {self.key}"}
    j = _check(client.post("https://api.soniox.com/v1/voices", headers=h, data={"name": NAME},
                           files={"file": (WAV, ref["wav"], "audio/wav")}), self.slug)
    vid = j.get("id") or j.get("voice_id")

    def ready(v):
        m = [x for x in v.get("models") or [] if x.get("model") == self.model]
        s = str((m[0] if m else v).get("status") or "").lower()
        return True if s == "ready" else (False if s in ("failed", "error") else None)
    _poll(client, self.slug, f"https://api.soniox.com/v1/voices/{vid}", h, ready)
    return vid


_setup(SonioxTTS2, "full", _soniox)


# StepFun: 5-10 s -> 10 s cut. Upload to Files, then create; clones speak zh/en only.
def _stepfun(self, client, ref):
    h = {"Authorization": f"Bearer {self.key}"}
    f = _check(client.post("https://api.stepfun.ai/v1/files", headers=h,
                           data={"purpose": "storage"},
                           files={"file": (WAV, ref["wav"], "audio/wav")}), self.slug)
    j = _check(client.post("https://api.stepfun.ai/v1/audio/voices", headers=h,
                           json={"model": self.model, "file_id": f["id"], "text": ref["text"]}),
               self.slug)
    return j["id"]


_setup(StepAudio3TTS, "10", _stepfun, langs={"en"})


# Smallest: 5-15 s, <5 MB -> 12 s cut; model must be lightning-v3.1-pro for a Pro voice.
def _smallest(self, client, ref):
    r = client.post("https://api.smallest.ai/waves/v1/voice-cloning",
                    headers={"Authorization": f"Bearer {self.key}"},
                    data={"displayName": NAME, "model": "lightning-v3.1-pro", "language": "en"},
                    files={"file": (WAV, ref["wav"], "audio/wav")})
    j = _check(r, self.slug)
    d = j.get("data") or j
    return d.get("voiceId") or d.get("voice_id") or d.get("id")


_setup(SmallestLightning31Pro, "15", _smallest)


# VUI: 3-60 s -> 58 s cut. Upload, then save under our own voice id. English-only model.
def _vui(self, client, ref):
    h = {"X-API-Key": self.key}
    up = _check(client.post("https://api.vuilabs.ai/v1/files/upload", headers=h,
                            data={"scene": "audio"},
                            files={"file": (WAV, ref["wav"], "audio/wav")}), self.slug)
    d = up.get("data") or up
    key = d.get("object_key") or d.get("key") or d.get("url")
    vid = "ttsbench_ref"
    _check(client.post("https://api.vuilabs.ai/v1/voice/save", headers=h,
                       json={"voice_id": vid, "ref_audio": key}), self.slug)
    return vid


_setup(VuiLuna, "60", _vui)


# HeyGen: first 3 min used, 1-3 min ideal -> full (base64 <16 MB). 202, then poll until ACTIVE.
def _heygen(self, client, ref):
    h = {"X-Api-Key": self.key}
    j = _check(client.post("https://api.heygen.com/v3/models/audio/voices", headers=h,
                           json={"mode": "instant", "name": NAME, "language": "en",
                                 "audio": [{"type": "base64", "media_type": "audio/x-wav",
                                            "data": base64.b64encode(ref["wav"]).decode()}]}),
               self.slug)
    d = j.get("data") or j
    vid = d.get("voice_id") or d.get("id")

    def ready(v):
        s = str((v.get("data") or v).get("status", "")).upper()
        return True if s == "ACTIVE" else (False if s in ("FAILED", "ERROR") else None)
    _poll(client, self.slug, f"https://api.heygen.com/v3/models/audio/voices/{vid}", h, ready)
    return vid


_setup(HeyGenVoice1, "full", _heygen)


# Bland: 1 file, <=10 MB -> 58 s cut (the full cut is 10.8 MB).
def _bland(self, client, ref):
    r = client.post("https://api.bland.ai/v1/voices/clone",
                    headers={"authorization": self.key},
                    data={"name": "ttsbench-ref", "isBTTS_V3": "true"},
                    files={"audio_samples": (WAV, ref["wav"], "audio/wav")})
    j = _check(r, self.slug)
    return (j.get("data") or j).get("voice_id")


_setup(BlandSpeech3, "60", _bland)


# Gradium: >=10 s, no max -> full; enhance=false for a raw clone.
def _gradium(self, client, ref):
    r = client.post("https://api.gradium.ai/api/voices/", headers={"x-api-key": self.key},
                    data={"name": NAME, "language": "en", "enhance": "false"},
                    files={"audio_file": (WAV, ref["wav"], "audio/wav")})
    j = _check(r, self.slug)
    return j.get("uid") or j.get("voice_id") or j.get("id")


_setup(GradiumTTS, "full", _gradium)


# Async: no max stated -> full; enhance=false. Max 3 custom voices per account.
def _async(self, client, ref):
    r = client.post("https://api.async.com/voices/clone",
                    headers={"x-api-key": self.key, "version": "v1"},
                    data={"name": NAME, "enhance": "false"},
                    files={"audio": (WAV, ref["wav"], "audio/wav")})
    j = _check(r, self.slug)
    return j.get("id") or j.get("voice_id") or (j.get("voice") or {}).get("id")


_setup(AsyncFlash15, "full", _async)


# Gemini voice replication: 10-30 s source (24 kHz mono recommended -> the 23 s 24k cut) plus a
# consent recording of the locale's fixed phrase, same mic and room (speaker-match checked).
def _gemini(self, client, ref):
    consent = (ref["path"].parent / "tym_consent_google.wav").read_bytes()
    b64 = lambda b: base64.b64encode(b).decode()
    r = client.post("https://generativelanguage.googleapis.com/v1beta/voices",
                    headers={"x-goog-api-key": self.key},
                    json={"store": True, "voice": {
                        "model": self.model, "type": "replicated", "display_name": NAME,
                        "replicated": {
                            "source_audio": {"mime_type": "audio/wav", "data": b64(ref["wav"])},
                            "consent_audio": {"mime_type": "audio/wav", "data": b64(consent)}}}})
    j = _check(r, self.slug)
    return j.get("id") or j.get("name", "").split("/")[-1]


from .batch_a import Gemini38FlashTTS  # noqa: E402

_setup(Gemini38FlashTTS, "30", _gemini)


# Speechify: voice created by scripts/hosted/speechify_clone.py (live consent); this only lets
# prepare_clone reuse the id it recorded. 10-30 s, <5 MB -> 23 s cut.
from .batch_a import SpeechifySimba35  # noqa: E402


def _speechify(self, client, ref):
    raise RuntimeError("run scripts.hosted.speechify_clone (live consent) first")


_setup(SpeechifySimba35, "30", _speechify)


# Palabra: instant clone, >=30 s, <=10 MB, one sample (platform.palabra.ai/docs/assets/voices,
# read 2026-11-02) -> full cut. The 113 s WAV is 10.8 MB, so it goes up as lossless FLAC (same
# samples). Flow: POST metadata -> POST the file to the returned pre-signed S3 form -> poll ready.
from .palabra import PalabraTTS  # noqa: E402


def _check_upload_url(url, slug):
    """The pre-signed upload target comes from the vendor's response; only POST the reference to
    an HTTPS S3 bucket host (Palabra's docs show palabra-prod-web-cdn.s3.amazonaws.com), never to
    an IP literal, localhost or LAN name a bad response could name."""
    from urllib.parse import urlsplit
    u = urlsplit(url)
    host = (u.hostname or "").lower()
    if u.scheme != "https" or not (host == "s3.amazonaws.com" or host.endswith(".amazonaws.com")):
        raise RuntimeError(f"{slug} clone: refusing upload to unexpected URL host {host!r}")


def _palabra(self, client, ref):
    import io
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(ref["wav"]), dtype="int16")
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="FLAC")
    flac = buf.getvalue()
    if len(flac) > 10 * 1024 * 1024:
        raise RuntimeError(f"{self.slug} clone: FLAC is {len(flac)} bytes, over the 10 MB cap")
    h = {"Authorization": f"Bearer {self.key}"}
    # The live API wants the documented payload wrapped in {"data": ...} (422 "Field required:
    # body.data" otherwise) and answers in the same envelope.
    j = _check(client.post("https://api.palabra.ai/saas/voice/clone", headers=h, json={"data": {
        "name": NAME, "samples": [{"filename": "reference.flac", "mime_type": "audio/flac",
                                   "denoise": False, "speech_normalization": False,
                                   "lang_code": "en"}]}}), self.slug)
    j = j.get("data") or j
    vid, s = j["voice_id"], j["samples"][0]
    _check_upload_url(s["url"], self.slug)
    up = client.post(s["url"], data=s["form_data"],
                     files={"file": ("reference.flac", flac, "audio/flac")})
    if up.status_code >= 300:
        raise RuntimeError(f"{self.slug} clone upload: HTTP {up.status_code}: {up.text[:300]}")
    _poll(client, self.slug, f"https://api.palabra.ai/saas/voice/m/{vid}", h,
          lambda r: {"ready": True, "failed": False}.get((r.get("data") or {})
                                                       .get("processing_status")),
          every=5.0, limit=600)
    return vid


_setup(PalabraTTS, "full", _palabra)
MIN_REC["palabra_tts"] = ("30", "—")


# Resemble: voice made in the console upload flow (5 s min, 1 min max, 25 MB, Enhance Audio off) -> 58 s cut; this only lets
# prepare_clone reuse the uuid recorded in _private/clones.json.
from .resemble import ResembleUltra  # noqa: E402


def _resemble(self, client, ref):
    raise RuntimeError("make the clone in Resemble's console and record its uuid in clones.json")


_setup(ResembleUltra, "60", _resemble)
MIN_REC["resemble_ultra"] = ("5", "≤60")
