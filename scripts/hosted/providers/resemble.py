"""Resemble Ultra over HTTP streaming (docs.resemble.ai, read 2026-11-02).

- Resemble Ultra (`resemble-ultra`) is the only live TTS model; every Chatterbox / Enhanced hosted
  model is end-of-life (getting-started/model-versions). The model follows the voice: there is no
  model field, so `model` here is the version the voice reports, for the sidecar only.
- Endpoint: POST https://f.cluster.resemble.ai/stream (voice-generation/text-to-speech/streaming-http).
  The WebSocket route is Business-plan only, so this row is HTTP.
- Body is a chunked single-channel WAV; PCM_16 at 48 kHz requested. core already strips a RIFF
  header in front of pcm16 for both the clip and first-speech timing.
- Voice: Luma (819fcc57), the first voice GET /api/v2/voices lists. The docs' synthesis example
  voice, Ember (55592656, a "professional" voice), is refused on a funded Flex wallet with a
  misleading 401 "You've consumed all your credits" (2026-11-02, wallet $10.00), while library
  "rapid" voices synthesize. Ultra is multilingual ("available in all languages" in the console)
  and the request has no language field, so the same voice reads fr/es.
- Price: Flex plan, $0.0005 per second of audio generated (billing API subscription, 2026-11-02).
- Cloning: made in Resemble's console ("Voice Clone", needs 30 s of audio); the voice uuid goes in
  hosted/_private/clones.json like xAI/Speechify.
"""

from ..core import Adapter


class ResembleUltra(Adapter):
    slug = "resemble_ultra"
    provider = "resemble"
    vendor = "Resemble AI"
    model = "resemble-ultra"
    langs = {"en", "fr", "es"}
    fmt = "pcm16"
    sr = 48000
    params = {"precision": "PCM_16", "sample_rate": 48000}

    def voice(self, lang):
        return "819fcc57"

    def request(self, text, lang, voice):
        return dict(method="POST", url="https://f.cluster.resemble.ai/stream",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={"voice_uuid": voice, "data": text, "precision": "PCM_16",
                          "sample_rate": self.sr})
