from ..core import Adapter

# Quickstart voice "George" (elevenlabs.io/docs quickstart); multilingual, so one voice for all langs.
QUICKSTART_VOICE = "JBFqnCBsd6RMkjVDRZzb"


class ElevenV4(Adapter):
    slug = "eleven_v4"
    provider = "elevenlabs"
    vendor = "ElevenLabs"
    model = "eleven_v4"
    langs = {"en", "fr", "es"}
    fmt = "pcm16"
    # pcm_44100 is Pro-plan only (403 subscription_required on Tym's plan, 2026-10-29).
    sr = 24000
    params = {"output_format": "pcm_24000"}

    def voice(self, lang):
        return QUICKSTART_VOICE

    def request(self, text, lang, voice):
        return dict(method="POST",
                    url=f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream",
                    params={"output_format": f"pcm_{self.sr}"},
                    headers={"xi-api-key": self.key},
                    json={"text": text, "model_id": self.model})


class ElevenV4Turbo(ElevenV4):
    slug = "eleven_v4_turbo"
    model = "eleven_v4_turbo"
