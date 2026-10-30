from .batch_a import (CartesiaSonic36, FishDrama3, Gemini38FlashTTS, InworldTTS2,
                      SpeechifySimba35, Voxtral3TTS)
from .batch_b import (AssemblyTTS1, DeepgramFluxTTS, SmallestLightning31Pro, SonioxTTS2,
                      StepAudio3TTS, XaiTTS)
from .batch_c import (HeyGenVoice1, MaiVoice21, MiniMaxSpeech28HD, MurfFalcon21, OpenAIMiniTTS,
                      VuiLuna)
from .batch_d import AsyncFlash15, BlandSpeech3, GradiumTTS, SpeechmaticsTTS
from .elevenlabs import ElevenV4, ElevenV4Turbo
from .giggy import GiggyTTS
from .openai_realtime import OpenAIRealtime21
from .rime import RimeCoda
from .ws import AssemblyTTS1WS, CartesiaSonic36WS

# Cartesia and AssemblyAI run on their WebSocket transport (faster first speech, see ws.py).
ADAPTERS = {a.slug: a for a in (
    ElevenV4, ElevenV4Turbo, RimeCoda, GiggyTTS,
    CartesiaSonic36WS, Gemini38FlashTTS, InworldTTS2, SpeechifySimba35, FishDrama3, Voxtral3TTS,
    XaiTTS, AssemblyTTS1WS, SonioxTTS2, StepAudio3TTS, SmallestLightning31Pro, DeepgramFluxTTS,
    MaiVoice21, VuiLuna, OpenAIMiniTTS, MurfFalcon21, HeyGenVoice1, MiniMaxSpeech28HD,
    OpenAIRealtime21, BlandSpeech3, GradiumTTS, AsyncFlash15, SpeechmaticsTTS)}
