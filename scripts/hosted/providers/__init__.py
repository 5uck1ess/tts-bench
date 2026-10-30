from .elevenlabs import ElevenV4, ElevenV4Turbo
from .giggy import GiggyTTS
from .rime import RimeCoda

ADAPTERS = {a.slug: a for a in (ElevenV4, ElevenV4Turbo, RimeCoda, GiggyTTS)}
