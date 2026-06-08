"""Public objective-metric scoring for the tts-bench leaderboard.

Scores published clips with three standard metrics:
  UTMOS (naturalness), WER (intelligibility), SIM (cloning fidelity).

These are standard, publishable metrics (SpeechMOS / Whisper / UniSpeech-SAT).
This package is entirely separate from the private NAQ R&D in naq_lab/ and must
never import from or reference it.
"""
