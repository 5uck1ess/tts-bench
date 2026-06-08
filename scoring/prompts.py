"""The 5 canonical bench prompts, mirrored from bench.PROMPTS.

Kept in sync by hand (they change rarely). Mirrored rather than imported so the
lightweight scoring venv never pulls bench.py's heavy transitive deps.
"""

# (id, lang, text) — MUST match bench.PROMPTS exactly.
PROMPTS = [
    (1, "en", "Open the browser and read my email."),
    (2, "en", "I'll start a new git branch, push the changes, and open a pull request when the tests pass."),
    (3, "en",
     "The Parakeet TDT zero point six billion parameter model achieves "
     "one point six nine percent word error rate on LibriSpeech test-clean, "
     "beating Whisper Large V3 at two point seven percent while running at "
     "over two thousand times realtime on a single GPU."),
    (4, "en", "Run pytest tests slash test underscore voice dot py with verbose flag and capture flag set to no."),
    (5, "fr", "Bonjour, je m'appelle Cicero et je vais vous aider avec votre code aujourd'hui."),
]

# prompt_id (str) -> (lang, text)
PROMPT_BY_ID = {str(pid): (lang, text) for pid, lang, text in PROMPTS}
