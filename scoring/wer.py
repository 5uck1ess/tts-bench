"""WER intelligibility scorer — Whisper-large-v3 ASR vs the intended prompt text.

A failure-detector: catches dropped/garbled words, gibberish, wrong language,
runaways. It does NOT finely rank already-intelligible models. Lower = better
(0 = perfect transcription).

WhisperProcessor + WhisperForConditionalGeneration.generate directly — NOT the
transformers ASR pipeline(), which hard-requires torchcodec/FFmpeg and dies in
this venv ("Could not load libtorchcodec"). The feature extractor consumes a
librosa-decoded numpy array, bypassing the codec. (POC-confirmed.)

Import note: `librosa` and `numpy` are imported lazily inside `transcribe()`
so that the pure helpers (`normalize_intl`, `wer_value`) remain importable in
environments where only `jiwer` is installed (e.g. test runners without the
full ML stack).
"""

import difflib
import re

import jiwer

_LANG_FULL = {"en": "english", "fr": "french", "es": "spanish"}


def normalize_intl(text):
    """Light non-English normalize: lowercase, drop punctuation, keep accented
    letters, squeeze spaces.

    Language-agnostic despite the FR-looking character class: `\\w` is Unicode-aware
    in Python 3, so every accented letter already survives (ñ, á, í, ó, ú, ü for
    Spanish included) and the explicit list is belt-and-braces. What it does strip
    is punctuation — which for Spanish means the inverted ¿ and ¡, dropped from
    the reference and the hypothesis alike, so they never move the WER.
    """
    s = text.lower()
    s = re.sub(r"[^\w\sàâäéèêëïîôöùûüÿçœæ]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


# Pre-2026-09 name, kept so external callers/notebooks don't break.
normalize_fr = normalize_intl


def wer_value(ref, hyp):
    """jiwer WER of already-normalized strings; None if ref is empty."""
    if not ref.strip():
        return None
    return float(jiwer.wer(ref, hyp))


# "head" flag: the clip's transcript only starts lining up with the prompt at word
# HEAD_MIN_WORDS or later, i.e. the opening words were never spoken. Catches the
# 2026-10 truncated-reference-transcript failure (Ming p1 "email.", cloners eating
# the first ~3 s) that a WER number alone blurs into "a bit wrong". A misheard
# first word (a=1) is NOT flagged, and neither is a misheard opening: if the
# transcript has at least HEAD_MISHEARD_FRAC as many characters before the first
# match as the prompt has, the words were spoken and garbled ("run pytest test" ->
# "paytesttest"), not dropped.
HEAD_MIN_WORDS = 2
HEAD_MIN_FRAC = 0.15
HEAD_MISHEARD_FRAC = 0.5


def head_dropped(ref, hyp):
    """True if `hyp` (normalized) is missing the opening words of `ref` (normalized).

    Aligns the word lists; the first matching run of words must start at ref index
    >= HEAD_MIN_WORDS and >= HEAD_MIN_FRAC of the prompt, with less than
    HEAD_MISHEARD_FRAC of the skipped text's length transcribed before it. A hyp that matches nothing
    is garbage, not a dropped head -- WER already says so -- and is not flagged.
    """
    r, h = ref.split(), hyp.split()
    if not r or not h:
        return False
    blocks = [b for b in difflib.SequenceMatcher(None, r, h, autojunk=False).get_matching_blocks()
              if b.size]
    if not blocks:
        return False
    a, b = blocks[0].a, blocks[0].b
    if a < HEAD_MIN_WORDS or a / len(r) < HEAD_MIN_FRAC:
        return False
    return len(" ".join(h[:b])) < HEAD_MISHEARD_FRAC * len(" ".join(r[:a]))


class WerScorer:
    name = "wer"

    def __init__(self, device=None, model_id="openai/whisper-large-v3"):
        import torch
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model_id = model_id
        self._proc = None
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            self._proc = WhisperProcessor.from_pretrained(self.model_id)
            self._model = WhisperForConditionalGeneration.from_pretrained(
                self.model_id, dtype=self.dtype).to(self.device).eval()
        return self._proc, self._model

    def _normalize(self, text, lang):
        proc, _ = self._load()
        if lang == "en":
            # Whisper's bundled EnglishTextNormalizer (numbers, casing, punctuation).
            return proc.tokenizer.normalize(text)
        return normalize_intl(text)

    def transcribe(self, wav_path, lang):
        import librosa
        import numpy as np
        proc, model = self._load()
        y, _ = librosa.load(wav_path, sr=16000, mono=True)
        feats = proc(np.asarray(y, dtype=np.float32), sampling_rate=16000,
                     return_tensors="pt").input_features.to(self.device, self.dtype)
        with self._torch.no_grad():
            ids = model.generate(feats, language=_LANG_FULL.get(lang, "english"),
                                 task="transcribe")
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def score(self, wav_path, ref_text, lang):
        return self.score_detail(wav_path, ref_text, lang)[0]

    def score_detail(self, wav_path, ref_text, lang):
        """(wer, flags) from one transcription; flags is a list like ["head"]."""
        hyp = self.transcribe(wav_path, lang)
        ref_n, hyp_n = self._normalize(ref_text, lang), self._normalize(hyp, lang)
        return wer_value(ref_n, hyp_n), (["head"] if head_dropped(ref_n, hyp_n) else [])
