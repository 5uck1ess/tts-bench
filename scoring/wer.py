"""WER intelligibility scorer — Whisper-large-v3 ASR vs the intended prompt text.

A failure-detector: catches dropped/garbled words, gibberish, wrong language,
runaways. It does NOT finely rank already-intelligible models. Lower = better
(0 = perfect transcription).

WhisperProcessor + WhisperForConditionalGeneration.generate directly — NOT the
transformers ASR pipeline(), which hard-requires torchcodec/FFmpeg and dies in
this venv ("Could not load libtorchcodec"). The feature extractor consumes a
librosa-decoded numpy array, bypassing the codec. (POC-confirmed.)

Import note: `librosa` and `numpy` are imported lazily inside `transcribe()`
so that the pure helpers (`normalize_fr`, `wer_value`) remain importable in
environments where only `jiwer` is installed (e.g. test runners without the
full ML stack).
"""

import re

import jiwer

_LANG_FULL = {"en": "english", "fr": "french"}


def normalize_fr(text):
    """Light FR normalize: lowercase, drop punctuation, keep accents, squeeze spaces."""
    s = text.lower()
    s = re.sub(r"[^\w\sàâäéèêëïîôöùûüÿçœæ]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def wer_value(ref, hyp):
    """jiwer WER of already-normalized strings; None if ref is empty."""
    if not ref.strip():
        return None
    return float(jiwer.wer(ref, hyp))


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
        return normalize_fr(text)

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
        hyp = self.transcribe(wav_path, lang)
        return wer_value(self._normalize(ref_text, lang), self._normalize(hyp, lang))
