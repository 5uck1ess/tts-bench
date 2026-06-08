"""SIM cloning-fidelity scorer — canonical seed-tts-eval UniSpeech-SAT.

Speaker-embedding cosine between a cloned clip and the lens reference voice.
Higher = more similar (−1..1). Cloning lens only.

Uses the canonical seed-tts-eval speaker-verification model `wavlm_large` with
the `wavlm_large_finetune.pth` checkpoint, so numbers are comparable to published
TTS papers. The checkpoint is NOT on HuggingFace and the model code lives in
seed-tts-eval's thirdparty/UniSpeech — both are fetched at install time
(install.sh / install.ps1). If either is absent, score() raises with the exact
fetch instructions.

Import note: `librosa` and `numpy` are imported lazily inside `_embed()` so that
the module remains importable in environments where only the guard check is needed
(e.g. Step-2 verification without the full ML stack). This mirrors the wer.py
lazy-import pattern.
"""

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CKPT = os.path.join(_REPO, "scoring", "checkpoints", "wavlm_large_finetune.pth")
_UNISPEECH = os.path.join(_REPO, "scoring", "thirdparty", "UniSpeech",
                          "downstreams", "speaker_verification")
_FETCH_HELP = (
    "Canonical UniSpeech-SAT SIM assets missing.\n"
    f"  checkpoint expected at: {_CKPT}\n"
    f"  model code expected at: {_UNISPEECH}\n"
    "Install them via the scoring venv stanza (install.sh / install.ps1), or fetch manually:\n"
    "  git clone https://github.com/microsoft/UniSpeech scoring/thirdparty/UniSpeech\n"
    "  # download wavlm_large_finetune.pth (see seed-tts-eval README) into scoring/checkpoints/\n"
    "  https://github.com/BytedanceSpeech/seed-tts-eval\n"
)


class SimScorer:
    name = "sim"

    def __init__(self, device=None):
        import torch
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._ref_cache = {}

    def _load(self):
        if self._model is None:
            if not (os.path.exists(_CKPT) and os.path.isdir(_UNISPEECH)):
                raise FileNotFoundError(_FETCH_HELP)
            if _UNISPEECH not in sys.path:
                sys.path.insert(0, _UNISPEECH)
            # init_model pulls the wavlm_large upstream via torch.hub.load, which
            # prompts to trust the repo on a TTY and raises EOFError when headless.
            # We vet the pinned UniSpeech/s3prl assets ourselves, so auto-trust here.
            import torch.hub as _hub
            _orig_trust = getattr(_hub, "_check_repo_is_trusted", None)
            if _orig_trust is not None:
                _hub._check_repo_is_trusted = lambda *a, **k: None
            try:
                # seed-tts-eval's verification.init_model builds the wavlm_large SV net.
                from verification import init_model
                model = init_model("wavlm_large", _CKPT)
            finally:
                if _orig_trust is not None:
                    _hub._check_repo_is_trusted = _orig_trust
            self._model = model.to(self.device).eval()
        return self._model

    def _embed(self, wav_path):
        import librosa
        import numpy as np
        model = self._load()
        y, _ = librosa.load(wav_path, sr=16000, mono=True)
        x = self._torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            emb = model(x)
            if isinstance(emb, (tuple, list)):
                emb = emb[-1]
        return self._torch.nn.functional.normalize(emb, dim=-1)

    def _ref_embed(self, ref_path):
        if ref_path not in self._ref_cache:
            self._ref_cache[ref_path] = self._embed(ref_path)
        return self._ref_cache[ref_path]

    def score(self, wav_path, ref_path):
        ref = self._ref_embed(ref_path)
        out = self._embed(wav_path)
        return float((ref * out).sum().item())
