"""UTMOS naturalness scorer — SpeechMOS utmos22_strong via torch.hub.

Reference-free learned MOS predictor. Higher = more natural (≈1..5).
Applies to every clip (no reference, no text).
"""

import librosa
import numpy as np
import torch


class UtmosScorer:
    name = "utmos"

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def _load(self):
        if self._model is None:
            # Auto-downloads on first use; cached under torch hub dir thereafter.
            self._model = torch.hub.load(
                "tarepan/SpeechMOS", "utmos22_strong", trust_repo=True
            ).to(self.device).eval()
        return self._model

    def score(self, wav_path):
        model = self._load()
        wave, _ = librosa.load(wav_path, sr=16000, mono=True)
        x = torch.from_numpy(np.asarray(wave, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(model(x, 16000).item())
