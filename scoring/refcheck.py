"""Reference-transcript check: does each reference/<stem>.txt match its .wav?

Why this exists: reference/chris_hemsworth_15s.txt was missing the clip's last
sentence from May to 2026-10-20. Transcript-conditioned cloners treat the
unspoken tail as the start of the target and skip the first ~3 s of every
prompt, and nothing compared the file with its audio (docs/known-issues.md,
Cross-cutting). bench.py runs this before every pass and refuses to start on a
mismatch.

Method: Whisper-large-v3 (the same model as scoring/wer.py) transcribes the wav
in <=28 s windows, so references longer than Whisper's 30 s window are covered.
A pair FAILS if either
  - WER(txt, asr) > MAX_WER (wording drift, e.g. a wrong sentence), or
  - the word counts differ by more than MAX_LEN_DRIFT (a missing or extra
    sentence; the 2026-10 bug was 38 txt words vs 49 spoken, 22% short).
Spelling-level differences ("Neuphonic"/"Nufonic") stay well under both.

Results are cached in reference/.refcheck.json keyed by sha1(wav)+sha1(txt),
so a pass only costs time when a file changes.

    python -m scoring.refcheck                  # every reference/*.wav with a .txt
    python -m scoring.refcheck reference/x.wav  # specific files
Exit 0 = all match, 1 = mismatch, 2 = could not run.
"""

import hashlib
import json
import sys
from pathlib import Path

MAX_WER = 0.15
MAX_LEN_DRIFT = 0.12
WINDOW_S = 28.0

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "reference" / ".refcheck.json"


def _sha1(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def _transcribe_long(scorer, wav: Path, lang: str) -> str:
    import librosa
    import numpy as np
    proc, model = scorer._load()
    y, _ = librosa.load(str(wav), sr=16000, mono=True)
    step = int(WINDOW_S * 16000)
    parts = []
    for i in range(0, len(y), step):
        chunk = np.asarray(y[i:i + step], dtype=np.float32)
        if len(chunk) < 1600:  # <0.1 s tail
            continue
        feats = proc(chunk, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(scorer.device, scorer.dtype)
        with scorer._torch.no_grad():
            ids = model.generate(feats, task="transcribe",
                                 language={"fr": "french", "es": "spanish"}.get(lang, "english"))
        parts.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return " ".join(parts)


def _guess_lang(scorer, wav: Path) -> str:
    # Language ID on the first window; the bench only uses en/fr/es references.
    import librosa
    import numpy as np
    proc, model = scorer._load()
    y, _ = librosa.load(str(wav), sr=16000, mono=True, duration=WINDOW_S)
    feats = proc(np.asarray(y, dtype=np.float32), sampling_rate=16000,
                 return_tensors="pt").input_features.to(scorer.device, scorer.dtype)
    with scorer._torch.no_grad():
        lang_id = model.detect_language(feats)
    token = proc.tokenizer.convert_ids_to_tokens(int(lang_id[0]))  # e.g. "<|fr|>"
    code = token.strip("<|>")
    return code if code in ("fr", "es") else "en"


def check(paths):
    """Return a list of (wav, ok, detail) for each wav that has a sibling .txt."""
    from scoring.wer import WerScorer, wer_value
    cache = {}
    if CACHE.exists():
        try:
            cache = json.loads(CACHE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
    scorer = None
    results = []
    for wav in paths:
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        key = f"{wav.name}:{_sha1(wav)}:{_sha1(txt)}"
        hit = cache.get(key)
        if hit is None:
            scorer = scorer or WerScorer()
            lang = _guess_lang(scorer, wav)
            hyp = _transcribe_long(scorer, wav, lang)
            ref_n = scorer._normalize(txt.read_text(encoding="utf-8"), lang)
            hyp_n = scorer._normalize(hyp, lang)
            w = wer_value(ref_n, hyp_n)
            nr, nh = len(ref_n.split()), len(hyp_n.split())
            drift = abs(nr - nh) / max(nh, 1)
            hit = {"lang": lang, "wer": round(w if w is not None else 1.0, 3),
                   "txt_words": nr, "asr_words": nh, "len_drift": round(drift, 3),
                   "asr": hyp}
            hit["ok"] = hit["wer"] <= MAX_WER and drift <= MAX_LEN_DRIFT
            cache[key] = hit
        detail = (f"WER {hit['wer']:.3f}, words txt/asr {hit['txt_words']}/"
                  f"{hit['asr_words']} (drift {hit['len_drift']:.0%})")
        if not hit["ok"]:
            detail += f"\n      audio says: {hit['asr']}"
        results.append((wav, hit["ok"], detail))
    try:
        CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return results


def main(argv) -> int:
    paths = ([Path(a).resolve() for a in argv] if argv
             else sorted((REPO / "reference").glob("*.wav")))
    try:
        results = check(paths)
    except Exception as e:  # noqa: BLE001 -- report, let the caller decide
        print(f"refcheck: could not run: {type(e).__name__}: {e}")
        return 2
    bad = 0
    for wav, ok, detail in results:
        print(f"refcheck: {'ok  ' if ok else 'FAIL'} {wav.name}: {detail}")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
