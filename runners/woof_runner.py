"""Underdog Woof runner (ConwayResearch; pure-numpy runtime, no torch).

Weights + code: ConwayResearch/Underdog-Woof @ 804d7ef48530b37041c3d0b8241a562e04c53043
(pinned below). The repo IS the package: `underdog_woof/` (numpy + stdlib only) sits next
to `woof_int8.npz`, so the runner snapshot-downloads the pinned revision and imports the
package from that snapshot. Nothing is pip-installed and nothing touches the network after
the download (read before running: `np.load` without pickle, no subprocess/urllib).

38.24M params (22.95M speech model + 15.3M Mimi-derived decoder, per the card), int8
weights, 24 kHz mono, English only. Two built-in voices in the one file (juno, mira); the
row uses **juno**, upstream's default and the voice behind the card's speed claim (mira is
their WER-headline voice, ~2x slower per the card).

API (this revision):
    from underdog_woof import Woof
    tts = Woof(voice="juno")          # temp=0.5, eos=0.9, retries=3 defaults
    wav = tts(text, seed=0)           # float32 mono, 24 kHz

Pipeline notes that affect the numbers:
  * A rule-based normalizer rewrites numbers/dates/codes first, then text is split into
    sentences (and >120-char sentences into clauses) and each piece is generated
    separately and concatenated. Prompt 3 is therefore several pieces, not one long take.
  * Each piece is checked for runaway length / >=1 s internal silence and REDRAWN up to
    3 times, keeping the best take. That is upstream's default call path, so it is in
    gen_s: a hard prompt can cost up to 4 generations of a piece. Measured at this
    revision with seed 0: ZERO retries on all four English bench prompts (p1/p2/p4 one
    piece, p3 three pieces, one generation each), so our speed and audio are single
    takes. Upstream's own hard-sentence WER figure may still benefit from the redraws.

Not streaming: the API returns the whole waveform, so ttfa_ms == gen_s (time to the
complete clip), not a streaming first-chunk latency.

No cloning: voices are fixed embeddings in the weights file and no encoder is shipped.
--reference is ignored, so the model has no cloning row.

Device: CPU only (numpy). Threads are left to numpy's BLAS default; the card's 13.8x is
on ONE thread of an M5 Max, which is not what this measures.

Determinism: the runtime's RNG is a module-global torch-compatible mt19937; we pass
seed=SEED on every call (upstream CLI default 0), so every run re-seeds and reruns of a
prompt produce identical audio.
"""

import argparse
import json
import sys
import time

import _meminfo

REPO = "ConwayResearch/Underdog-Woof"
REVISION = "804d7ef48530b37041c3d0b8241a562e04c53043"
VOICE = "juno"
SEED = 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Ignored -- Woof has two fixed voices and no voice encoder.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1
    if args.device != "cpu":
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"woof has no device={args.device} (numpy CPU runtime)"}))
        return 1
    if args.language != "en":
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"woof is English-only (language={args.language})"}))
        return 1
    try:
        from huggingface_hub import snapshot_download

        d = snapshot_download(REPO, revision=REVISION, allow_patterns=[
            "underdog_woof/*", "woof_int8.npz"])
        sys.path.insert(0, d)
        from underdog_woof import Woof

        tts = Woof(voice=VOICE)
        sample_rate = tts.sr
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            wav = tts(text, seed=SEED)
            t_end = time.perf_counter()
            if write_wav:
                tts.save(out_path, wav)
            gen_s = t_end - t0
            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": gen_s * 1000,
                "gen_s": gen_s, "audio_s": float(len(wav) / sample_rate),
                **_meminfo.sample(args.device),
            }), flush=True)
            return True
        except Exception as e:
            print(json.dumps({"ok": False, "run_index": run_index,
                              "error": f"{type(e).__name__}: {e}"}), flush=True)
            return False

    if args.stdin:
        idx = 0
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                job = json.loads(line)
            except json.JSONDecodeError as e:
                print(json.dumps({"ok": False, "run_index": idx,
                                  "error": f"json parse: {e}"}), flush=True)
                idx += 1
                continue
            _one(job["text"], job["out"], idx, write_wav=True)
            idx += 1
        return 0

    for i in range(args.runs):
        if not _one(args.text, args.out, i, write_wav=(i == 0)):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
