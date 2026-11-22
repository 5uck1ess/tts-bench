"""Spanish long-form benchmark: preset-voice models reading four public-domain passages.

Each passage (scoring/longform_es.py, 100-180 words, a minute or more of speech) is split with
scoring/longform_chunker.py (<= 240 characters per chunk) and read through the model's runner in
--stdin mode, one warm-up chunk discarded. A chunk whose audio runs over 28 s is split in two and
re-read, because scoring/wer.py hands Whisper only the first 30 s of a file.

    # synthesize (any interpreter; runners run in their own venvs)
    python scripts/spanish_longform.py --models supertonic,kokoro_mlx --out results/es-longform
    python scripts/spanish_longform.py --models qwentts_06b_custom --voice aiden --passages pd_galdos
    # score (scoring venv: Whisper-large-v3 WER per chunk with Spanish digit spelling, Health per chunk)
    venvs/scoring/bin/python scripts/spanish_longform.py --score results/es-longform/results.csv
    # the markdown table in docs/spanish-longform.md
    python scripts/spanish_longform.py --render-table results/es-longform/results.csv

WER is total edits / total reference words over every chunk. RTFx is total audio / total generation
time over the warm chunks. results.csv columns: model, voice, argv, passage_id, chunk_seq, chars,
audio_s, gen_s, wav, ok, error, hyp, edits, ref_words, wer, health_flags.
"""

import argparse
import csv
import json
import os
import platform
import select
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scoring.longform_chunker import chunk, split_in_two  # noqa: E402
from scoring.longform_es import PASSAGES, join_letter_runs, spell_digits  # noqa: E402

MAX_AUDIO_S = 28.0
WARMUP = "Esta es una frase de calentamiento para cargar el modelo."

# Preset-voice runners that speak Spanish. voice_flag None = the runner's own Spanish preset.
MODELS = {
    "supertonic":         ("supertonic", "supertonic_runner.py", ["--device", "cpu"], "--reference", "M4"),
    "kokoro_mlx":         ("kokoro_mlx", "kokoro_mlx_runner.py", ["--device", "mps"], "--reference", "em_alex"),
    "paradee":            ("paradee", "paradee_runner.py", ["--device", "cpu"], None, "builtin"),
    "reverb":             ("reverb", "reverb_runner.py", ["--device", "cpu"], None, "builtin"),
    "lfm3_audio":         ("lfm3_audio", "lfm3_audio_runner.py", ["--device", "cpu"], "--reference", "male"),
    "qwentts_06b_custom": ("qwentts", "qwentts_runner.py", ["--device", "cpu", "--variant", "custom_06b"],
                           "--speaker", "aiden"),
    "zonos3_mini":        ("zonos3", "zonos3_mini_runner.py", ["--device", "cpu"], None, "builtin"),
}
FIELDS = ["model", "voice", "argv", "passage_id", "chunk_seq", "chars", "audio_s", "gen_s", "wav", "ok",
          "error", "hyp", "edits", "ref_words", "wer", "health_flags"]


def runner_argv(model, voice):
    venv, runner, args, flag, _ = MODELS[model]
    py = REPO / "venvs" / venv / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
    argv = [str(py), str(REPO / "runners" / runner), "--stdin", "--language", "es", *args]
    if flag and voice != "builtin":
        argv += [flag, voice]
    return argv


class Runner:
    def __init__(self, argv, ready_timeout=900):
        self.p = subprocess.Popen(argv, cwd=str(REPO), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                  env=dict(os.environ, PYTHONUNBUFFERED="1"))
        if not self._read(ready_timeout).get("ready"):
            raise RuntimeError("runner did not report ready")

    def _read(self, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            r, _, _ = select.select([self.p.stdout], [], [], max(0.1, deadline - time.time()))
            if not r:
                continue
            line = self.p.stdout.readline()
            if line == "":
                raise RuntimeError(f"runner exited rc={self.p.poll()}")
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "ok" in obj or "ready" in obj:
                return obj
        raise TimeoutError(f"no runner reply within {timeout}s")

    def say(self, text, out, timeout=300):
        self.p.stdin.write(json.dumps({"text": text, "out": str(out), "language": "es"}) + "\n")
        self.p.stdin.flush()
        return self._read(timeout)

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


def wav_seconds(path):
    import wave
    with wave.open(str(path)) as w:
        return w.getnframes() / float(w.getframerate())


def synthesize(model, voice, argv, passages, out):
    rows = []
    runner = Runner(argv)
    try:
        (out / "_warmup").mkdir(parents=True, exist_ok=True)
        runner.say(WARMUP, out / "_warmup" / f"{model}.wav")
        for p in passages:
            pdir = out / f"{model}__{voice}" / p["id"]
            pdir.mkdir(parents=True, exist_ok=True)
            queue, seq = list(chunk(p["text"])), 0
            while queue:
                text = queue.pop(0)
                wav = pdir / f"{seq:04d}.wav"
                res = runner.say(text, wav)
                row = {"model": model, "voice": voice, "argv": json.dumps(argv[1:]), "passage_id": p["id"],
                       "chunk_seq": seq, "chars": len(text), "wav": str(wav)}
                if res.get("ok"):
                    if wav_seconds(wav) > MAX_AUDIO_S and len(text.split()) > 3:
                        queue = split_in_two(text) + queue
                        continue
                    wav.with_suffix(".txt").write_text(text, encoding="utf-8")
                    row.update(ok=True, audio_s=f"{res.get('audio_s', 0):.3f}", gen_s=f"{res.get('gen_s', 0):.3f}")
                else:
                    row.update(ok=False, error=str(res.get("error", ""))[:300])
                rows.append(row)
                seq += 1
    finally:
        runner.close()
    return rows


def score(csv_path):
    from scoring.health import HealthScorer
    from scoring.wer import WerScorer, normalize_intl
    import jiwer

    wer, health = WerScorer(), HealthScorer()
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    for r in rows:
        if r["ok"] != "True":
            continue
        ref = normalize_intl(join_letter_runs(Path(r["wav"]).with_suffix(".txt").read_text(encoding="utf-8")))
        hyp = wer.transcribe(r["wav"], "es")
        m = jiwer.process_words(ref, normalize_intl(join_letter_runs(spell_digits(hyp))))
        edits = m.substitutions + m.deletions + m.insertions
        r.update(hyp=hyp, edits=edits, ref_words=len(ref.split()),
                 wer=f"{edits / max(1, len(ref.split())):.4f}", health_flags=health.score(r["wav"]))
    write_rows(csv_path, rows)


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def render_table(csv_path):
    """Markdown table: one row per (model, voice), sorted by WER then RTFx (descending)."""
    agg = {}
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        if "stage" in r and r["stage"] != "2":
            continue
        a = agg.setdefault((r["model"], r["voice"]), {"e": 0, "w": 0, "flag": 0, "audio": 0.0, "gen": 0.0,
                                                      "chunks": 0, "failed": 0})
        a["chunks"] += 1
        if r["ok"] != "True" or r.get("wer", "") == "":
            a["failed"] += 1
            continue
        a["e"] += int(r["edits"])
        a["w"] += int(r["ref_words"])
        a["flag"] += bool(r["health_flags"])
        a["audio"] += float(r["audio_s"])
        a["gen"] += float(r["gen_s"])
    lines = ["| model | voice | chunks | failed | WER | Health-flagged chunks | audio (s) | RTFx |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    def key(kv):
        a = kv[1]
        return (a["e"] / a["w"] if a["w"] else 9, -(a["audio"] / a["gen"] if a["gen"] else 0))
    for (m, v), a in sorted(agg.items(), key=key):
        wer = f"{a['e'] / a['w']:.4f}" if a["w"] else "n/a"
        rtfx = f"{a['audio'] / a['gen']:.2f}" if a["gen"] else "n/a"
        lines.append(f"| {m} | {v} | {a['chunks']} | {a['failed']} | {wer} | {a['flag']} | {a['audio']:.1f} | {rtfx} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--voice", default=None, help="Override the model's default voice (single model only).")
    ap.add_argument("--argv-json", default=None, help="Exact runner argv (JSON list, without the interpreter).")
    ap.add_argument("--passages", default=None, help="Comma-separated passage ids (default: all four).")
    ap.add_argument("--out", default=str(REPO / "results" / f"{time.strftime('%Y-%m-%d')}_es-longform"))
    ap.add_argument("--score", metavar="CSV", default=None)
    ap.add_argument("--render-table", metavar="CSV", default=None)
    a = ap.parse_args(argv)
    if a.render_table:
        sys.stdout.write(render_table(a.render_table))
        return 0
    if a.score:
        score(a.score)
        return 0
    passages = [p for p in PASSAGES if not a.passages or p["id"] in a.passages.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in a.models.split(","):
        voice = a.voice or MODELS[model][4]
        argv = runner_argv(model, voice)
        if a.argv_json:
            argv = [argv[0], *json.loads(a.argv_json)]
        rows += synthesize(model, voice, argv, passages, out)
    write_rows(out / "results.csv", rows)
    meta = {"set": "es-longform-pd", "rig_os": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "passages": [p["id"] for p in passages]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
