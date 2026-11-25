"""Spanish long-form benchmark: preset-voice models reading four public-domain passages.

Each passage (scoring/longform_es.py, 100-180 words, about 40-75 s of speech) is split with
scoring/longform_chunker.py (<= 240 characters per chunk) and read through the model's runner in
--stdin mode, one warm-up chunk discarded. A chunk whose audio runs over 28 s is split in two and
re-read, because scoring/wer.py hands Whisper only the first 30 s of a file; a chunk that is still over
28 s and cannot be split further is recorded as failed, never scored.

    # synthesize (any interpreter; runners run in their own venvs)
    python scripts/spanish_longform.py --models supertonic,kokoro_mlx --out results/es-longform
    python scripts/spanish_longform.py --models qwentts_06b_custom --voice aiden --passages pd_galdos
    # score (scoring venv: Whisper-large-v3 WER per chunk with Spanish digit spelling, Health per chunk)
    venvs/scoring/bin/python scripts/spanish_longform.py --score results/es-longform/es_longform.csv
    # the markdown table in docs/spanish-longform.md
    python scripts/spanish_longform.py --render-table results/es-longform/es_longform.csv

The manifest is es_longform.csv (not results.csv, so report.py and publish.py never mistake it for a bench
run). It is rewritten after every model: a model that fails to start or times out leaves its failure rows,
and every model that completed keeps its rows. Re-running a model on some passages replaces that model's rows
and WAVs for those passages only.
WER is total edits / total reference words over every scored chunk; RTFx is total audio / total
generation time over the warm chunks.
"""

import argparse
import csv
import json
import os
import platform
import select
import shutil
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
MANIFEST = "es_longform.csv"
META = "es_longform_meta.json"
READY_TIMEOUT_S = 900
CHUNK_TIMEOUT_S = 300

# model -> (venv, runner, fixed args, voice flag, default voice, accepted voices or None = any name).
# voice flag None = the runner's own Spanish preset; only "builtin" is accepted there. A model with a voice flag
# needs an explicit voice name (its runner's no-flag default may be a different, even female, voice).
MODELS = {
    "supertonic":         ("supertonic", "supertonic_runner.py", ["--device", "cpu"], "--reference", "M4",
                           ("M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5")),
    "kokoro_mlx":         ("kokoro_mlx", "kokoro_mlx_runner.py", ["--device", "mps"], "--reference", "em_alex",
                           None),
    "paradee":            ("paradee", "paradee_runner.py", ["--device", "cpu"], None, "builtin", ("builtin",)),
    "reverb":             ("reverb", "reverb_runner.py", ["--device", "cpu"], None, "builtin", ("builtin",)),
    "lfm3_audio":         ("lfm3_audio", "lfm3_audio_runner.py", ["--device", "cpu"], "--reference", "male",
                           ("male", "female")),
    "qwentts_06b_custom": ("qwentts", "qwentts_runner.py", ["--device", "cpu", "--variant", "custom_06b"],
                           "--speaker", "aiden", None),
    "zonos3_mini":        ("zonos3", "zonos3_mini_runner.py", ["--device", "cpu"], None, "builtin", ("builtin",)),
}
FIELDS = ["model", "voice", "argv", "passage_id", "chunk_seq", "chars", "audio_s", "gen_s", "wav", "ok",
          "error", "hyp", "edits", "ref_words", "wer", "health_flags"]


def runner_argv(model, voice):
    if model not in MODELS:
        raise SystemExit(f"unknown model {model!r}; choose from {', '.join(MODELS)}")
    venv, runner, args, flag, _, accepted = MODELS[model]
    if accepted is not None and voice not in accepted:
        raise SystemExit(f"{model}: voice {voice!r} cannot be applied by its runner; accepted: {', '.join(accepted)}")
    if flag and voice == "builtin":
        raise SystemExit(f"{model}: name a voice (its runner takes {flag} <voice>); 'builtin' would record a label "
                         "the runner does not apply")
    py = REPO / "venvs" / venv / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
    argv = [str(py), str(REPO / "runners" / runner), "--stdin", "--language", "es", *args]
    if flag and voice != "builtin":
        argv += [flag, voice]
    return argv


class Runner:
    """One runner subprocess in --stdin mode. Reads stdout through the raw fd with its own line buffer, so a
    deadline is honoured even on a partial line and a reply that arrives in the same write as other output is
    never missed. The child is killed if startup fails."""

    def __init__(self, argv, ready_timeout=READY_TIMEOUT_S):
        self.p = subprocess.Popen(argv, cwd=str(REPO), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        self._fd = self.p.stdout.fileno()
        os.set_blocking(self._fd, False)
        self._buf = b""
        try:
            if not self._read(ready_timeout).get("ready"):
                raise RuntimeError("runner did not report ready")
        except BaseException:
            self.close(graceful=False)
            raise

    def _lines(self):
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            yield line

    def _read(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            for raw in self._lines():
                try:
                    obj = json.loads(raw.decode("utf-8", "replace"))
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and ("ok" in obj or "ready" in obj):
                    return obj
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no runner reply within {timeout}s")
            r, _, _ = select.select([self._fd], [], [], left)
            if not r:
                continue
            try:
                data = os.read(self._fd, 65536)
            except BlockingIOError:
                continue
            if not data:
                raise RuntimeError(f"runner exited rc={self.p.poll()}")
            self._buf += data

    def say(self, text, out, timeout=CHUNK_TIMEOUT_S):
        self.p.stdin.write((json.dumps({"text": text, "out": str(out), "language": "es"}) + "\n").encode("utf-8"))
        self.p.stdin.flush()
        return self._read(timeout)

    def close(self, graceful=True):
        """End the child: EOF on stdin and a short wait when graceful, otherwise (or on a hang) kill it."""
        try:
            if self.p.stdin and not self.p.stdin.closed:
                self.p.stdin.close()
        except OSError:
            pass
        try:
            self.p.wait(timeout=10 if graceful else 0.1)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()


def _opt_name(token, option):
    """True when argv token is `option` itself, `option=value`, or an unambiguous argparse abbreviation of it
    (runners use argparse, which accepts --ref and --reference=x as well as --reference x)."""
    name = token.split("=", 1)[0]
    return name.startswith("--") and len(name) > 2 and option.startswith(name)


def check_argv_voice(model, voice, argv):
    """The recorded voice label must be what the final argv applies (also after --argv-json)."""
    flag = MODELS[model][3]
    tail = argv[1:]
    if flag is None:
        if voice != "builtin":
            raise SystemExit(f"{model}: only the builtin preset can be recorded")
        if any(_opt_name(a, o) for a in tail for o in ("--reference", "--speaker")):
            raise SystemExit(f"{model}: 'builtin' is recorded but the runner argv selects a reference/speaker")
        return
    applied = []
    for i, a in enumerate(tail):
        if not _opt_name(a, flag):
            continue
        if "=" in a:
            applied.append(a.split("=", 1)[1])
        elif i + 1 < len(tail):
            applied.append(tail[i + 1])
    if applied != [voice]:
        raise SystemExit(f"{model}: recorded voice {voice!r} but the runner argv applies {applied or 'no voice'}")


def wav_seconds(path):
    import wave
    with wave.open(str(path)) as w:
        return w.getnframes() / float(w.getframerate())


def synthesize(model, voice, argv, passages, out, ready_timeout=READY_TIMEOUT_S, chunk_timeout=CHUNK_TIMEOUT_S):
    """Rows for one model. Never raises for a runner problem: a startup or reply failure becomes failure rows
    (one per passage not finished) so the other models' results still get written."""
    rows = []
    mdir = out / f"{model}__{voice}"
    run_tag = f"r{time.time_ns()}"  # new takes go in their own directory; old ones stay until main() commits
    base = {"model": model, "voice": voice, "argv": json.dumps(argv[1:])}
    runner = None
    done = set()
    current = None
    try:
        runner = Runner(argv, ready_timeout)
        (out / "_warmup").mkdir(parents=True, exist_ok=True)
        runner.say(WARMUP, out / "_warmup" / f"{model}.wav", chunk_timeout)
        for p in passages:
            current = p["id"]
            pdir = mdir / p["id"] / run_tag
            pdir.mkdir(parents=True, exist_ok=True)
            queue, seq = list(chunk(p["text"])), 0
            while queue:
                text = queue.pop(0)
                wav = pdir / f"{seq:04d}.wav"
                res = runner.say(text, wav, chunk_timeout)
                row = {**base, "passage_id": p["id"], "chunk_seq": seq, "chars": len(text), "wav": str(wav)}
                if res.get("ok"):
                    dur = wav_seconds(wav)
                    if dur > MAX_AUDIO_S:
                        parts = split_in_two(text)
                        if len(parts) == 2:
                            queue = parts + queue
                            continue
                        row.update(ok=False, audio_s=f"{dur:.3f}",
                                   error=f"audio {dur:.1f}s over {MAX_AUDIO_S:.0f}s and the text cannot be split")
                    else:
                        wav.with_suffix(".txt").write_text(text, encoding="utf-8")
                        row.update(ok=True, audio_s=f"{res.get('audio_s', dur):.3f}",
                                   gen_s=f"{res.get('gen_s', 0):.3f}")
                else:
                    row.update(ok=False, error=str(res.get("error", ""))[:300])
                rows.append(row)
                seq += 1
            done.add(p["id"])
    except Exception as e:  # any runner or WAV problem becomes failure rows; earlier models are already saved
        ids = [p["id"] for p in passages]
        for p in passages:
            if p["id"] in done:
                continue
            if current is not None and ids.index(p["id"]) > ids.index(current):
                msg = f"skipped: not attempted after the runner failed on {current} ({type(e).__name__})"
            else:
                msg = f"{type(e).__name__}: {e}"
            rows.append({**base, "passage_id": p["id"],
                         "chunk_seq": len([r for r in rows if r["passage_id"] == p["id"]]),
                         "ok": False, "error": msg[:300]})
    finally:
        if runner is not None:
            runner.close(graceful=not any("TimeoutError" in str(r.get("error", "")) for r in rows))
    return rows


def prune_passage_dirs(out, rows, model, voice, passage_ids):
    """Delete takes of the re-run passages that no manifest row references. Called only after the manifest that
    replaces them has been written, so an interrupted rerun never leaves a saved row pointing at deleted audio."""
    keep = set()
    for r in rows:
        if r.get("wav"):
            w = Path(r["wav"]).resolve()
            keep |= {w, w.with_suffix(".txt")}
    for pid in passage_ids:
        root = out / f"{model}__{voice}" / pid
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir():
                if not any(k.is_relative_to(child.resolve()) for k in keep):
                    shutil.rmtree(child, ignore_errors=True)
            elif child.resolve() not in keep:
                child.unlink(missing_ok=True)


def read_rows(path):
    if not Path(path).exists():
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))


def write_rows(path, rows):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def score(csv_path, wer=None, health=None):
    """Fill WER and Health for every completed chunk. A chunk that cannot be scored keeps an error note and the
    others are still scored; the manifest is written in every case."""
    from scoring.wer import normalize_intl
    import jiwer

    if wer is None:
        from scoring.wer import WerScorer
        wer = WerScorer()
    if health is None:
        from scoring.health import HealthScorer
        health = HealthScorer()
    with open(csv_path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    if header != FIELDS:
        raise SystemExit(f"{csv_path}: not an es-longform manifest (expected columns {FIELDS}); nothing written")
    rows = read_rows(csv_path)
    try:
        for r in rows:
            if r["ok"] != "True":
                continue
            try:
                dur = wav_seconds(r["wav"])
                if dur > MAX_AUDIO_S:  # Whisper sees only the first 30 s: never score a longer take as complete
                    r.update(hyp="", edits="", ref_words="", wer="", health_flags="",
                             error=f"score: audio {dur:.1f}s over {MAX_AUDIO_S:.0f}s, left unscored")
                    continue
                ref = normalize_intl(join_letter_runs(Path(r["wav"]).with_suffix(".txt").read_text(encoding="utf-8")))
                hyp = wer.transcribe(r["wav"], "es")
                m = jiwer.process_words(ref, normalize_intl(join_letter_runs(spell_digits(hyp))))
                edits = m.substitutions + m.deletions + m.insertions
                r.update(hyp=hyp, edits=edits, ref_words=len(ref.split()), error="",
                         wer=f"{edits / max(1, len(ref.split())):.4f}", health_flags=health.score(r["wav"]))
            except Exception as e:  # one bad clip must not cost the rest of the scores
                r.update(hyp="", edits="", ref_words="", wer="", health_flags="",
                         error=f"score: {type(e).__name__}: {e}"[:300])
    finally:
        write_rows(csv_path, rows)


def render_table(csv_path):
    """Markdown table: one row per (model, voice), sorted by WER then RTFx (descending)."""
    agg = {}
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        if "stage" in r and r["stage"] != "2":
            continue
        a = agg.setdefault((r["model"], r["voice"]), {"e": 0, "w": 0, "flag": 0, "audio": 0.0, "gen": 0.0,
                                                      "chunks": 0, "failed": 0})
        a["chunks"] += 1
        if r["ok"] != "True":
            a["failed"] += 1
            continue
        a["audio"] += float(r["audio_s"])  # speed counts every successful synthesis, scored or not
        a["gen"] += float(r["gen_s"])
        if r.get("wer", "") == "":
            a["failed"] += 1  # synthesized but could not be scored: kept out of WER only
            continue
        a["e"] += int(r["edits"])
        a["w"] += int(r["ref_words"])
        a["flag"] += bool(r["health_flags"])
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
    ap.add_argument("--out", default=str(REPO / "results" / "es-longform"))
    ap.add_argument("--score", metavar="CSV", default=None)
    ap.add_argument("--render-table", metavar="CSV", default=None)
    a = ap.parse_args(argv)
    if a.render_table:
        sys.stdout.write(render_table(a.render_table))
        return 0
    if a.score:
        score(a.score)
        return 0
    models = [m for m in a.models.split(",") if m]
    if a.voice and len(models) != 1:
        ap.error("--voice applies to a single model")
    known = {p["id"] for p in PASSAGES}
    wanted = a.passages.split(",") if a.passages else sorted(known)
    if set(wanted) - known:
        ap.error(f"unknown passage ids: {', '.join(sorted(set(wanted) - known))}")
    passages = [p for p in PASSAGES if p["id"] in wanted]
    plan = []
    for model in models:
        voice = a.voice or MODELS.get(model, (None,) * 5)[4]
        argv = runner_argv(model, voice)  # validates model and voice before anything runs
        if a.argv_json:
            argv = [argv[0], *json.loads(a.argv_json)]
        check_argv_voice(model, voice, argv)
        plan.append((model, voice, argv))
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / MANIFEST
    rows = read_rows(manifest)
    for model, voice, argv in plan:
        new = synthesize(model, voice, argv, passages, out)
        rerun = {p["id"] for p in passages}
        rows = [r for r in rows if not ((r["model"], r["voice"]) == (model, voice) and r["passage_id"] in rerun)] + new
        write_rows(manifest, rows)
        prune_passage_dirs(out, rows, model, voice, rerun)
        bad = sum(1 for r in new if str(r.get("ok")) != "True")
        print(f"{model}/{voice}: {len(new)} rows, {bad} failed", flush=True)
    meta = {"set": "es-longform-pd", "rig_os": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "passages": [p["id"] for p in passages]}
    (out / META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
